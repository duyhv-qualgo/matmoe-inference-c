#!/usr/bin/env python3
"""
export_tflite_dense.py — v7.2.6 Dense model: MatMoE JAX → TFLite (fast decoder)

Two TFLite models:

  tflite_dense/encode_prefill.tflite
      inputs : source_ids [1,128] int32, src_mask [1,128] int32
      outputs: cross_kv [8,2,1,128,8,64] float32, direction_ids [1] int32

  tflite_dense/decode_step.tflite
      inputs : token_id [1,1] int32, step [1] int32,
               self_kv [8,2,1,128,8,64] float32,
               cross_kv [8,2,1,128,8,64] float32,
               src_mask [1,128] int32, direction_ids [1] int32
      outputs: logits [1,VOCAB] float32,
               self_kv_next [8,2,1,128,8,64] float32

Usage:
    cd export/
    python export_tflite_dense.py \\
        --model-msg ../../v7.2.6_step80000/k0_dense_256.msg \\
        --tokenizer-dir ../../v7.2.2_step80000/t5-en-vi-tokenizer-20k-v3 \\
        --mlp-dim 256 \\
        [--out-dir tflite_dense] \\
        [--quantize none|float16|dynamic] \\
        [--skip-check]
"""

import argparse
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

import tensorflow as tf
from jax.experimental import jax2tf

from release_utils import load_tokenizer, load_msgpack_model
from config import config as base_config, MoEModelConfig
from dense_model import DenseTranslationModel, rotate_half


# ─── Constants ────────────────────────────────────────────────────────────────

MAX_SRC_LEN = 128
MAX_TGT_LEN = 128
D_MODEL     = 512
NUM_LAYERS  = 8
NUM_HEADS   = 8
HEAD_DIM    = D_MODEL // NUM_HEADS  # 64

SANITY_SENTENCES = [
    "<translate-en-vi> Hello, how are you today?",
    "<translate-vi-en> Xin chào, bạn có khỏe không?",
    "<translate-en-vi> The weather is beautiful today.",
    "<translate-en-vi> Artificial intelligence is transforming the world.",
]


# ─── Model builder ────────────────────────────────────────────────────────────

def build_dense_model(tokenizer, mlp_dim: int):
    model_cfg = MoEModelConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        vi_en_token_id=tokenizer.convert_tokens_to_ids("<translate-vi-en>"),
        d_model=base_config.d_model,
        num_heads=base_config.num_heads,
        mlp_dim=mlp_dim,
        num_layers=base_config.num_layers,
        num_experts=1,  # unused by DenseTranslationModel
        top_k=1,        # unused by DenseTranslationModel
        semantic_dim=base_config.semantic_dim,
        dropout_rate=0.0,
        max_seq_len=base_config.max_length_inference,
        dtype=jnp.bfloat16,
    )
    return DenseTranslationModel(model_cfg, rngs=nnx.Rngs(base_config.seed))


# ─── JAX forward functions ────────────────────────────────────────────────────

def make_encode_prefill_fn(model):
    """
    (source_ids, src_mask) → (cross_kv, direction_ids)

    Runs the encoder, then projects encoder output through each decoder
    layer's cross-attention key/value projections (+ k_norm) and stacks
    them into a single cross_kv tensor.

    cross_kv shape: [NUM_LAYERS, 2, 1, MAX_SRC_LEN, NUM_HEADS, HEAD_DIM]
    """
    vi_en_token_id = int(model.cfg.vi_en_token_id)

    def encode_prefill_fn(source_ids, src_mask):
        direction_ids = (source_ids[:, 0] == vi_en_token_id).astype(jnp.int32)

        enc_out = model.encode(source_ids, src_mask, deterministic=True)
        enc_out = enc_out.astype(jnp.float32)  # [1, 128, 512]

        cross_kv_list = []
        for block in model.decoder.blocks:
            ca = block.cross_attn
            k = ca.key(enc_out).reshape(1, MAX_SRC_LEN, NUM_HEADS, HEAD_DIM)
            v = ca.value(enc_out).reshape(1, MAX_SRC_LEN, NUM_HEADS, HEAD_DIM)
            k = ca.k_norm(k)
            kv = jnp.stack([k, v], axis=0)  # [2, 1, 128, 8, 64]
            cross_kv_list.append(kv)

        cross_kv = jnp.stack(cross_kv_list, axis=0)  # [8, 2, 1, 128, 8, 64]
        return cross_kv, direction_ids

    return encode_prefill_fn


def make_decode_step_fn(model):
    """
    Single-token decode step with self-attention KV cache.

    Inputs:
      token_id      [1, 1]                  int32
      step          [1]                     int32
      self_kv       [8, 2, 1, 128, 8, 64]  float32
      cross_kv      [8, 2, 1, 128, 8, 64]  float32
      src_mask      [1, 128]                int32
      direction_ids [1]                     int32

    Outputs:
      logits        [1, vocab_size]         float32
      self_kv_next  [8, 2, 1, 128, 8, 64]  float32
    """
    d_model = model.cfg.d_model

    def decode_step_fn(token_id, step, self_kv, cross_kv, src_mask, direction_ids):
        # ── 1. Embed current token ──────────────────────────────────────────
        x = model.embed_norm(model.embedding(token_id))  # [1, 1, 512]

        # ── 2. RoPE sin/cos at position `step` ─────────────────────────────
        sin, cos = model._get_rope(step)  # each [1, 1, 1, HEAD_DIM]

        # ── 3. Decoder blocks ───────────────────────────────────────────────
        new_self_kv_list = []

        for layer_idx, block in enumerate(model.decoder.blocks):

            # ── Self-attention ──────────────────────────────────────────────
            xn = block.ln_1(x)

            q_s = block.self_attn.query(xn).reshape(1, 1, NUM_HEADS, HEAD_DIM)
            k_s = block.self_attn.key(xn).reshape(1, 1, NUM_HEADS, HEAD_DIM)
            v_s = block.self_attn.value(xn).reshape(1, 1, NUM_HEADS, HEAD_DIM)

            q_s = block.self_attn.q_norm(q_s)
            k_s = block.self_attn.k_norm(k_s)

            q_s = (q_s * cos) + (rotate_half(q_s) * sin)
            k_s = (k_s * cos) + (rotate_half(k_s) * sin)

            k_cache = self_kv[layer_idx, 0]  # [1, MAX_TGT_LEN, 8, 64]
            v_cache = self_kv[layer_idx, 1]

            k_cache = jax.lax.dynamic_update_slice(k_cache, k_s, [0, step[0], 0, 0])
            v_cache = jax.lax.dynamic_update_slice(v_cache, v_s, [0, step[0], 0, 0])

            new_self_kv_list.append(jnp.stack([k_cache, v_cache], axis=0))

            self_scores = (jnp.einsum('bqhd,bkhd->bhqk', q_s, k_cache)
                           / jnp.sqrt(jnp.float32(HEAD_DIM)))

            positions = jnp.arange(MAX_TGT_LEN, dtype=jnp.int32)
            causal_mask = (positions[None, :] <= step)[:, None, None, :]
            self_scores = jnp.where(causal_mask, self_scores,
                                    jnp.array(-1e9, dtype=self_scores.dtype))

            self_probs = jax.nn.softmax(
                self_scores.astype(jnp.float32), axis=-1).astype(self_scores.dtype)

            self_out = jnp.einsum('bhqk,bkhd->bqhd', self_probs, v_cache)
            self_out = self_out.reshape(1, 1, d_model)
            self_out = block.self_attn.out(self_out)

            x = x + self_out

            # ── Cross-attention ─────────────────────────────────────────────
            xn = block.ln_2(x)

            q_c = block.cross_attn.query(xn).reshape(1, 1, NUM_HEADS, HEAD_DIM)
            q_c = block.cross_attn.q_norm(q_c)
            # No RoPE on cross-attention

            k_c = cross_kv[layer_idx, 0]  # [1, 128, 8, 64]  (k_norm already applied)
            v_c = cross_kv[layer_idx, 1]

            cross_scores = (jnp.einsum('bqhd,bkhd->bhqk', q_c, k_c)
                            / jnp.sqrt(jnp.float32(HEAD_DIM)))

            src_mask_4d = src_mask[:, None, None, :]
            cross_scores = jnp.where(src_mask_4d == 1, cross_scores,
                                     jnp.array(-1e9, dtype=cross_scores.dtype))

            cross_probs = jax.nn.softmax(
                cross_scores.astype(jnp.float32), axis=-1).astype(cross_scores.dtype)

            cross_out = jnp.einsum('bhqk,bkhd->bqhd', cross_probs, v_c)
            cross_out = cross_out.reshape(1, 1, d_model)
            cross_out = block.cross_attn.out(cross_out)

            x = x + cross_out

            # ── Dense FFN ───────────────────────────────────────────────────
            ffn_out, _, _ = block.moe(block.ln_3(x), direction_ids, deterministic=True)
            x = x + ffn_out

        # ── 4. Final norm + logits ──────────────────────────────────────────
        x = model.decoder.ln_final(x)

        x_fp32   = x[:, 0, :].astype(jnp.float32)
        emb_fp32 = jnp.asarray(model.embedding.embedding, jnp.float32)
        logits   = jnp.dot(x_fp32, emb_fp32.T)
        logits   = logits * (d_model ** -0.5)
        cap      = jnp.float32(30.0)
        logits   = cap * jnp.tanh(logits / cap)

        # ── 5. Stack updated self-KV cache ──────────────────────────────────
        self_kv_next = jnp.stack(new_self_kv_list, axis=0)

        return logits, self_kv_next

    return decode_step_fn


# ─── TF Module wrappers ───────────────────────────────────────────────────────

def build_encode_prefill_module(tf_fn):
    class EncodePrefillModule(tf.Module):
        @tf.function(input_signature=[
            tf.TensorSpec([1, MAX_SRC_LEN], tf.int32, name="source_ids"),
            tf.TensorSpec([1, MAX_SRC_LEN], tf.int32, name="src_mask"),
        ])
        def __call__(self, source_ids, src_mask):
            cross_kv, direction_ids = tf_fn(source_ids, src_mask)
            return cross_kv, direction_ids
    return EncodePrefillModule()


def build_decode_step_module(tf_fn, vocab_size):
    class DecodeStepModule(tf.Module):
        @tf.function(input_signature=[
            tf.TensorSpec([1, 1],                                                  tf.int32,   name="token_id"),
            tf.TensorSpec([1],                                                     tf.int32,   name="step"),
            tf.TensorSpec([NUM_LAYERS, 2, 1, MAX_TGT_LEN, NUM_HEADS, HEAD_DIM],   tf.float32, name="self_kv"),
            tf.TensorSpec([NUM_LAYERS, 2, 1, MAX_SRC_LEN, NUM_HEADS, HEAD_DIM],   tf.float32, name="cross_kv"),
            tf.TensorSpec([1, MAX_SRC_LEN],                                        tf.int32,   name="src_mask"),
            tf.TensorSpec([1],                                                     tf.int32,   name="direction_ids"),
        ])
        def __call__(self, token_id, step, self_kv, cross_kv, src_mask, direction_ids):
            logits, self_kv_next = tf_fn(token_id, step, self_kv, cross_kv, src_mask, direction_ids)
            return logits, self_kv_next
    return DecodeStepModule()


# ─── TFLite conversion ────────────────────────────────────────────────────────

def to_tflite(savedmodel_path, out_path, quantize="none"):
    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if quantize == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_bytes = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_bytes)
    return tflite_bytes


# ─── TFLite helpers ───────────────────────────────────────────────────────────

def _build_ep_index(ep_interp):
    cross_kv_idx = dir_ids_idx = None
    for t in ep_interp.get_output_details():
        if len(t["shape"]) == 6:
            cross_kv_idx = t["index"]
        else:
            dir_ids_idx = t["index"]
    assert cross_kv_idx is not None and dir_ids_idx is not None
    return cross_kv_idx, dir_ids_idx


def _build_ds_index(ds_interp):
    token_idx = step_idx = self_kv_idx = cross_kv_idx = None
    src_mask_idx = dir_ids_idx = logits_idx = self_kv_out_idx = None

    ambiguous_1d = []
    for t in ds_interp.get_input_details():
        sh   = tuple(t["shape"])
        name = t["name"].lower()
        if sh == (1, 1):
            token_idx = t["index"]
        elif len(sh) == 6 and sh[0] == NUM_LAYERS:
            if "cross" in name:
                cross_kv_idx = t["index"]
            else:
                self_kv_idx = t["index"]
        elif sh == (1, MAX_SRC_LEN):
            src_mask_idx = t["index"]
        elif sh == (1,):
            ambiguous_1d.append(t)

    for t in ambiguous_1d:
        name = t["name"].lower()
        if "direction" in name:
            dir_ids_idx = t["index"]
        elif "step" in name:
            step_idx = t["index"]

    if (step_idx is None or dir_ids_idx is None) and len(ambiguous_1d) == 2:
        ambiguous_1d.sort(key=lambda t: t["index"])
        step_idx    = ambiguous_1d[0]["index"]
        dir_ids_idx = ambiguous_1d[1]["index"]

    for t in ds_interp.get_output_details():
        sh = tuple(t["shape"])
        if len(sh) == 2:
            logits_idx = t["index"]
        else:
            self_kv_out_idx = t["index"]

    assert all(v is not None for v in [
        token_idx, step_idx, self_kv_idx, cross_kv_idx,
        src_mask_idx, dir_ids_idx, logits_idx, self_kv_out_idx
    ]), (f"Could not identify all decode_step tensors.\n"
         f"  inputs:  {[(t['name'], t['shape']) for t in ds_interp.get_input_details()]}\n"
         f"  outputs: {[(t['name'], t['shape']) for t in ds_interp.get_output_details()]}")

    return dict(
        token=token_idx, step=step_idx,
        self_kv=self_kv_idx, cross_kv=cross_kv_idx,
        src_mask=src_mask_idx, dir_ids=dir_ids_idx,
        logits=logits_idx, self_kv_out=self_kv_out_idx,
    )


# ─── JAX KV-cache decode (reference) ─────────────────────────────────────────

def jax_kvcache_decode(model, tokenizer, src_ids, src_mask):
    pad_id = tokenizer.pad_token_id or 0
    eos_id = tokenizer.eos_token_id or 1

    encode_prefill_jax = make_encode_prefill_fn(model)
    decode_step_jax    = make_decode_step_fn(model)

    cross_kv, direction_ids = encode_prefill_jax(
        jnp.array(src_ids), jnp.array(src_mask))

    self_kv = jnp.zeros((NUM_LAYERS, 2, 1, MAX_TGT_LEN, NUM_HEADS, HEAD_DIM),
                         dtype=jnp.float32)

    token  = jnp.array([[pad_id]], dtype=jnp.int32)
    tokens = []
    for s in range(MAX_TGT_LEN):
        step = jnp.array([s], dtype=jnp.int32)
        logits, self_kv = decode_step_jax(
            token, step, self_kv, cross_kv,
            jnp.array(src_mask, dtype=jnp.int32), direction_ids)
        next_tok = int(jnp.argmax(logits[0]))
        tokens.append(next_tok)
        token = jnp.array([[next_tok]], dtype=jnp.int32)
        if next_tok == eos_id:
            break

    return np.array(tokens)


# ─── TFLite KV-cache decode ───────────────────────────────────────────────────

def tflite_kvcache_decode(ep_interp, ds_interp, ep_idx, ds_idx,
                           src_ids, src_mask, tokenizer):
    pad_id = tokenizer.pad_token_id or 0
    eos_id = tokenizer.eos_token_id or 1

    ep_src_idx = ep_msk_idx = None
    for t in ep_interp.get_input_details():
        if "source_ids" in t["name"]:
            ep_src_idx = t["index"]
        elif "src_mask" in t["name"]:
            ep_msk_idx = t["index"]
    if ep_src_idx is None or ep_msk_idx is None:
        ep_ins = ep_interp.get_input_details()
        ep_src_idx, ep_msk_idx = ep_ins[0]["index"], ep_ins[1]["index"]

    ep_interp.set_tensor(ep_src_idx, src_ids)
    ep_interp.set_tensor(ep_msk_idx, src_mask)
    ep_interp.invoke()

    cross_kv = ep_interp.get_tensor(ep_idx["cross_kv"]).copy()
    dir_ids  = ep_interp.get_tensor(ep_idx["dir_ids"]).copy()

    self_kv = np.zeros(
        (NUM_LAYERS, 2, 1, MAX_TGT_LEN, NUM_HEADS, HEAD_DIM), dtype=np.float32)

    token  = np.array([[pad_id]], dtype=np.int32)
    tokens = []
    for s in range(MAX_TGT_LEN):
        step = np.array([s], dtype=np.int32)

        ds_interp.set_tensor(ds_idx["token"],    token)
        ds_interp.set_tensor(ds_idx["step"],     step)
        ds_interp.set_tensor(ds_idx["self_kv"],  self_kv)
        ds_interp.set_tensor(ds_idx["cross_kv"], cross_kv)
        ds_interp.set_tensor(ds_idx["src_mask"], src_mask)
        ds_interp.set_tensor(ds_idx["dir_ids"],  dir_ids)
        ds_interp.invoke()

        logits  = ds_interp.get_tensor(ds_idx["logits"])
        self_kv = ds_interp.get_tensor(ds_idx["self_kv_out"]).copy()

        next_tok = int(np.argmax(logits[0]))
        tokens.append(next_tok)
        token = np.array([[next_tok]], dtype=np.int32)
        if next_tok == eos_id:
            break

    return np.array(tokens)


# ─── Sanity check ─────────────────────────────────────────────────────────────

def sanity_check(model, tokenizer, ep_interp, ep_idx, ds_interp, ds_idx):
    print("\n" + "=" * 70)
    print("SANITY CHECK: JAX KV-cache  vs  TFLite KV-cache")
    print("=" * 70)

    for text in SANITY_SENTENCES:
        enc = tokenizer([text], padding="max_length", truncation=True,
                        max_length=MAX_SRC_LEN, return_tensors="np")
        src_ids  = enc.input_ids.astype(np.int32)
        src_mask = enc.attention_mask.astype(np.int32)

        kv_tokens  = jax_kvcache_decode(model, tokenizer, src_ids, src_mask)
        tfl_tokens = tflite_kvcache_decode(
            ep_interp, ds_interp, ep_idx, ds_idx, src_ids, src_mask, tokenizer)

        kv_text  = tokenizer.decode(kv_tokens,  skip_special_tokens=True)
        tfl_text = tokenizer.decode(tfl_tokens, skip_special_tokens=True)

        n = min(len(kv_tokens), len(tfl_tokens))
        match = float(np.mean(kv_tokens[:n] == tfl_tokens[:n]))

        print(f"\nInput    : {text}")
        print(f"JAX kv   : {kv_text}")
        print(f"TFLite kv: {tfl_text}  (tok_match: {match:.2%})")

    print("\n" + "=" * 70)


# ─── Speed benchmark ──────────────────────────────────────────────────────────

def benchmark_speed(ep_interp, ep_idx, ds_interp, ds_idx, tokenizer, n_steps=20):
    text = SANITY_SENTENCES[0]
    enc = tokenizer([text], padding="max_length", truncation=True,
                    max_length=MAX_SRC_LEN, return_tensors="np")
    src_ids  = enc.input_ids.astype(np.int32)
    src_mask = enc.attention_mask.astype(np.int32)

    ep_src_idx = ep_msk_idx = None
    for t in ep_interp.get_input_details():
        if "source_ids" in t["name"]:
            ep_src_idx = t["index"]
        elif "src_mask" in t["name"]:
            ep_msk_idx = t["index"]
    if ep_src_idx is None or ep_msk_idx is None:
        ep_ins = ep_interp.get_input_details()
        ep_src_idx, ep_msk_idx = ep_ins[0]["index"], ep_ins[1]["index"]

    # Warm up
    ep_interp.set_tensor(ep_src_idx, src_ids)
    ep_interp.set_tensor(ep_msk_idx, src_mask)
    ep_interp.invoke()
    cross_kv = ep_interp.get_tensor(ep_idx["cross_kv"]).copy()
    dir_ids  = ep_interp.get_tensor(ep_idx["dir_ids"]).copy()
    self_kv  = np.zeros((NUM_LAYERS, 2, 1, MAX_TGT_LEN, NUM_HEADS, HEAD_DIM), dtype=np.float32)
    token = np.array([[tokenizer.pad_token_id or 0]], dtype=np.int32)
    ds_interp.set_tensor(ds_idx["token"],    token)
    ds_interp.set_tensor(ds_idx["step"],     np.array([0], dtype=np.int32))
    ds_interp.set_tensor(ds_idx["self_kv"],  self_kv)
    ds_interp.set_tensor(ds_idx["cross_kv"], cross_kv)
    ds_interp.set_tensor(ds_idx["src_mask"], src_mask)
    ds_interp.set_tensor(ds_idx["dir_ids"],  dir_ids)
    ds_interp.invoke()

    # Time encode+prefill
    t0 = time.perf_counter()
    for _ in range(5):
        ep_interp.set_tensor(ep_src_idx, src_ids)
        ep_interp.set_tensor(ep_msk_idx, src_mask)
        ep_interp.invoke()
    ep_ms = (time.perf_counter() - t0) / 5 * 1000

    # Time one decode step
    cross_kv = ep_interp.get_tensor(ep_idx["cross_kv"]).copy()
    dir_ids  = ep_interp.get_tensor(ep_idx["dir_ids"]).copy()
    self_kv  = np.zeros((NUM_LAYERS, 2, 1, MAX_TGT_LEN, NUM_HEADS, HEAD_DIM), dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(n_steps):
        step = np.array([i % MAX_TGT_LEN], dtype=np.int32)
        ds_interp.set_tensor(ds_idx["token"],    token)
        ds_interp.set_tensor(ds_idx["step"],     step)
        ds_interp.set_tensor(ds_idx["self_kv"],  self_kv)
        ds_interp.set_tensor(ds_idx["cross_kv"], cross_kv)
        ds_interp.set_tensor(ds_idx["src_mask"], src_mask)
        ds_interp.set_tensor(ds_idx["dir_ids"],  dir_ids)
        ds_interp.invoke()
        self_kv = ds_interp.get_tensor(ds_idx["self_kv_out"]).copy()
    step_ms = (time.perf_counter() - t0) / n_steps * 1000

    total_est = ep_ms + step_ms * MAX_TGT_LEN
    print(f"\nSpeed benchmark:")
    print(f"  encode+prefill : {ep_ms:.1f} ms")
    print(f"  decode step    : {step_ms:.1f} ms/step")
    print(f"  est. {MAX_TGT_LEN}-step  : {total_est:.0f} ms")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="v7.2.6 Dense model: MatMoE JAX → TFLite")
    parser.add_argument("--model-msg",     required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--mlp-dim",  type=int, default=256,
                        help="mlp_dim of the Dense checkpoint (256 or 512)")
    parser.add_argument("--out-dir",  default="tflite_dense")
    parser.add_argument("--quantize", choices=["none", "dynamic", "float16"],
                        default="none")
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sm_dir = os.path.join(args.out_dir, "savedmodels")
    os.makedirs(sm_dir, exist_ok=True)

    # ── 1. Load model ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Loading Dense model (mlp_dim={args.mlp_dim}) …")
    t0 = time.time()
    tokenizer = load_tokenizer(args.tokenizer_dir)
    model     = build_dense_model(tokenizer, mlp_dim=args.mlp_dim)
    model     = load_msgpack_model(model, args.model_msg)
    print(f"  Loaded in {time.time()-t0:.1f}s  |  vocab={model.cfg.vocab_size}")

    # ── 2. Float32 patch ─────────────────────────────────────────────────────
    print("  Patching model to float32 for TFLite …")

    state = nnx.state(model)
    def _cast_leaf_to_f32(x):
        try:
            if hasattr(x, 'dtype') and np.dtype(x.dtype) == np.dtype('bfloat16'):
                return jnp.asarray(x, jnp.float32)
        except TypeError:
            pass
        return x
    nnx.update(model, jax.tree.map(_cast_leaf_to_f32, state))

    _visited = set()
    def _patch_dtype(obj):
        oid = id(obj)
        if oid in _visited:
            return
        _visited.add(oid)
        if not isinstance(obj, nnx.Module):
            return
        for key, val in list(vars(obj).items()):
            if key in ('dtype', 'param_dtype'):
                if val is jnp.bfloat16 or val == jnp.bfloat16:
                    try:
                        setattr(obj, key, jnp.float32)
                    except Exception:
                        object.__setattr__(obj, key, jnp.float32)
            elif isinstance(val, nnx.Module):
                _patch_dtype(val)
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, nnx.Module):
                        _patch_dtype(item)
    _patch_dtype(model)

    try:
        model.cfg = model.cfg.replace(dtype=jnp.float32)
        print("  model.cfg.dtype → float32 ✓")
    except Exception as e:
        print(f"  Warning: cfg.replace failed ({e})")

    def _tflite_embed_call(self, inputs):
        emb  = jnp.asarray(self.embedding, jnp.float32)
        flat = inputs.reshape(-1)
        out  = jnp.take(emb, flat, axis=0, mode='clip')
        return out.reshape(inputs.shape + (emb.shape[-1],))
    nnx.Embed.__call__ = _tflite_embed_call
    print("  nnx.Embed monkey-patched ✓")

    # ── 3. Build JAX functions ───────────────────────────────────────────────
    print("\nBuilding JAX forward functions …")
    encode_prefill_jax = make_encode_prefill_fn(model)
    decode_step_jax    = make_decode_step_fn(model)

    # ── 4. jax2tf conversion ─────────────────────────────────────────────────
    print("\nConverting JAX → TensorFlow via jax2tf …")
    print("  [encode_prefill] …")
    tf_encode_prefill = jax2tf.convert(encode_prefill_jax, native_serialization=False)
    print("  [decode_step] …")
    tf_decode_step    = jax2tf.convert(decode_step_jax,    native_serialization=False)

    # ── 5. Build TF modules ──────────────────────────────────────────────────
    print("\nBuilding tf.Module wrappers …")
    ep_module = build_encode_prefill_module(tf_encode_prefill)
    ds_module = build_decode_step_module(tf_decode_step, vocab_size=model.cfg.vocab_size)

    # ── 6. Save SavedModels ──────────────────────────────────────────────────
    print("\nSaving SavedModels …")
    ep_sm = os.path.join(sm_dir, "encode_prefill")
    ds_sm = os.path.join(sm_dir, "decode_step")

    print(f"  Tracing encode_prefill → {ep_sm} …")
    t0 = time.time()
    tf.saved_model.save(ep_module, ep_sm)
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"  Tracing decode_step → {ds_sm} …")
    t0 = time.time()
    tf.saved_model.save(ds_module, ds_sm)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── 7. Convert to TFLite ─────────────────────────────────────────────────
    ep_tfl = os.path.join(args.out_dir, "encode_prefill.tflite")
    ds_tfl = os.path.join(args.out_dir, "decode_step.tflite")

    print(f"\nConverting SavedModels → TFLite (quantize={args.quantize}) …")
    print(f"  encode_prefill → {ep_tfl} …")
    t0 = time.time()
    ep_bytes = to_tflite(ep_sm, ep_tfl, quantize=args.quantize)
    print(f"  Done in {time.time()-t0:.1f}s  ({len(ep_bytes)/1024/1024:.1f} MB)")

    print(f"  decode_step → {ds_tfl} …")
    t0 = time.time()
    ds_bytes = to_tflite(ds_sm, ds_tfl, quantize=args.quantize)
    print(f"  Done in {time.time()-t0:.1f}s  ({len(ds_bytes)/1024/1024:.1f} MB)")

    total_mb = (len(ep_bytes) + len(ds_bytes)) / 1024 / 1024
    print(f"\n  Total TFLite size: {total_mb:.1f} MB")

    # ── 8. Instantiate interpreters ──────────────────────────────────────────
    ep_interp = tf.lite.Interpreter(model_content=ep_bytes)
    ep_interp.allocate_tensors()
    ds_interp = tf.lite.Interpreter(model_content=ds_bytes)
    ds_interp.allocate_tensors()

    ep_idx = {}
    cross_kv_idx, dir_ids_idx = _build_ep_index(ep_interp)
    ep_idx["cross_kv"] = cross_kv_idx
    ep_idx["dir_ids"]  = dir_ids_idx

    ds_idx = _build_ds_index(ds_interp)

    # ── 9. Print tensor details ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TFLite tensor details:")
    for name, bts in [("encode_prefill", ep_bytes), ("decode_step", ds_bytes)]:
        interp = tf.lite.Interpreter(model_content=bts)
        interp.allocate_tensors()
        print(f"\n  {name}.tflite")
        print("    INPUTS:")
        for t in interp.get_input_details():
            print(f"      [{t['index']}] {t['name']:45s} {str(t['shape']):30s} {t['dtype']}")
        print("    OUTPUTS:")
        for t in interp.get_output_details():
            print(f"      [{t['index']}] {t['name']:45s} {str(t['shape']):30s} {t['dtype']}")

    # ── 10. Speed benchmark ───────────────────────────────────────────────────
    benchmark_speed(ep_interp, ep_idx, ds_interp, ds_idx, tokenizer)

    # ── 11. Sanity check ─────────────────────────────────────────────────────
    if not args.skip_check:
        ep_interp2 = tf.lite.Interpreter(model_content=ep_bytes)
        ep_interp2.allocate_tensors()
        ds_interp2 = tf.lite.Interpreter(model_content=ds_bytes)
        ds_interp2.allocate_tensors()
        sanity_check(model, tokenizer,
                     ep_interp2, ep_idx, ds_interp2, ds_idx)

    print(f"\nDone. Files written to: {args.out_dir}/")
    print(f"  {ep_tfl}")
    print(f"  {ds_tfl}")


if __name__ == "__main__":
    main()
