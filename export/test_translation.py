#!/usr/bin/env python3
"""Quick translation test: JAX vs TFLite KV-cache, side-by-side.

Supports both MoE (v7.2.2) and Dense (v7.2.6) models via --dense flag.
"""

import argparse
import os
import sys

import numpy as np

CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../v7.2.2_step80000/code")
sys.path.insert(0, CODE_DIR)

import jax
import jax.numpy as jnp
import tensorflow as tf

from release_utils import load_msgpack_model, load_tokenizer

MAX_SRC_LEN = 128
MAX_TGT_LEN = 128

INPUTS = [
    '<translate-en-vi> Honestly, I think the "dead internet theory" is starting to feel less like a conspiracy and more like a Tuesday afternoon.',
    "<translate-en-vi> I hear you. It's getting harder to tell if I'm arguing with a human or a very sophisticated script that really wants me to buy a specific brand of blender.",
    '<translate-en-vi> Exactly! I saw a thread earlier where twenty different accounts used the exact same "unpopular opinion" word-for-word. It\'s like a digital house of mirrors.',
    "<translate-en-vi> The irony is that we're using AI to filter out the noise created by other AI. We've essentially built an arms race for our own attention spans.",
    "<translate-en-vi> It makes me miss the old web—shoddy HTML, neon backgrounds, and people just talking about their niche hobbies because they actually liked them.",
    '<translate-en-vi> The "Geocities" era? It was messy, but at least it was authentic. Now everything is optimized for an algorithm that doesn\'t even have a pulse.',
    '<translate-en-vi> Do you think we\'ll ever go back? Like a "digital Renaissance" where people prioritize verified human spaces?',
    "<translate-en-vi> Maybe. I think we'll see a rise in gated communities or \"analog-first\" hobbies. People are going to crave the friction of real life again.",
    "<translate-en-vi> I hope so. I'm tired of feeling like I'm just a data point being bounced around a server farm.",
    "<translate-en-vi> Same. Anyway, enough existential dread for one day—want to go grab a coffee and look at some actual trees?",
]


def jax_orig_translate(model, tokenizer, mlp_dim, text):
    """Original JAX inference via generate_fast_greedy_jitted (MoE only)."""
    from moe_inference import generate_fast_greedy_jitted
    enc = tokenizer([text], padding="max_length", truncation=True,
                    max_length=MAX_SRC_LEN, return_tensors="np")
    src_ids  = jnp.array(enc.input_ids.astype("int32"))
    src_mask = jnp.array(enc.attention_mask.astype("int32"))
    ids, _ = generate_fast_greedy_jitted(
        model, src_ids, src_mask,
        max_len=MAX_TGT_LEN,
        pad_id=tokenizer.pad_token_id or 0,
        eos_id=tokenizer.eos_token_id or 1,
        current_mlp_dim=mlp_dim,
    )
    return tokenizer.decode(np.array(ids)[0], skip_special_tokens=True)


def jax_kv_translate(model, tokenizer, text, mlp_dim=None, dense=False):
    """JAX KV-cache decode via the same decode_step_fn that's exported to TFLite."""
    enc = tokenizer([text], padding="max_length", truncation=True,
                    max_length=MAX_SRC_LEN, return_tensors="np")
    src_ids  = enc.input_ids.astype("int32")
    src_mask = enc.attention_mask.astype("int32")
    if dense:
        from export_tflite_dense import jax_kvcache_decode
        tokens = jax_kvcache_decode(model, tokenizer, src_ids, src_mask)
    else:
        from export_tflite import jax_kvcache_decode
        tokens = jax_kvcache_decode(model, tokenizer, mlp_dim, src_ids, src_mask)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def tflite_translate(ep_interp, ds_interp, ep_idx, ds_idx, tokenizer, text):
    enc = tokenizer([text], padding="max_length", truncation=True,
                    max_length=MAX_SRC_LEN, return_tensors="np")
    src_ids  = enc.input_ids.astype("int32")
    src_mask = enc.attention_mask.astype("int32")
    from export_tflite import tflite_kvcache_decode
    tokens = tflite_kvcache_decode(
        ep_interp, ds_interp, ep_idx, ds_idx, src_ids, src_mask, tokenizer)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def tflite_translate_dense(ep_interp, ds_interp, ep_idx, ds_idx, tokenizer, text):
    enc = tokenizer([text], padding="max_length", truncation=True,
                    max_length=MAX_SRC_LEN, return_tensors="np")
    src_ids  = enc.input_ids.astype("int32")
    src_mask = enc.attention_mask.astype("int32")
    from export_tflite_dense import tflite_kvcache_decode
    tokens = tflite_kvcache_decode(
        ep_interp, ds_interp, ep_idx, ds_idx, src_ids, src_mask, tokenizer)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-msg",      required=True)
    parser.add_argument("--tokenizer-dir",  required=True)
    parser.add_argument("--tflite-int8",    default=None, help="INT8 tflite dir")
    parser.add_argument("--tflite-f16",     default=None, help="float16 tflite dir")
    parser.add_argument("--tflite-f32",     default=None, help="float32 tflite dir")
    parser.add_argument("--mlp-dim",  type=int, default=256)
    parser.add_argument("--ckpt-dim", type=int, default=512)
    parser.add_argument("--dense",    action="store_true",
                        help="Use Dense model (v7.2.6) instead of MoE")
    args = parser.parse_args()

    from flax import nnx
    print("Loading model …")
    tokenizer = load_tokenizer(args.tokenizer_dir)

    if args.dense:
        from export_tflite_dense import build_dense_model, _build_ep_index, _build_ds_index
        model = build_dense_model(tokenizer, mlp_dim=args.mlp_dim)
    else:
        from release_utils import build_model_for_dim
        from export_tflite import _build_ep_index, _build_ds_index
        model = build_model_for_dim(tokenizer, mlp_dim=args.ckpt_dim)

    model = load_msgpack_model(model, args.model_msg)
    print(f"  vocab={model.cfg.vocab_size}")

    # Patch bfloat16 → float32 (required for jax_kvcache_decode and TFLite export)
    state = nnx.state(model)
    def _cast_f32(x):
        try:
            if hasattr(x, 'dtype') and np.dtype(x.dtype) == np.dtype('bfloat16'):
                return jnp.asarray(x, jnp.float32)
        except TypeError:
            pass
        return x
    nnx.update(model, jax.tree.map(_cast_f32, state))
    _visited = set()
    def _patch_dtype(obj):
        if id(obj) in _visited or not isinstance(obj, nnx.Module): return
        _visited.add(id(obj))
        for k, v in list(vars(obj).items()):
            if k in ('dtype', 'param_dtype') and v in (jnp.bfloat16,):
                try: setattr(obj, k, jnp.float32)
                except Exception: object.__setattr__(obj, k, jnp.float32)
            elif isinstance(v, nnx.Module): _patch_dtype(v)
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, nnx.Module): _patch_dtype(item)
    _patch_dtype(model)
    try: model.cfg = model.cfg.replace(dtype=jnp.float32)
    except Exception: pass
    def _tflite_embed_call(self, inputs):
        emb  = jnp.asarray(self.embedding, jnp.float32)
        flat = inputs.reshape(-1)
        out  = jnp.take(emb, flat, axis=0, mode='clip')
        return out.reshape(inputs.shape + (emb.shape[-1],))
    nnx.Embed.__call__ = _tflite_embed_call
    print("  float32 patch applied ✓")

    def load_tflite(tflite_dir):
        ep = tf.lite.Interpreter(model_path=os.path.join(tflite_dir, "encode_prefill.tflite"))
        ds = tf.lite.Interpreter(model_path=os.path.join(tflite_dir, "decode_step.tflite"))
        ep.allocate_tensors(); ds.allocate_tensors()
        ck, dk = _build_ep_index(ep)
        ep_idx = {"cross_kv": ck, "dir_ids": dk}
        ds_idx = _build_ds_index(ds)
        return ep, ds, ep_idx, ds_idx

    variants = {}
    if args.tflite_int8: variants["INT8 "] = load_tflite(args.tflite_int8)
    if args.tflite_f16:  variants["fp16 "] = load_tflite(args.tflite_f16)
    if args.tflite_f32:  variants["fp32 "] = load_tflite(args.tflite_f32)

    _tfl_translate = tflite_translate_dense if args.dense else tflite_translate

    SEP = "─" * 80
    totals = {k: {"match": 0, "total": 0} for k in variants}

    for i, text in enumerate(INPUTS, 1):
        enc = tokenizer([text], padding="max_length", truncation=True,
                        max_length=MAX_SRC_LEN, return_tensors="np")
        src_len = int(enc.attention_mask.sum())

        print(f"\n{SEP}")
        print(f"[{i}/{len(INPUTS)}] ({src_len} src toks) {text}")

        if args.dense:
            ref_out = jax_kv_translate(model, tokenizer, text, dense=True)
            print(f"  JAX kv ref: {ref_out}")
        else:
            ref_out = jax_orig_translate(model, tokenizer, args.mlp_dim, text)
            print(f"  Flax ref  : {ref_out}")

        ref_toks = tokenizer(ref_out, return_tensors="np").input_ids[0]

        for label, (ep, ds, ep_idx, ds_idx) in variants.items():
            tfl_out  = _tfl_translate(ep, ds, ep_idx, ds_idx, tokenizer, text)
            tfl_toks = tokenizer(tfl_out, return_tensors="np").input_ids[0]
            n = min(len(ref_toks), len(tfl_toks))
            match = int(np.sum(ref_toks[:n] == tfl_toks[:n]))
            total = max(len(ref_toks), len(tfl_toks))
            totals[label]["match"] += match
            totals[label]["total"] += total
            icon = "✓" if tfl_out == ref_out else "~"
            print(f"  {label}    : {tfl_out}  {icon} ({match}/{total} toks)")

    print(f"\n{SEP}")
    print("OVERALL TOKEN MATCH vs Flax reference (generate_fast_greedy_jitted):")
    for label, counts in totals.items():
        pct = counts["match"] / counts["total"] * 100 if counts["total"] else 0
        print(f"  {label}: {counts['match']}/{counts['total']} = {pct:.1f}%")
    print(SEP)


if __name__ == "__main__":
    main()
