#!/usr/bin/env python3
"""
benchmark_tflite.py — TFLite BLEU on PhoMT + latency/throughput (MoE and Dense)

Usage:
    # MoE
    python3 benchmark_tflite.py \
        --tokenizer-dir ../../v7.2.2_step80000/t5-en-vi-tokenizer-20k-v3 \
        --tflite-dir    tflite_kvcache_256 \
        --n 100

    # Dense
    python3 benchmark_tflite.py \
        --tokenizer-dir ../../v7.2.2_step80000/t5-en-vi-tokenizer-20k-v3 \
        --tflite-dir    tflite_dense \
        --n 100
"""

import argparse
import os
import sys
import time
import random

import numpy as np

CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../v7.2.2_step80000/code")
sys.path.insert(0, CODE_DIR)

import tensorflow as tf
from sacrebleu.metrics import BLEU

from release_utils import load_tokenizer
from export_tflite import _build_ep_index, _build_ds_index

MAX_SRC_LEN = 128
MAX_TGT_LEN = 128
NUM_LAYERS  = 8
NUM_HEADS   = 8
HEAD_DIM    = 64


def load_tflite(tflite_dir):
    ep = tf.lite.Interpreter(model_path=os.path.join(tflite_dir, "encode_prefill.tflite"))
    ds = tf.lite.Interpreter(model_path=os.path.join(tflite_dir, "decode_step.tflite"))
    ep.allocate_tensors()
    ds.allocate_tensors()
    ck, dk = _build_ep_index(ep)
    return ep, ds, {"cross_kv": ck, "dir_ids": dk}, _build_ds_index(ds)


def tokenize(tokenizer, text):
    enc = tokenizer([text], padding="max_length", truncation=True,
                    max_length=MAX_SRC_LEN, return_tensors="np")
    return enc.input_ids.astype(np.int32), enc.attention_mask.astype(np.int32)


def translate(ep, ds, ep_idx, ds_idx, tokenizer, text):
    src_ids, src_mask = tokenize(tokenizer, text)
    pad_id = tokenizer.pad_token_id or 0
    eos_id = tokenizer.eos_token_id or 1

    ep_src = ep_msk = None
    for t in ep.get_input_details():
        if "source_ids" in t["name"]: ep_src = t["index"]
        elif "src_mask"  in t["name"]: ep_msk = t["index"]
    if ep_src is None:
        ep_src, ep_msk = ep.get_input_details()[0]["index"], ep.get_input_details()[1]["index"]

    t0 = time.perf_counter()
    ep.set_tensor(ep_src, src_ids)
    ep.set_tensor(ep_msk, src_mask)
    ep.invoke()
    ep_ms = (time.perf_counter() - t0) * 1000

    cross_kv = ep.get_tensor(ep_idx["cross_kv"]).copy()
    dir_ids  = ep.get_tensor(ep_idx["dir_ids"]).copy()
    self_kv  = np.zeros((NUM_LAYERS, 2, 1, MAX_TGT_LEN, NUM_HEADS, HEAD_DIM), dtype=np.float32)

    token = np.array([[pad_id]], dtype=np.int32)
    toks  = []
    step_ms_list = []

    for s in range(MAX_TGT_LEN):
        t0 = time.perf_counter()
        ds.set_tensor(ds_idx["token"],    token)
        ds.set_tensor(ds_idx["step"],     np.array([s], dtype=np.int32))
        ds.set_tensor(ds_idx["self_kv"],  self_kv)
        ds.set_tensor(ds_idx["cross_kv"], cross_kv)
        ds.set_tensor(ds_idx["src_mask"], src_mask)
        ds.set_tensor(ds_idx["dir_ids"],  dir_ids)
        ds.invoke()
        step_ms_list.append((time.perf_counter() - t0) * 1000)

        logits  = ds.get_tensor(ds_idx["logits"])
        self_kv = ds.get_tensor(ds_idx["self_kv_out"]).copy()
        next_tok = int(np.argmax(logits[0]))
        toks.append(next_tok)
        token = np.array([[next_tok]], dtype=np.int32)
        if next_tok == eos_id:
            break

    toks     = np.array(toks)
    out      = tokenizer.decode(toks, skip_special_tokens=True)
    n_out    = int(np.sum((toks != pad_id) & (toks != eos_id)))
    total_ms = ep_ms + sum(step_ms_list)
    return out, total_ms, n_out, ep_ms, step_ms_list


def stats(values):
    a = np.array(values)
    return dict(mean=np.mean(a), median=np.median(a),
                p95=np.percentile(a, 95), min=np.min(a), max=np.max(a))


def fmt(s, unit="ms"):
    return (f"mean={s['mean']:.1f}  median={s['median']:.1f}"
            f"  p95={s['p95']:.1f}  min={s['min']:.1f}  max={s['max']:.1f}  [{unit}]")


def main():
    parser = argparse.ArgumentParser(
        description="TFLite BLEU + latency benchmark (MoE and Dense)")
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--tflite-dir",    required=True)
    parser.add_argument("--n",    type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    SEP  = "=" * 70
    SEP2 = "─" * 70

    tokenizer = load_tokenizer(args.tokenizer_dir)
    ep, ds, ep_idx, ds_idx = load_tflite(args.tflite_dir)

    print("Warming up …")
    for _ in range(3):
        translate(ep, ds, ep_idx, ds_idx, tokenizer, "<translate-en-vi> Hello world.")
    print("Ready.\n")

    print("Loading PhoMT test set …")
    from datasets import load_dataset
    dataset = load_dataset("ura-hcmut/PhoMT", split="test")
    rng     = random.Random(args.seed)
    n_each  = args.n // 2
    indices = rng.sample(range(len(dataset)), min(args.n, len(dataset)))

    en_vi_rows = [(f"<translate-en-vi> {dataset[i]['en']}", dataset[i]['vi'])
                  for i in indices[:n_each]]
    vi_en_rows = [(f"<translate-vi-en> {dataset[i]['vi']}", dataset[i]['en'])
                  for i in indices[n_each:n_each * 2]]
    all_rows = en_vi_rows + vi_en_rows
    print(f"  {len(en_vi_rows)} en→vi + {len(vi_en_rows)} vi→en = {len(all_rows)} total\n")
    print(SEP)

    results = {"en_vi": [], "vi_en": []}
    all_ms, all_ep_ms, all_step_ms, all_ntoks = [], [], [], []

    for i, (src, ref) in enumerate(all_rows, 1):
        direction = "en_vi" if "<translate-en-vi>" in src else "vi_en"
        label     = "en→vi" if direction == "en_vi" else "vi→en"

        out, total_ms, n_out, ep_ms, step_ms_list = translate(
            ep, ds, ep_idx, ds_idx, tokenizer, src)

        results[direction].append((out, ref))
        all_ms.append(total_ms)
        all_ep_ms.append(ep_ms)
        all_step_ms.extend(step_ms_list)
        all_ntoks.append(n_out)

        print(f"[{i:3d}/{len(all_rows)}] {label}  {total_ms:6.0f}ms  {n_out}toks  {out[:60]}")

    bleu = BLEU(effective_order=True)
    def score(records):
        return bleu.corpus_score([r[0] for r in records],
                                 [[r[1] for r in records]]).score

    b_en  = score(results["en_vi"])
    b_vi  = score(results["vi_en"])
    b_avg = (b_en + b_vi) / 2

    print(f"\n{SEP}")
    print("BLEU  (sacrebleu corpus BLEU vs PhoMT reference)")
    print(SEP2)
    print(f"  en→vi : {b_en:.2f}")
    print(f"  vi→en : {b_vi:.2f}")
    print(f"  avg   : {b_avg:.2f}")

    print(f"\n{SEP}")
    print("LATENCY PER SENTENCE")
    print(SEP2)
    print(f"  total          : {fmt(stats(all_ms))}")
    print(f"  encode+prefill : {fmt(stats(all_ep_ms))}")
    print(f"  decode step    : {fmt(stats(all_step_ms))}")

    total_toks = sum(all_ntoks)
    total_s    = sum(all_ms) / 1000
    print(f"\n{SEP}")
    print("THROUGHPUT")
    print(SEP2)
    print(f"  {total_toks} output tokens in {total_s:.1f}s = {total_toks/total_s:.1f} tokens/sec")
    print(f"  {len(all_rows)} sentences in {total_s:.1f}s = {len(all_rows)/total_s:.2f} sentences/sec")

    print(f"\n{SEP}\nDone.\n{SEP}")


if __name__ == "__main__":
    main()
