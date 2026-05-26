#!/usr/bin/env python3
"""
dump_prompt.py — tokenize a sentence with the HF tokenizer and write
src_ids + src_mask as a single packed int32 binary file readable by
the C++ test harness.

File layout (matches what test_main.cc reads):
    [src_ids   x kMaxSrcLen (int32, little-endian)]   ← 128 * 4 = 512 B
    [src_mask  x kMaxSrcLen (int32, little-endian)]   ← 128 * 4 = 512 B
Total: 1024 bytes.

Usage:
    python3 scripts/dump_prompt.py \\
        --tokenizer /Users/duy.hv/qualgo/t5-translator-ios/T5TranslatorApp/tokenizer.json \\
        --text "<translate-en-vi> Hello, how are you today?" \\
        --out prompt.bin
"""
import argparse
import numpy as np
from tokenizers import Tokenizer

MAX_SRC_LEN = 128


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True,
                   help="Path to tokenizer.json (HuggingFace tokenizers format).")
    p.add_argument("--text", required=True,
                   help="Source sentence, e.g. '<translate-en-vi> Hello world.'")
    p.add_argument("--out", default="prompt.bin")
    p.add_argument("--max-len", type=int, default=MAX_SRC_LEN)
    args = p.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    # Make sure padding pads to fixed length so the C++ side gets exactly
    # max_len entries.
    tok.enable_padding(length=args.max_len, pad_id=0, pad_token="<pad>")
    tok.enable_truncation(max_length=args.max_len)

    enc = tok.encode(args.text)
    ids  = np.array(enc.ids,             dtype=np.int32)
    mask = np.array(enc.attention_mask,  dtype=np.int32)

    # Pad / truncate to be safe.
    if len(ids) < args.max_len:
        pad = args.max_len - len(ids)
        ids  = np.concatenate([ids,  np.zeros(pad, dtype=np.int32)])
        mask = np.concatenate([mask, np.zeros(pad, dtype=np.int32)])
    ids  = ids[:args.max_len]
    mask = mask[:args.max_len]

    payload = np.concatenate([ids, mask]).astype(np.int32)
    payload.tofile(args.out)

    n_tok = int(mask.sum())
    print(f"Wrote {args.out}  ({payload.nbytes} bytes)")
    print(f"  text          : {args.text}")
    print(f"  real tokens   : {n_tok}/{args.max_len}")
    print(f"  first 16 ids  : {ids[:16].tolist()}")


if __name__ == "__main__":
    main()
