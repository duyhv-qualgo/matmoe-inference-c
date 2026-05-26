#!/usr/bin/env python3
"""
decode_output.py — read int32 token IDs produced by the C++ engine
(test_slm --out-bin) and print the decoded text via the HF tokenizer.

Usage:
    python3 scripts/decode_output.py \\
        --tokenizer /Users/duy.hv/qualgo/t5-translator-ios/T5TranslatorApp/tokenizer.json \\
        --in out_ids.bin
"""
import argparse
import numpy as np
from tokenizers import Tokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--in", dest="inp", default="out_ids.bin")
    args = p.parse_args()

    ids = np.fromfile(args.inp, dtype=np.int32)
    tok = Tokenizer.from_file(args.tokenizer)
    txt = tok.decode(ids.tolist(), skip_special_tokens=True)

    print(f"raw ids ({len(ids)}): {ids.tolist()}")
    print(f"decoded: {txt}")


if __name__ == "__main__":
    main()
