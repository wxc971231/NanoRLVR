#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download (cache) Qwen3 model + tokenizer using Transformers.

Default:
  - Downloads to HF cache (respects HF_HOME/TRANSFORMERS_CACHE if set).

Optional:
  - Set --local-dir to also materialize a full copy under ../checkpoints/<name>
    (useful for offline, or pinning exact files).

"""

import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B",
                   help="HF model id, e.g., Qwen/Qwen3-0.6B")
    p.add_argument("--dtype", type=str, default="bf16",
                   choices=["bf16", "fp16", "fp32"],
                   help="dtype to instantiate model (download happens either way)")
    p.add_argument("--local-dir", type=str, default="checkpoints/Qwen3-1.7B",
                   help="If set, also save a copy under this directory (relative or absolute).")
    return p.parse_args()


def main():
    args = parse_args()

    dtype_map = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }

    print(f"[INFO] downloading tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    print("[OK] tokenizer cached.")

    print(f"[INFO] downloading model weights: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(__import__("torch"), dtype_map[args.dtype]),
        trust_remote_code=False,
    )
    print("[OK] model cached.")

    if args.local_dir:
        local_dir = os.path.abspath(args.local_dir)
        os.makedirs(local_dir, exist_ok=True)
        print(f"[INFO] saving tokenizer/model to local dir: {local_dir}")
        tok.save_pretrained(local_dir)
        model.save_pretrained(local_dir, safe_serialization=True)
        print("[OK] saved to local dir.")

    print("[INFO] done.")


if __name__ == "__main__":
    main()
