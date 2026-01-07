#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download GSM8K from HuggingFace datasets and export to jsonl.
Outputs:
  ../data/gsm8k/train.jsonl
  ../data/gsm8k/test.jsonl
"""

import os
import json
from datasets import load_dataset

DATASET_NAME = "openai/gsm8k"
CONFIG_NAME = "main"

def dump_jsonl(split, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in split:
            # Keep only the fields we need
            obj = {
                "question": ex["question"],
                "answer": ex["answer"],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    print(f"[OK] wrote {n} lines -> {out_path}")


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, "data", "gsm8k")

    print(f"[INFO] loading dataset: {DATASET_NAME} ({CONFIG_NAME})")
    ds = load_dataset(DATASET_NAME, CONFIG_NAME)

    dump_jsonl(ds["train"], os.path.join(out_dir, "train.jsonl"))
    dump_jsonl(ds["test"], os.path.join(out_dir, "test.jsonl"))

    # Small sanity print
    print("[INFO] sample question:", ds["train"][0]["question"][:120].replace("\n", " "))
    print("[INFO] done.")


if __name__ == "__main__":
    main()
