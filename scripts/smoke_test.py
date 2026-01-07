#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import random
import argparse
from typing import Optional, Tuple, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def extract_final_answer(text: str) -> Optional[str]:
    """
    Extract the final answer from GSM8K-style output: '#### <number>'
    Returns the last matched number as a string, or None.
    """
    matches = re.findall(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    return matches[-1] if matches else None


def load_jsonl(path: str, max_items: int = 0) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_items and i >= max_items:
                break
            items.append(json.loads(line))
    return items


def build_prompt(tok: AutoTokenizer, question: str) -> str:
    """
    Qwen3 chat template prompt.
    """
    msgs = [
        {
            "role": "system",
            "content": "请先推理，再在最后一行输出 `#### <最终答案>`，只输出一个最终答案。",
        },
        {
            "role": "user",
            "content": f"Question: {question}\nAnswer:",
        },
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate_one(
    model,
    tok,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    inputs = tok([prompt_text], return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        use_cache=True,  # generation should use KV cache
    )
    return tok.decode(out[0], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default="checkpoints/Qwen3-0.6B",
                  help="Local model dir (recommended). You can also pass HF id.")
    p.add_argument("--data", type=str, default="data/gsm8k/train.jsonl")
    p.add_argument("--n", type=int, default=8, help="How many samples to test")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    args.n = 2
    args.max_new_tokens = 512
    args.model_dir = "checkpoints/Qwen3-1.7B"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    print(f"[INFO] loading dataset: {args.data}")
    data = load_jsonl(args.data)
    if len(data) == 0:
        raise RuntimeError("Dataset file is empty.")
    print(f"[OK] dataset size: {len(data)}")

    print(f"[INFO] loading tokenizer/model from: {args.model_dir}")
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False)

    # "sdpa" can be changed to "flash_attention_2" if flash-attn have been installed 
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    ).cuda()
    model.eval()

    # Print attention backend info if available
    attn_impl = getattr(model.config, "_attn_implementation", None)
    print(f"[OK] model loaded. attn_impl={attn_impl}  dtype={next(model.parameters()).dtype}")
    print(f"[INFO] device: {torch.cuda.get_device_name(0)}  torch={torch.__version__}")

    # Sample n examples
    idxs = [random.randrange(len(data)) for _ in range(args.n)]
    correct = 0

    t0 = time.time()
    for j, idx in enumerate(idxs, 1):
        ex = data[idx]
        q = ex["question"]
        gold = extract_final_answer(ex["answer"])

        prompt = build_prompt(tok, q)
        gen = generate_one(
            model, tok, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        pred = extract_final_answer(gen)
        ok = (pred is not None) and (gold is not None) and (pred == gold)
        correct += int(ok)

        print("=" * 90)
        print(f"[{j}/{args.n}] idx={idx}  ok={ok}")
        print(f"gold: {gold}")
        print(f"pred: {pred}")
        # show tail for brevity
        print("--- output tail ---")
        print(gen[-900:])

    dt = time.time() - t0
    print("=" * 90)
    print(f"[SUMMARY] correct={correct}/{args.n}  acc={correct/args.n:.3f}  time={dt:.1f}s")

    # Simple GPU memory report
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"[CUDA] max_memory_allocated={mem:.2f} GB")


if __name__ == "__main__":
    main()
