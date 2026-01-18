#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
import argparse
import glob
import tqdm
import wandb
import numpy as np
from pprint import pprint
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import *

# ---------------------------
# PPO rollout
# ---------------------------
@torch.no_grad()
def rollout_ppo(
    model: nn.Module,
    ref_model: Optional[nn.Module],
    tokenizer: AutoTokenizer,
    batch_questions: List[str],
    batch_answers: List[str],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    kl_beta: float,
    autocast_dtype: torch.dtype,
    device: Union[int, torch.device],
    prompt_batch_size: int = 4
) -> RolloutBatch:
    """
    Generate G samples per prompt. Compute reward and PPO advantages.
    Store old_logp_sum from π_old (current model at rollout time).
    Optionally store ref token logp (for KL regularization).
    """
    # Rollout G samples per prompt
    all_chunk_res:Dict[str, torch.Tensor] = rollout(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        batch_questions=batch_questions,
        batch_answers=batch_answers,
        group_size=group_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        kl_beta=kl_beta,
        autocast_dtype=autocast_dtype,
        device=device,
        prompt_batch_size=prompt_batch_size,
    )
    ref_logp_tok = all_chunk_res['ref_logp_tok'] if 'ref_logp_tok' in all_chunk_res else None

    # PPO advantages: value baseline & std normalization if std is non-trivial
    returns = all_chunk_res['rewards']                              # [B*G]
    critic_values = all_chunk_res['critic_values']                  # [B*G]
    advantages = returns - critic_values                            # [B*G]
    adv_mean = advantages.mean()
    adv_std = advantages.std(dim=0, unbiased=False)
    advantages = torch.where(adv_std > 1e-6, (advantages - adv_mean) / (adv_std + 1e-6), advantages - adv_mean)

    return RolloutBatch(
        input_ids=all_chunk_res['out_ids'],                 # [B*G, max_traj_len]
        attention_mask=all_chunk_res['out_attn'],           # [B*G, max_traj_len]
        old_logp_tok=all_chunk_res['old_logp_tok'],         # [B*G, max_traj_len-1]
        ref_logp_tok=ref_logp_tok,                          # [B*G, max_traj_len-1] or None
        completion_mask=all_chunk_res['completion_mask'],   # [B*G, max_traj_len-1]
        completion_lens=all_chunk_res['completion_lens'],   # [B*G]
        prompt_lens=all_chunk_res['prompt_lens'],           # [B*G]
        rewards=all_chunk_res['rewards'],                   # [B*G]
        advantages=advantages,                              # [B*G]
    )

# ---------------------------
# PPO loss
# ---------------------------
def ppo_loss(
    model: nn.Module,
    batch: RolloutBatch,
    clip_eps: float,
    kl_beta: float,
    value_coef: float,
    autocast_dtype: torch.dtype,
    ratio_len_norm: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    tok_logp_new = compute_token_logps(
        model=model,
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        autocast_dtype=autocast_dtype,
        require_grad=True,
    )

    # sum logp on completion tokens
    comp_mask = build_completion_mask_targets(batch.attention_mask, batch.prompt_lens)
    logp_new_sum = (tok_logp_new * comp_mask).sum(dim=1)
    logp_old_sum = batch.old_logp_sum.detach()

    # ratio: support optional length-normalization to stabilize long sequences
    if ratio_len_norm:
        completion_len = comp_mask.sum(dim=1).float().clamp(min=1.0)
        log_ratio = (logp_new_sum - logp_old_sum) / completion_len
        ratio = torch.exp(log_ratio.clamp(-10, 10))
    else:
        log_ratio = (logp_new_sum - logp_old_sum).clamp(-20, 20)
        ratio = torch.exp(log_ratio)
    
    # clipped surrogate
    A = batch.advantages.detach()
    unclipped = ratio * A
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * A
    policy_loss = -torch.mean(torch.minimum(unclipped, clipped))

    # KL penalty to reference model (token-level)
    kl = torch.tensor(0.0, device=policy_loss.device)
    if batch.ref_logp_tok is not None and kl_beta > 0.0:
        ref_tok = batch.ref_logp_tok.detach()
        if ratio_len_norm:
            completion_len = comp_mask.sum(dim=1).float().clamp(min=1.0)
            kl_per_seq = ((tok_logp_new - ref_tok) * comp_mask).sum(dim=1) / completion_len
        else:
            kl_per_seq = ((tok_logp_new - ref_tok) * comp_mask).sum(dim=1)
        kl = kl_per_seq.mean()

    # PPO loss
    values_new = compute_values(
        model=model,
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        autocast_dtype=autocast_dtype,
        require_grad=True,
    )
    value_targets = batch.returns.detach()
    value_loss = F.mse_loss(values_new, value_targets)
    loss = policy_loss + kl_beta * kl + value_coef * value_loss

    stats = {
        "loss": loss.detach(),
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "value_mean": values_new.detach().mean(),
        "kl": kl.detach(),
        "ratio_mean": ratio.mean().detach(),
        "ratio_max": ratio.max().detach(),
        "reward_mean": batch.rewards.mean().detach(),
        "adv_mean": batch.advantages.mean().detach(),
        "len_mean": batch.completion_lens.float().mean().detach(),
    }
    return loss, stats

# ---------------------------
# evaluation
# ---------------------------
@torch.no_grad()
def evaluate_greedy(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    device: Union[int, torch.device],
    autocast_dtype: torch.dtype,
    max_new_tokens: int,
    eval_batches: int,
) -> Dict[str, float]:
    model_eval = model.module if hasattr(model, "module") else model
    model_eval.eval()

    # Only show progress bar on main rank
    correct, total = 0, 0
    total_gen_len, total_gen_cnt = 0.0, 0.0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    iterator = dataloader if not is_main_rank() else tqdm.tqdm(dataloader, desc=f"[GPU0-{world_size-1}]: Evaluating", leave=False)
    for b, items in enumerate(iterator):
        if b >= eval_batches:
            break

        questions = list(items["question"])
        answers = list(items["answer"])

        prompts_text = prepare_prompt(questions, tokenizer)
        tok = tokenizer(prompts_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = tok["input_ids"].to(device)
        attn = tok["attention_mask"].to(device)

        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            out = model_eval.generate(
                input_ids=input_ids,    # [B*G, L]
                attention_mask=attn,    # [B*G, L]
                do_sample=False,
                temperature=None,
                top_k=None,
                top_p=None,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,         # Force use_cache=True during rollout for speed, even if gradient checkpointing disables it globally
            )

        pad_id = tokenizer.pad_token_id
        input_valid = (input_ids != pad_id).sum(dim=1)
        out_valid = (out != pad_id).sum(dim=1)
        comp_len = (out_valid - input_valid).clamp(min=0)
        total_gen_len += comp_len.sum().item()
        total_gen_cnt += float(comp_len.numel())

        texts = tokenizer.batch_decode(out, skip_special_tokens=True)
        for t, gt in zip(texts, answers):
            r = gsm8k_reward(t, gt)
            correct += int(r > 0.5)     # gsm8k_reward ∈ {0.0,1.0}，用 >0.5 保证可扩展性（比如做了平滑）
            total += 1

    corr_t = torch.tensor([correct], device=device, dtype=torch.float32)
    tot_t = torch.tensor([total], device=device, dtype=torch.float32)
    len_t = torch.tensor([total_gen_len], device=device, dtype=torch.float32)
    cnt_t = torch.tensor([total_gen_cnt], device=device, dtype=torch.float32)
    corr_t = ddp_all_reduce_sum(corr_t)
    tot_t = ddp_all_reduce_sum(tot_t)
    len_t = ddp_all_reduce_sum(len_t)
    cnt_t = ddp_all_reduce_sum(cnt_t)

    acc = (corr_t / tot_t).item() if tot_t.item() > 0 else 0.0
    avg_len = (len_t / cnt_t).item() if cnt_t.item() > 0 else 0.0
    model_eval.train()
    return {"greedy_acc": acc, "avg_completion_len": avg_len}

# ---------------------------
# main
# ---------------------------
def parse_args():
    # 1. First pass: Check for --config argument
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--config", type=str, default="config/ppo_config.json", help="Path to JSON config file")
    known_args, remaining_args = conf_parser.parse_known_args()

    # 2. Load config if exists
    defaults = {}
    if known_args.config and os.path.exists(known_args.config):
        clean_print(f"Loading configuration from {known_args.config}", "[INFO]")
        with open(known_args.config, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            # Flatten the nested config for argparse
            for k, v in config_data.items():
                if isinstance(v, dict):
                    defaults.update(v)
                else:
                    defaults[k] = v
    else:
        clean_print(f"Config file {known_args.config} not found. Using defaults.", "[INFO]")

    # 3. Define main parser
    p = argparse.ArgumentParser(parents=[conf_parser])

    # --- Group: Paths ---
    g_path = p.add_argument_group("Paths")
    g_path.add_argument("--model_path", type=str, default=None)
    g_path.add_argument("--resume_path", type=str, default=None)
    g_path.add_argument("--train_jsonl", type=str, default=None)
    g_path.add_argument("--test_jsonl", type=str, default=None)

    # --- Group: Training Hypers ---
    g_train = p.add_argument_group("Training")
    g_train.add_argument("--total_steps", type=int, default=2000)
    g_train.add_argument("--batch_size", type=int, default=8, help="prompts per rank within a rollout")
    g_train.add_argument("--prompt_batch_size", type=int, default=4, help="prompts per generation chunk within a rollout")
    g_train.add_argument("--group_size", type=int, default=8, help="samples per prompt within a rollout")
    g_train.add_argument("--chunk_size", type=int, default=4, help="Mini-batch size for gradient accumulation within a rollout")
    g_train.add_argument("--max_new_tokens", type=int, default=512)
    g_train.add_argument("--temperature", type=float, default=1.0)
    g_train.add_argument("--top_p", type=float, default=0.95)
    g_train.add_argument("--seed", type=int, default=42)
    g_train.add_argument("--bf16", action="store_true")
    g_train.add_argument("--gradient_checkpointing", action="store_true")
    g_train.add_argument("--ratio_len_norm", action="store_true")

    # --- Group: Optimization ---
    g_optim = p.add_argument_group("Optimization")
    g_optim.add_argument("--lr", type=float, default=2e-6)
    g_optim.add_argument("--weight_decay", type=float, default=0.0)
    g_optim.add_argument("--optim", type=str, default="adamw", choices=["adamw", "adamw8bit"])
    g_optim.add_argument("--max_grad_norm", type=float, default=1.0)

    # --- Group: GRPO ---
    g_rl = p.add_argument_group("PPO")
    g_rl.add_argument("--clip_eps", type=float, default=0.2)
    g_rl.add_argument("--kl_beta", type=float, default=0.02)
    g_rl.add_argument("--k_epochs", type=int, default=2)
    g_rl.add_argument("--disable_ref", action="store_true", help="Disable KL regularization by not using reference model.")
    g_rl.add_argument("--value_coef", type=float, default=1.0)

    # --- Group: Evaluation & Saving ---
    g_eval = p.add_argument_group("Evaluation")
    g_eval.add_argument("--eval_skip_first", action="store_true")
    g_eval.add_argument("--eval_every", type=int, default=5)
    g_eval.add_argument("--eval_batches", type=int, default=10)
    g_eval.add_argument("--eval_batch_size", type=int, default=8, help="prompts per step (per rank)")
    g_eval.add_argument("--save_every", type=int, default=None)
    g_eval.add_argument("--save_best", type=int, default=5)

    # --- Group: Wandb ---
    g_wandb = p.add_argument_group("Wandb")
    g_wandb.add_argument("--wandb_project", type=str, default="grpo-gsm8k", help="Wandb project name")
    g_wandb.add_argument("--wandb_group_enforce", type=str, default=None, help="If not set, use auto group name")
    g_wandb.add_argument("--wandb_offline", action="store_true", help="Run wandb in offline mode")

    # --- Group: LoRA ---
    g_lora = p.add_argument_group("LoRA")
    g_lora.add_argument("--use_lora", action="store_true")
    g_lora.add_argument("--lora_r", type=int, default=16, help='Rank of the LoRA matrices')
    g_lora.add_argument("--lora_alpha", type=int, default=32, help='Scaling factor for the LoRA weights')
    g_lora.add_argument("--lora_dropout", type=float, default=0.05)

    # Apply defaults from config file (this overrides argparse defaults but is overridden by CLI args)
    p.set_defaults(**defaults)
    args = p.parse_args(remaining_args)

    return args


def main():
    # DDP init
    RANK, LOCAL_RANK, WORLD_SIZE = ddp_init()

    # args
    args = parse_args()
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float32
    best_performance = float("-inf")
    set_seed(args.seed + RANK)

    # save setting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_group_name = args.wandb_group_enforce if args.wandb_group_enforce is not None else \
        f'PPO-{args.model_path.split("/")[-1]}-LoRA' if args.use_lora else \
        f'PPO-{args.model_path.split("/")[-1]}'
    args.save_dir = f"runs/{exp_group_name}/{timestamp}"

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, fix_mistral_regex=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # model
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=autocast_dtype, low_cpu_mem_usage=True).to(LOCAL_RANK)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.use_lora:
        model = maybe_apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None and hasattr(model, "lm_head"):
        hidden_size = model.lm_head.in_features
    model.v_head = nn.Linear(hidden_size, 1).to(LOCAL_RANK) # add value head for critic

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK], find_unused_parameters=False)

    # reference model: frozen copy of initial checkpoint
    ref_model = None
    if (not args.disable_ref) and args.kl_beta > 0.0:
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=autocast_dtype, low_cpu_mem_usage=True).to(LOCAL_RANK)
        ref_model.eval()
        for p_ in ref_model.parameters():
            p_.requires_grad_(False)

    # data
    train_set = GSM8KJsonl(args.train_jsonl)
    test_set = GSM8KJsonl(args.test_jsonl)
    train_sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed)
    eval_sampler = DistributedSampler(test_set, shuffle=False, seed=args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda x: x, # keep raw strings; build prompts later
    )
    eval_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        sampler=eval_sampler,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    train_iter = iter(train_loader)
    
    # Only optimize trainable parameters (important if LoRA)
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(optim_params, args.lr, args.weight_decay, args.optim)

    # load snapshot if exists
    if args.resume_path is not None and os.path.exists(os.path.join(args.resume_path, f"snapshot_seed{args.seed}.pt")):
        snapshot = torch.load(os.path.join(args.resume_path, f"snapshot_seed{args.seed}.pt"), map_location="cpu", weights_only=False)
        model.module.load_state_dict(snapshot["model_state_dict"])
        optimizer.load_state_dict(snapshot["optimizer_state_dict"])
        rng_states = snapshot.get("rng_states", None)
        random.setstate(rng_states["python"])
        np.random.set_state(rng_states["numpy"])
        torch.set_rng_state(rng_states["torch"])
        torch.cuda.set_rng_state_all(rng_states["torch_cuda"])
        args.save_dir = args.resume_path
        step_begin = snapshot["step"]
        wandb_id = snapshot["wandb_id"]
        best_performance = snapshot["best_performance"]
        clean_print(f"Resuming training from snapshot at step {step_begin}", "[Trainer]")
    else:
        # save setting
        if is_main_rank():
            create_folder_if_not_exist(args.save_dir)
            shutil.copy2(src=os.path.abspath(__file__), dst=f"{args.save_dir}/PPO.py")            
            with open("config/ppo_config.json", "r") as f:
                config_data = json.load(f)
                config_data.setdefault("Training", {})["world_size"] = WORLD_SIZE
            with open(f"{args.save_dir}/ppo_config.json", "w") as f:
                json.dump(config_data, f, indent=4)

        step_begin = 1
        wandb_id = wandb.util.generate_id() # This unique id is necessary for log resuming
        best_performance = float("-inf")
        clean_print("Snapshot not found. Training model from scratch", "[Trainer]")
    
    # wandb init
    if is_main_rank() and wandb is not None:
        wandb.init(
            project=args.wandb_project, group=exp_group_name, name=timestamp,
            config=vars(args), dir="Wandb",
            id=wandb_id, resume="allow",
            mode="offline" if args.wandb_offline else "online",
        )

    # training loop
    model.train()
    t0 = time.time()
    for step in range(step_begin, args.total_steps + 1):
        wandb_log_dict = {}
        train_sampler.set_epoch(step)
        clean_print("-" * 50 + f" RL Step [{step}] " + "-" * 50)

        # ================ (1) eval before training to check initial peformance ================
        if (step % args.eval_every == 0) or (step == 1 and not args.eval_skip_first):
            torch.cuda.empty_cache()        # Free memory
            model.eval()                    # eval model for inference
            metrics = evaluate_greedy(
                model=model,
                tokenizer=tokenizer,
                dataloader=eval_loader,
                device=LOCAL_RANK,
                autocast_dtype=autocast_dtype,
                max_new_tokens=args.max_new_tokens,
                eval_batches=args.eval_batches,
            )
            torch.cuda.empty_cache()

            greedy_acc, avg_len = metrics["greedy_acc"], metrics["avg_completion_len"]
            clean_print(f"greedy_acc={greedy_acc:.4f}, avg_len={avg_len:.1f}", f'[eval @step {step}]')
            wandb_log_dict.update({
                "eval/greedy_acc": greedy_acc,
                "eval/avg_completion_len": avg_len,
            })

            if is_main_rank():
                if args.save_best is not None and greedy_acc > best_performance:
                    best_performance = greedy_acc
                    ckpt_path = os.path.join(f"{args.save_dir}/best", f"{round(best_performance,3)}_seed{args.seed}_step{step}")
                    os.makedirs(ckpt_path, exist_ok=True)
                    save_model = (model.module if hasattr(model, "module") else model)  # unwrap DDP
                    save_model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)
                    print(f"[save best] {ckpt_path}")

                    # Only keep top-k best checkpoints (directories)
                    ckpt_dirs = [d for d in glob.glob(os.path.join(f'{args.save_dir}/best', "*")) if os.path.isdir(d)]
                    if len(ckpt_dirs) > args.save_best:
                        ckpt_dirs_sorted = sorted(ckpt_dirs, key=lambda x: float(os.path.basename(x).split("_")[0]), reverse=True)
                        for stale_dir in ckpt_dirs_sorted[args.save_best:]:
                            shutil.rmtree(stale_dir, ignore_errors=True)

        # ================ (2) save before training to make sure resume from least step ================
        if is_main_rank():
            # save snapshot every step for resume training
            snapshot = {
                "step": step,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "rng_states": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state() if np is not None else None,
                    "torch": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all(),
                },
                "wandb_id": (wandb.run.id if (wandb is not None and wandb.run is not None) else None),
                "best_performance": best_performance,
            }
            torch.save(snapshot, os.path.join(args.save_dir, f"snapshot_seed{args.seed}.pt"))

            # save ckpt at every `args.save_every` steps
            if args.save_every is not None and (step % args.save_every == 0):
                ckpt_path = os.path.join(f"{args.save_dir}/interval", f"seed{args.seed}_step{step}")
                os.makedirs(ckpt_path, exist_ok=True)
                save_model = model.module if hasattr(model, "module") else model
                save_model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                print(f"[save] {ckpt_path}")

        # ================ (3) rollout ================
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        questions = [x["question"] for x in batch]
        answers = [x["answer"] for x in batch]

        torch.cuda.empty_cache()    # Free memory
        model.eval()                # eval model for inference
        rollout = rollout_ppo(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            batch_questions=questions,
            batch_answers=answers,
            group_size=args.group_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            kl_beta=args.kl_beta,
            autocast_dtype=autocast_dtype,
            device=LOCAL_RANK,
            prompt_batch_size=args.prompt_batch_size,
        )
        torch.cuda.empty_cache()    # Free memory after inference

        # ================ (4) update on same rollout for k_epochs (so clip can matter) ================
        stats_last = None
        TOTAL_SIZE = rollout.input_ids.size(0)
        CHUNK_SIZE = int(args.chunk_size)

        # update model for k_epochs
        model.train()
        for epoch_i in range(args.k_epochs):
            optimizer.zero_grad(set_to_none=True)
            step_stats = {}
            # Inner loop: Split batch into mini-batches for gradient accumulation to save memory
            iterator = range(0, TOTAL_SIZE, CHUNK_SIZE)
            iterator = tqdm.tqdm(iterator, position=RANK, desc=f"[GPU0-{WORLD_SIZE-1}]: Train (Ep {epoch_i+1}/{args.k_epochs})", disable=LOCAL_RANK!=0)
            for i in iterator:
                # Slice RolloutBatch    
                end = min(i + CHUNK_SIZE, TOTAL_SIZE)
                mini_batch = rollout[i:end]

                # Compute GRPO loss for mini-batch
                loss, stats = ppo_loss(
                    model=model,
                    batch=mini_batch,
                    clip_eps=args.clip_eps,
                    kl_beta=args.kl_beta,
                    value_coef=args.value_coef,
                    autocast_dtype=autocast_dtype,
                    ratio_len_norm=args.ratio_len_norm,
                )

                # update progress bar
                stats_tensor = {k: (v.detach() if torch.is_tensor(v) else torch.tensor(v, device=LOCAL_RANK)) for k, v in stats.items()}
                stats_synced = ddp_sync_stats_for_progress(stats_tensor)
                if is_main_rank():
                    iterator.set_postfix(
                        loss=f"{stats_synced['loss'].item():.4f}" if 'loss' in stats_synced else "nan",
                        rwd=f"{stats_synced['reward_mean'].item():.3f}" if 'reward_mean' in stats_synced else "nan",
                        adv=f"{stats_synced['adv_mean'].item():.3f}" if 'adv_mean' in stats_synced else "nan",
                        ratio_mean=f"{stats_synced['ratio_mean'].item():.3f}" if 'ratio_mean' in stats_synced else "nan",
                        ratio_max=f"{stats_synced['ratio_max'].item():.3f}" if 'ratio_max' in stats_synced else "nan",
                    )

                # Scale loss by TOTAL_SIZE
                weight = (end - i) / TOTAL_SIZE
                loss_scaled = loss * weight

                # Backward with DDP gradient synchronization only on the last chunk
                if (i + CHUNK_SIZE) < TOTAL_SIZE:
                    with model.no_sync():
                        loss_scaled.backward()
                else:
                    loss_scaled.backward()

                # Accumulate stats (weighted average)
                for k_stat, v_stat in stats.items():
                    if k_stat not in step_stats:
                        step_stats[k_stat] = 0.0
                    step_stats[k_stat] += v_stat.item() * weight

            # Clip gradients
            if args.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(optim_params, args.max_grad_norm)
                step_stats["grad_norm"] = grad_norm.item()

            # Update stats_last for logging (convert back to tensors)
            stats_last = {k: torch.tensor(v, device=LOCAL_RANK) for k, v in step_stats.items()}

            if not assert_finite_grad(model):
                optimizer.zero_grad(set_to_none=True)
                clean_print(f"Skip step {step} due to non-finite grad", "[WARN]")
                continue
            optimizer.step()
            assert_finite_model(model)

        if stats_last is not None:
            # reduce stats across ranks for logging
            for kk in list(stats_last.keys()):
                stats_last[kk] = ddp_all_reduce_mean(stats_last[kk])

        # ================ (5) log ================
        if is_main_rank() and stats_last is not None:
            dt = time.time() - t0
            t0 = time.time()
            stats_last["dt"] = dt
            pprint({k: v.item() if torch.is_tensor(v) else v for k, v in stats_last.items()})

            if wandb is not None and wandb.run is not None:
                wandb_log_dict.update(
                    {
                        "train/loss": stats_last["loss"].item(),
                        "train/policy_loss": stats_last["policy_loss"].item(),
                        "train/value_loss": stats_last["value_loss"].item(),
                        "train/kl": stats_last["kl"].item(),
                        "train/reward": stats_last["reward_mean"].item(),
                        "train/length": stats_last["len_mean"].item(),
                        "train/ratio": stats_last["ratio_mean"].item(),
                        "train/grad_norm": stats_last["grad_norm"].item(),
                        "train/dt": dt,
                        "step": step,
                    }
                )
                wandb.log(wandb_log_dict)

    dist.barrier()
    clean_print("Done.", "[INFO]")
    if is_main_rank() and wandb is not None and wandb.run is not None:
        wandb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()