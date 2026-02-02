#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO (outcome reward) training + evaluation on GSM8K.
- Minimal "NanoGPT-like" loop: rollout -> reward -> GRPO loss -> update -> eval
- Works with torchrun DDP
- Optional: KL to reference model, 8bit optimizer, LoRA, gradient checkpointing, bf16

Notes:
- Reward = 1 if final answer matches, else 0.
- Prompt encourages "#### <answer>" format (GSM8K convention).
"""

import os
import time
import random
import glob
import tqdm
import wandb
import numpy as np
from pprint import pprint
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import *

# ---------------------------
# GRPO rollout
# ---------------------------
@torch.no_grad()
def rollout_grpo(
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
    Generate G samples per prompt. Compute reward and group-normalized advantages.
    Store old_logp_tok from π_old (current model at rollout time).
    Optionally store ref_logp_tok from π_ref (for KL regularization).
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

    # GRPO advantages: group-wise baseline & std normalization if std is non-trivial
    # NOTE that group_answer of same prompt located on same GPU, no need to gather
    B = len(batch_questions)
    r_group = all_chunk_res['rewards'].view(B, group_size)  # [B, G]
    mean = r_group.mean(dim=1, keepdim=True)                # [B, 1]
    std = r_group.std(dim=1, keepdim=True, unbiased=False)  # [B, 1]
    adv = r_group - mean                                    # [B, G]
    adv = torch.where(std > 1e-6, adv / (std + 1e-6), adv)  # [B, G]
    advantages = adv.view(B * group_size)                   # [B*G]
    
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
# GRPO loss
# ---------------------------
def grpo_loss(
    model: nn.Module,
    batch: RolloutBatch,
    clip_eps: float,
    kl_beta: float,
    autocast_dtype: torch.dtype,
    ratio_len_norm: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    PPO-style clipped objective with GRPO advantages, plus KL penalty to ref model.
    Uses sequence-level ratio = exp(sum_logp_new - sum_logp_old).
    KL is estimated on sampled tokens: mean over completion tokens of (logp_new - logp_ref).
    """
    tok_logp_new, _ = compute_token_logps_and_values(       # [N, L-1]
        model=model,
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        autocast_dtype=autocast_dtype,
        require_grad=True,
    )  
    tok_logp_old = batch.old_logp_tok.detach()              # [N, L-1]
    
    # sum logp on completion tokens
    comp_mask = batch.completion_mask                       # [N, L-1]
    logp_new_sum = (tok_logp_new * comp_mask).sum(dim=1)    # [N]
    logp_old_sum = (tok_logp_old * comp_mask).sum(dim=1)    # [N]

    # ratio: support optional length-normalization to stabilize long sequences
    if ratio_len_norm:
        completion_len = comp_mask.sum(dim=1).float().clamp(min=1.0)    # [N]
        log_ratio = (logp_new_sum - logp_old_sum) / completion_len      # [N]
        ratio = torch.exp(log_ratio.clamp(-10, 10))                     # [N]
    else:
        log_ratio = (logp_new_sum - logp_old_sum).clamp(-20, 20)
        ratio = torch.exp(log_ratio)

    # KL reward shaping to reference model (token-level)
    kl_penalty = torch.zeros_like(batch.rewards)                        # [N]
    kl = torch.tensor(0.0, device=ratio.device)
    if batch.ref_logp_tok is not None and kl_beta > 0.0:
        ref_tok = batch.ref_logp_tok.detach()
        if ratio_len_norm:
            completion_len = comp_mask.sum(dim=1).float().clamp(min=1.0)
            kl_per_seq = ((tok_logp_new - ref_tok) * comp_mask).sum(dim=1) / completion_len  # [N]
        else:
            kl_per_seq = ((tok_logp_new - ref_tok) * comp_mask).sum(dim=1)  # [N]
        kl = kl_per_seq.mean()
        kl_penalty = kl_beta * kl_per_seq.detach()

    # clipped surrogate
    A = (batch.advantages - kl_penalty).detach()
    unclipped = ratio * A                                                   # [N]
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * A        # [N]
    clip_mask = (ratio < (1.0 - clip_eps)) | (ratio > (1.0 + clip_eps))     # [N]
    
    # GRPO loss
    policy_loss = -torch.mean(torch.minimum(unclipped, clipped))
    loss = policy_loss

    stats = {
        "loss": loss.detach(),
        "policy_loss": policy_loss.detach(),
        "kl": kl.detach(),
        "ratio_mean": ratio.mean().detach(),
        "ratio_max": ratio.max().detach(),
        "ratio_clip_mask": clip_mask.detach(),
        "reward_mean": batch.rewards.mean().detach(),
        "adv_mean": A.mean().detach(),
        "len_mean": batch.completion_lens.float().mean().detach(),
    }
    return loss, stats

# ---------------------------
# main
# ---------------------------
def parse_args():
    # Load basic args
    p, defaults, remaining_args = parse_basic_args(default_config_path="config/grpo_config.json")
    
    # Group: GRPO
    g_rl = p.add_argument_group("GRPO")
    g_rl.add_argument("--clip_eps", type=float, default=0.2)
    g_rl.add_argument("--kl_beta", type=float, default=0.02)
    g_rl.add_argument("--k_epochs", type=int, default=2)
    g_rl.add_argument("--disable_ref", action="store_true", help="Disable KL regularization by not using reference model.")

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
    set_seed(args.seed + RANK)

    # save setting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_group_name = args.wandb_group_enforce if args.wandb_group_enforce is not None else \
                f'GRPO-{args.model_path.split("/")[-1]}-LoRA' if args.use_lora else \
                f'GRPO-{args.model_path.split("/")[-1]}'
    exp_name = args.wandb_name_enforce if args.wandb_name_enforce is not None else timestamp
    args.save_dir = f"runs/{exp_group_name}/{exp_name}"
    if os.path.exists(args.save_dir) and (args.resume_path is None or args.save_dir not in args.resume_path):
        clean_print(f"Save_dir {args.save_dir} already exists. You can resume training by setting --resume_path.", '[Error]')
        if dist.is_initialized():
            dist.destroy_process_group()
        exit(0)
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, fix_mistral_regex=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # Required for generation to work properly with variable length prompts

    # data
    train_loader, eval_loader, train_sampler, _ = build_data_component(
        args.train_jsonl, args.test_jsonl, args.batch_size, args.eval_batch_size, args.seed
    )
    train_iter = iter(train_loader)

    # model
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=autocast_dtype, low_cpu_mem_usage=True).to(LOCAL_RANK)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.use_lora:
        model = maybe_apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK], find_unused_parameters=False,)   # DDP wrap

    # reference model: frozen copy of initial checkpoint
    ref_model = None
    if (not args.disable_ref) and args.kl_beta > 0.0:
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=autocast_dtype, low_cpu_mem_usage=True).to(LOCAL_RANK)
        ref_model.eval()
        for p_ in ref_model.parameters():
            p_.requires_grad_(False)

    # optimizer: Only optimize trainable parameters (important if LoRA)
    optim_params = [p for p in (model.parameters()) if p.requires_grad]
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
        clean_print(f"Resuming training from snapshot at step {step_begin}", '[Trainer]')
    else:
        # save setting
        if is_main_rank():
            create_folder_if_not_exist(args.save_dir)
            shutil.copy2(src=os.path.abspath(__file__), dst=f"{args.save_dir}/GRPO.py")
            with open('config/grpo_config.json', 'r') as f:
                config_data = json.load(f)
                config_data["Training"]["world_size"] = WORLD_SIZE
            with open(f"{args.save_dir}/grpo_config.json", 'w') as f:
                json.dump(config_data, f, indent=4)
        
        step_begin = 1
        wandb_id = wandb.util.generate_id()     # This unique id is necessary for log resuming
        best_performance = float('-inf')
        clean_print(f"Snapshot not found. Training model from scratch", '[Trainer]')

    # wandb init    
    if is_main_rank() and wandb is not None:
        wandb.init(
            project=args.wandb_project, group=exp_group_name, name=exp_name, 
            config=vars(args), dir=f'Wandb',
            id=wandb_id, resume='allow',
            mode="offline" if args.wandb_offline else "online",
        )

    # training loop
    model.train()
    t0 = time.time()
    for step in range(step_begin, args.total_steps + 1):
        wandb_log_dict = {}
        train_sampler.set_epoch(step)
        clean_print('-'*50 + f' RL Step [{step}] ' + '-'*50)
    
        # ================ (1) eval before training to check initial peformance ================
        if (step % args.eval_every == 0) or (step == 1 and not args.eval_skip_first):
            torch.cuda.empty_cache()    # Free memory
            model.eval()                # eval model for inference
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
                    ckpt_path = os.path.join(f'{args.save_dir}/best', f"{round(best_performance,3)}_seed{args.seed}_step{step}")
                    os.makedirs(ckpt_path, exist_ok=True)
                    save_model = model.module if hasattr(model, "module") else model    # unwrap DDP
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
                "wandb_id": (wandb.run.id if (wandb is not None and wandb.run is not None) else None),
                "best_performance": best_performance,
                "rng_states": {
                    "python": random.getstate(),
                    "numpy": (np.random.get_state() if np is not None else None),
                    "torch": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all(),
                },
            }
            torch.save(snapshot, os.path.join(args.save_dir, f"snapshot_seed{args.seed}.pt"))

            # save ckpt at every `args.save_every` steps
            if args.save_every is not None and (step % args.save_every == 0):
                ckpt_path = os.path.join(f'{args.save_dir}/interval', f"seed{args.seed}_step{step}")
                os.makedirs(ckpt_path, exist_ok=True)
                save_model = model.module if hasattr(model, "module") else model    # unwrap DDP
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
        rollout = rollout_grpo(
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
        TOTAL_SIZE = rollout.input_ids.size(0)  # B*G (batch_size * group_size)
        CHUNK_SIZE = int(args.chunk_size)       # mini_batch_size
        ratio_clip_sum_total = torch.tensor(0.0, device=LOCAL_RANK)
        ratio_clip_count_total = torch.tensor(0.0, device=LOCAL_RANK)

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
                loss, stats = grpo_loss(
                    model=model,
                    batch=mini_batch,
                    clip_eps=args.clip_eps,
                    kl_beta=args.kl_beta,
                    autocast_dtype=autocast_dtype,
                    ratio_len_norm=args.ratio_len_norm,
                )
                ratio_clip_mask = stats["ratio_clip_mask"]
                ratio_clip_sum_total += ratio_clip_mask.float().sum().detach()
                ratio_clip_count_total += torch.tensor(float(ratio_clip_mask.numel()), device=LOCAL_RANK).detach()
                
                # update progress bar
                stats_tensor = {k: (v.detach() if torch.is_tensor(v) else torch.tensor(v, device=LOCAL_RANK)) for k, v in stats.items() if k != "ratio_clip_mask"}
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
                    if k_stat in ["ratio_clip_mask"]:
                        continue
                    if k_stat not in step_stats:
                        step_stats[k_stat] = 0.0
                    step_stats[k_stat] += v_stat.item() * weight

            # Clip gradients
            if args.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(optim_params, args.max_grad_norm)
                step_stats['grad_norm'] = grad_norm.item()
            
            # Update stats_last for logging (convert back to tensors)
            stats_last = {k: torch.tensor(v, device=LOCAL_RANK) for k, v in step_stats.items()}
            
            # flush the gradients and skip update if non-finite grad exists
            if not assert_finite_grad(model):
                optimizer.zero_grad(set_to_none=True)
                clean_print(f"Skip step{step}-epoch{epoch_i} due to non-finite grad", "[WARN]")
                continue
            optimizer.step()
            assert_finite_model(model)

        ratio_clip_sum_total = ddp_all_reduce_sum(ratio_clip_sum_total)
        ratio_clip_count_total = ddp_all_reduce_sum(ratio_clip_count_total)
        ratio_clip_frac_step = ratio_clip_sum_total / ratio_clip_count_total.clamp(min=1.0)

        if stats_last is not None:
            # reduce stats across ranks for logging
            for kk in list(stats_last.keys()):
                stats_last[kk] = ddp_all_reduce_mean(stats_last[kk])
            stats_last["ratio_clip_frac"] = ratio_clip_frac_step

        # ================ (5) log ================
        if is_main_rank() and stats_last is not None:
            dt = time.time() - t0
            t0 = time.time()
            stats_last['dt'] = dt
            pprint({k: v.item() if torch.is_tensor(v) else v for k, v in stats_last.items()})
            
            if wandb is not None and wandb.run is not None:
                wandb_log_dict.update({
                    "train/loss": stats_last['loss'].item(),
                    "train/policy_loss": stats_last['policy_loss'].item(),
                    "train/kl": stats_last['kl'].item(),
                    "train/reward": stats_last['reward_mean'].item(),
                    "train/length": stats_last['len_mean'].item(),
                    "train/ratio": stats_last['ratio_mean'].item(),
                    "train/ratio_clip_frac": stats_last['ratio_clip_frac'].item(),
                    "train/grad_norm": stats_last['grad_norm'].item(),
                    "train/dt": dt,
                    "step": step
                })
                wandb.log(wandb_log_dict)

    dist.barrier()
    clean_print("Done.", "[INFO]")
    if is_main_rank() and wandb is not None and wandb.run is not None:
        wandb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
