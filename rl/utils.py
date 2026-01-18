import os
import re
import json
import random
import tqdm
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Union, Tuple
from transformers import AutoTokenizer
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from dataclasses import dataclass

# ---------------------------
# utils: basic
# ---------------------------
def create_folder_if_not_exist(floder_path):
    os.makedirs(floder_path, exist_ok=True)

def create_folder_overwrite_if_exist(floder_path):
    if os.path.exists(floder_path):
        shutil.rmtree(floder_path)    
    create_folder_if_not_exist(floder_path)

def assert_finite_model(model: nn.Module):
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            x = p.detach().view(-1)
            isnan = torch.isnan(x)
            isinf = torch.isinf(x)
            msg = f"Found NaN/Inf in parameter {name}: "
            msg += f"nan={isnan.any().item()}, inf={isinf.any().item()}"
            raise RuntimeError(msg)

def assert_finite_grad(model: nn.Module) -> bool:
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            x = p.grad.detach().view(-1)
            isnan = torch.isnan(x)
            isinf = torch.isinf(x)
            msg = f"Found NaN/Inf in grad of {name}: "
            msg += f"nan={isnan.any().item()}, inf={isinf.any().item()}"
            clean_print(msg, "[WARN]")
            return False
    return True

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clean_print(str:str, prefix:str=''):
    if is_main_rank():
        print("\r\033[K", end="")   # 使用 ANSI 转义码清空当前行
        print(prefix + '\t' + str if prefix != '' else str)

# ---------------------------
# utils: DDP + logging
# ---------------------------
def ddp_is_on() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ

def is_main_rank() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def ddp_all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if not ddp_is_on():
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y

def ddp_all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if not ddp_is_on():
        return x
    return ddp_all_reduce_sum(x) / int(os.environ.get("WORLD_SIZE", "1"))

def ddp_all_reduce_max(x: torch.Tensor) -> torch.Tensor:
    if not ddp_is_on():
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.MAX)
    return y

def ddp_sync_stats_for_progress(stats: dict) -> dict:
    out = {}
    for k, v in stats.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k in ["loss", "reward_mean", "adv_mean", "len_mean", "kl", "ratio_mean", "grad_norm"]:
            out[k] = ddp_all_reduce_mean(v)
        elif k in ["ratio_max"]:
            out[k] = ddp_all_reduce_max(v)
    return out

def ddp_init():
    assert ddp_is_on(), "Only DDP mode is supported"
    num_cores = os.cpu_count()
    num_threads = max(1, min(16, num_cores // 2))    # Each process uses part of the CPU cores
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    RANK = int(os.environ["RANK"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    torch.cuda.set_device(LOCAL_RANK)

    return RANK, LOCAL_RANK, WORLD_SIZE

# ---------------------------
# GSM8K: load + parse
# ---------------------------
_ANS_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
_LAST_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")
SYSTEM_PROMPT = (
    "You are a helpful assistant that solves math word problems.\n"
    "Solve the problem step by step, and give the final answer in the format '#### <number>'."
)

class GSM8KJsonl(Dataset):
    def __init__(self, path: str):
        self.items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # expected keys: question, answer
                self.items.append({"question": obj["question"], "answer": obj["answer"]})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]

def build_data_component(train_jsonl, test_jsonl, batch_size, eval_batch_size, seed):
    train_set = GSM8KJsonl(train_jsonl)
    test_set = GSM8KJsonl(test_jsonl)
    train_sampler = DistributedSampler(train_set, shuffle=True, seed=seed)
    eval_sampler = DistributedSampler(test_set, shuffle=False, seed=seed)  # no shuffle to fix eval subset
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda x: x, # keep raw strings; build prompts later
    )
    eval_loader = DataLoader(
        test_set,
        batch_size=eval_batch_size,
        sampler=eval_sampler,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    return train_loader, eval_loader, train_sampler, eval_sampler

def extract_final_number(text: str) -> Optional[str]:
    """
    GSM8K official answers contain '#### <number>'.
    If not found, fall back to last number in the text.
    Return normalized string (remove trailing .0).
    """
    m = _ANS_RE.search(text)
    if m:
        s = m.group(1)
    else:
        nums = _LAST_NUM_RE.findall(text)
        if not nums:
            return None
        s = nums[-1]
    # normalize 12.0 -> 12
    if s.endswith(".0"):
        s = s[:-2]
    return s

def gsm8k_reward(pred_text: str, gt_answer: str) -> float:
    pred = extract_final_number(pred_text)
    gt = extract_final_number(gt_answer)
    return 1.0 if (pred is not None and gt is not None and pred == gt) else 0.0

def prepare_prompts_tok(questions: List[str], tokenizer):
    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q}
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    
    prompts_tok = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt", max_length=tokenizer.model_max_length)
    return prompts_tok

# ---------------------------
# utils: model + optimizer
# ---------------------------
def build_optimizer(params, lr: float, weight_decay: float, optim_name: str):
    if optim_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if optim_name == "adamw8bit":
        import bitsandbytes as bnb  # type: ignore
        return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {optim_name}")

def maybe_apply_lora(model, r: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model, TaskType  # type: ignore
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,                    # LoRA 的秩（Rank），决定了可训练参数量的大小，通常设为 8, 16, 32 等
        lora_alpha=alpha,       # 缩放系数，用于调整 LoRA 权重的贡献程度
        lora_dropout=dropout,
        bias="none",            # 是否训练 bias 参数，"none" 表示全部冻结，只训练 LoRA 矩阵
        target_modules=None,    # 目标模块：可指定 q_proj,k_proj,v_proj,o_proj。设置为 None 时 PEFT 库自动推断需要注入 LoRA 的层
    )
    model = get_peft_model(model, cfg)  # 将原始模型转换为 PEFT 模型，这会冻结原始模型的大部分参数，只插入并激活 LoRA 层的参数
    model.print_trainable_parameters()
    return model

def build_completion_mask_targets(
    attention_mask: torch.Tensor, prompt_lens: torch.Tensor
) -> torch.Tensor:
    """
    For targets positions (1..L-1), completion tokens start at j>=prompt_len.
    In targets indexing (j-1), mask positions >= prompt_len-1.
    Also exclude padding via attention_mask[:,1:].
    Returns bool mask [B, L-1]
    """
    B, L = attention_mask.shape
    pos = torch.arange(L - 1, device=attention_mask.device).unsqueeze(0).expand(B, -1)  # [B*G, L-1]
    start = (prompt_lens - 1).clamp(min=0).unsqueeze(1)     # [B*G, 1]
    mask = (pos >= start) & (attention_mask[:, 1:] == 1)    # [B*G, L-1]
    return mask

def compute_token_logps_and_values(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    autocast_dtype: torch.dtype,
    require_grad: bool,
    chunk_size: int = 4,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Return token log-probabilities for each next-token target (input_ids[:, 1:]), predicted by logits[:, :-1].
    Shape: [B*G, max_traj_L-1]
    If require_grad is False, splits input into chunks to save memory.
    """
    calc_values = hasattr(model, "v_head")
    if not require_grad:
        # Chunked inference to save memory
        B_full = input_ids.shape[0]
        all_tok_logps = []
        all_values = [] if calc_values else None
        for i in range(0, B_full, chunk_size):
            chunk_ids = input_ids[i : i + chunk_size]               # [chunk_B, L]
            chunk_mask = attention_mask[i : i + chunk_size]         # [chunk_B, L]
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=autocast_dtype):
                out = model(input_ids=chunk_ids, attention_mask=chunk_mask, output_hidden_states=calc_values, use_cache=False,)
                logits = out.logits                                 # [chunk_B, L, V]
                logp = F.log_softmax(logits[:, :-1, :], dim=-1)     # [chunk_B, L-1, V]
                tgt = chunk_ids[:, 1:]                              # [chunk_B, L-1]
                tok_logp = torch.gather(logp, dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)  # [chunk_B, L-1]
                all_tok_logps.append(tok_logp)
                if calc_values:
                    # find the idx of last vaild token
                    hidden = out.hidden_states[-1]
                    L = chunk_mask.size(1)
                    pos = torch.arange(L, device=hidden.device).unsqueeze(0)    # [1, L]
                    last_idx = (chunk_mask.to(torch.long) * pos).amax(dim=1)    # [chunk_B]
                    
                    # project hidden states to values
                    hidden_last = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]
                    all_values.append(model.v_head(hidden_last).squeeze(-1))   
        
        all_tok_logps = torch.cat(all_tok_logps, dim=0)                     # [B, L-1]
        all_values = torch.cat(all_values, dim=0) if all_values else None   # [B]    
        return all_tok_logps, all_values
    else:
        values = None
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=calc_values, use_cache=False)
            logits = out.logits                             # [B, L, V]

        logp = F.log_softmax(logits[:, :-1, :], dim=-1)     # [B, L-1, V]
        tgt = input_ids[:, 1:]                              # [B, L-1]
        tok_logp = torch.gather(logp, dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
        
        if calc_values:    
            # find the idx of last vaild token
            hidden = out.hidden_states[-1]
            L = attention_mask.size(1)
            pos = torch.arange(L, device=hidden.device).unsqueeze(0)        # [1, L]
            last_idx = (attention_mask.to(torch.long) * pos).amax(dim=1)    # [B]

            # project hidden states to values
            hidden_last = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]
            values = model.v_head(hidden_last).squeeze(-1)
        return tok_logp, values



# ---------------------------
# utils: rollout + eval
# ---------------------------
@dataclass
class RolloutBatch:
    # flattened batch size N = B * G
    input_ids: torch.Tensor                 # [B*G, max_traj_len]
    attention_mask: torch.Tensor            # [B*G, max_traj_len]
    old_logp_tok: torch.Tensor              # [B*G, max_traj_len-1] token logp under π_old (for ratio)
    ref_logp_tok: Optional[torch.Tensor]    # [B*G, max_traj_len-1] token logp under π_ref (for KL); None if disabled
    completion_mask: torch.Tensor           # [B*G, max_traj_len-1] (completion token mask)
    completion_lens: torch.Tensor           # [B*G] (number of completion tokens)
    prompt_lens: torch.Tensor               # [B*G] (prompt boundary index, used for masking)
    rewards: torch.Tensor                   # [B*G]
    advantages: torch.Tensor                # [B*G]
    
    def __getitem__(self, index):
        return RolloutBatch(
            input_ids=self.input_ids[index],
            attention_mask=self.attention_mask[index],
            old_logp_tok=self.old_logp_tok[index],
            ref_logp_tok=self.ref_logp_tok[index] if self.ref_logp_tok is not None else None,
            completion_mask=self.completion_mask[index],
            completion_lens=self.completion_lens[index],
            prompt_lens=self.prompt_lens[index],
            rewards=self.rewards[index],
            advantages=self.advantages[index],
        )

@torch.no_grad()
def rollout(
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
    prompt_batch_size: int = 4,
) -> Dict[str, torch.Tensor]:
    """
    Generate G samples per prompt. Compute reward and group-normalized advantages.
    Store old_logp_sum from π_old (current model at rollout time).
    Optionally store ref token logp (for KL regularization).
    """
    # Build & Tokenize all prompts 
    prompts_tok = prepare_prompts_tok(batch_questions, tokenizer)
    all_input_ids = prompts_tok["input_ids"]
    all_attention_mask = prompts_tok["attention_mask"]

    # Generate in chunks to save memory
    rank, B = int(os.environ.get("RANK", "0")), len(batch_questions)
    iterator = tqdm.tqdm(range(0, B, prompt_batch_size), desc=f"[Rank {rank}] Rollout", leave=False, position=rank)
    chunk_results = []
    for start_idx in iterator:
        chunk_res = {}

        # Slice inputs and move to device
        end_idx = min(start_idx + prompt_batch_size, B)
        a_chunk = batch_answers[start_idx:end_idx]
        input_ids = all_input_ids[start_idx:end_idx].to(device)                 # [chunk_B, chunk_prompt_boundary]
        attention_mask = all_attention_mask[start_idx:end_idx].to(device)       # [chunk_B, chunk_prompt_boundary]
        
        # Repeat
        chunk_B, chunk_prompt_boundary = input_ids.size()
        input_ids_rep = input_ids.repeat_interleave(group_size, dim=0)          # [chunk_B*G, chunk_prompt_boundary]
        attn_rep = attention_mask.repeat_interleave(group_size, dim=0)          # [chunk_B*G, chunk_prompt_boundary]
        prompt_boundary_rep = torch.full((chunk_B * group_size,), chunk_prompt_boundary, device=device, dtype=torch.long) # [chunk_B*G,]
        
        # Generate
        gen_model = model.module if hasattr(model, "module") else model
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            out_ids = gen_model.generate(                           # [chunk_B*G, max_traj_L]
                input_ids=input_ids_rep,
                attention_mask=attn_rep,
                num_return_sequences=1,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            out_attn = (out_ids != tokenizer.pad_token_id).long()   # [chunk_B*G, max_traj_L]
            texts = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            
        del input_ids, input_ids_rep, attn_rep  # Free memory
        torch.cuda.empty_cache()                # Free memory
        
        # Process outputs
        rewards = []
        for i in range(chunk_B):
            for g in range(group_size):
                idx = i * group_size + g
                r = gsm8k_reward(texts[idx], a_chunk[i])
                rewards.append(r)
        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)   # [chunk_B*G]

        # Old logp
        old_logp_tok, critic_values = compute_token_logps_and_values(
            model=gen_model,
            input_ids=out_ids,
            attention_mask=out_attn,
            autocast_dtype=autocast_dtype,
            require_grad=False,
        )
        
        # completion mask
        comp_mask = build_completion_mask_targets(out_attn, prompt_boundary_rep).to(dtype=old_logp_tok.dtype)
        completion_lens = comp_mask.sum(dim=1)

        # Store results
        chunk_res.update({
            "out_ids": out_ids,                 # [chunk_B*G, max_traj_L]
            "out_attn": out_attn,               # [chunk_B*G, max_traj_L]
            "old_logp_tok": old_logp_tok,       # [chunk_B*G, max_traj_L-1]
            "rewards": rewards_t,               # [chunk_B*G, ]
            "prompt_lens": prompt_boundary_rep, # [chunk_B*G, ]
            "completion_lens": completion_lens, # [chunk_B*G, ]
            "completion_mask": comp_mask,       # [chunk_B*G, max_traj_L-1]
        })
        
        # Extra: Ref logp
        if ref_model is not None and kl_beta > 0.0:
            ref_logp_tok, _ = compute_token_logps_and_values( 
                model=ref_model,
                input_ids=out_ids,
                attention_mask=out_attn,
                autocast_dtype=autocast_dtype,
                require_grad=False,
            )
            chunk_res.update({
                "ref_logp_tok": ref_logp_tok    # [chunk_B*G, max_traj_L-1]
            })
        # Extra: Critic values
        if critic_values is not None:
            chunk_res.update({
                "critic_values": critic_values  # [chunk_B*G, ]
            })

        chunk_results.append(chunk_res)
    
    # Combine results and pad to max_traj_len
    max_traj_len = max(res["out_ids"].size(1) for res in chunk_results)  
    all_chunk_res = {k: [] for k in chunk_results[0].keys()}
    for res in chunk_results:
        # Pad to max_traj_len
        pad_len = max_traj_len - res["out_ids"].size(1)
        if pad_len > 0:
            res['out_ids'] = F.pad(res["out_ids"], (0, pad_len), value=tokenizer.pad_token_id)
            res['out_attn'] = F.pad(res["out_attn"], (0, pad_len), value=0)
            res['old_logp_tok'] = F.pad(res["old_logp_tok"], (0, pad_len), value=0.0)
            res['completion_mask'] = F.pad(res["completion_mask"], (0, pad_len), value=0)
            if "ref_logp_tok" in res:
                res["ref_logp_tok"] = F.pad(res["ref_logp_tok"], (0, pad_len), value=0.0)

        for k, v in res.items():
            all_chunk_res[k].append(v)

    all_chunk_res = {k: torch.cat(v, dim=0) for k, v in all_chunk_res.items()}
    return all_chunk_res

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

    # Only show progress bar on main rank
    correct_cnt, total_cnt = 0, 0
    total_gen_len = 0.0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    iterator = dataloader if not is_main_rank() else tqdm.tqdm(dataloader, desc=f"[GPU0-{world_size-1}]: Evaluating", leave=False)
    for b, items in enumerate(iterator):
        if eval_batches is not None and b >= eval_batches:
            break

        questions = list(items["question"])
        answers = list(items["answer"])
        
        prompts_tok = prepare_prompts_tok(questions, tokenizer)
        input_ids = prompts_tok["input_ids"].to(device)
        attn = prompts_tok["attention_mask"].to(device)

        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            out = model_eval.generate(
                input_ids=input_ids,    # [eval_B, L]
                attention_mask=attn,    # [eval_B, L]
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
        input_valid = (input_ids != pad_id).sum(dim=1)      # [eval_B]
        out_valid = (out != pad_id).sum(dim=1)              # [eval_B]
        comp_len = (out_valid - input_valid).clamp(min=0)   # [eval_B]
        total_gen_len += comp_len.sum().item()

        texts = tokenizer.batch_decode(out, skip_special_tokens=True)
        for t, gt in zip(texts, answers):
            r = gsm8k_reward(t, gt)
            correct_cnt += int(r > 0.5)   # gsm8k_reward ∈ {0.0,1.0}，用 >0.5 保证可扩展性（比如做了平滑）
            total_cnt += 1

    corr_t = torch.tensor([correct_cnt], device=device, dtype=torch.float32)
    tot_t = torch.tensor([total_cnt], device=device, dtype=torch.float32)
    len_t = torch.tensor([total_gen_len], device=device, dtype=torch.float32)
    corr_t = ddp_all_reduce_sum(corr_t)
    tot_t = ddp_all_reduce_sum(tot_t)
    len_t = ddp_all_reduce_sum(len_t)

    acc = (corr_t / tot_t).item() if tot_t.item() > 0 else 0.0
    avg_len = (len_t / tot_t).item() if tot_t.item() > 0 else 0.0
    return {"greedy_acc": acc, "avg_completion_len": avg_len}
