import os
import re
import json
import random
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, List

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

def compute_token_logps(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    autocast_dtype: torch.dtype,
    require_grad: bool,
    chunk_size: int = 4,
) -> torch.Tensor:
    """
    Return token log-probabilities for each position in targets (input_ids[:,1:]).
    Shape: [B*G, max_traj_L-1]
    If require_grad is False, splits input into chunks to save memory.
    """
    if not require_grad:
        # Chunked inference to save memory
        B_full = input_ids.shape[0]
        all_tok_logps = []
        
        for i in range(0, B_full, chunk_size):
            chunk_ids = input_ids[i : i + chunk_size]               # [chunk_B, L]
            chunk_mask = attention_mask[i : i + chunk_size]         # [chunk_B, L]
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=autocast_dtype):
                out = model(input_ids=chunk_ids, attention_mask=chunk_mask, use_cache=False)
                logits = out.logits                                 # [chunk_B, L, V]
                logp = F.log_softmax(logits[:, :-1, :], dim=-1)     # [chunk_B, L-1, V]
                tgt = chunk_ids[:, 1:]                              # [chunk_B, L-1]
                tok_logp = torch.gather(logp, dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)  # [chunk_B, L-1]
                all_tok_logps.append(tok_logp)
        
        all_tok_logps = torch.cat(all_tok_logps, dim=0)             # [B, L-1]
        return all_tok_logps
    else:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = out.logits # [B, L, V]

        logp = F.log_softmax(logits[:, :-1, :], dim=-1)             # [B, L-1, V]
        tgt = input_ids[:, 1:]                                      # [B, L-1]
        tok_logp = torch.gather(logp, dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
        return tok_logp
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

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clean_print(str:str, prefix:str=''):
    if is_main_rank():
        print("\r\033[K", end="")   # 使用 ANSI 转义码清空当前行
        print(prefix + '\t' + str if prefix != '' else str)

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

def prepare_prompt(questions: List[str], tokenizer):
    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q}
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    
    return prompts