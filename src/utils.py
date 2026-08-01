import os
import random
import time
from functools import wraps

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"Function '{func.__name__}' took {duration:.4f} seconds to run.")
        return result
    return wrapper


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Fix random seeds across python, numpy and torch for reproducibility.

    Args:
        seed: seed value to set everywhere
        deterministic: if True, force deterministic (and slower) CUDA/cuDNN
            algorithms so runs are bit-for-bit reproducible. Useful for
            strictly tracking loss convergence when adding a new feature.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # cuBLAS reproducibility for matmul on CUDA >= 10.2
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def is_distributed() -> bool:
    """True when running under torchrun (i.e. RANK/WORLD_SIZE are set)."""
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def setup_distributed() -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps) -> torch.optim.lr_scheduler.LRScheduler:
    """Scheduler for Optimizer with linear warmup and linear decay to the end of training

    Args:
        optimizer: torch optimizer to control learning rate
        num_warmup_steps: number of warmup steps
        num_training_steps: total number of training steps
    Return:
        torch learning rate scheduler
    """
    assert num_training_steps >= num_warmup_steps

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        return max(
            0.0,
            float(num_training_steps - current_step) / max(1, num_training_steps - num_warmup_steps),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def cross_entropy_loss(
    input_ids: Tensor,
    attention_mask: Tensor,
    logits: Tensor,
    loss_mask: Tensor | None = None,
) -> Tensor:
    """Calculate Cross-Entropy loss for Language Modeling task
    Under the hood:
    1. Create targtes based on input ids
    2. Masked out tokens corresponded to paddings
    3. Calculate cross entropy loss

    Args:
        input_ids: tensor with input ids, shape [bs, seq len]
        attention_mask: mask with zeros for pad tokens, shape [bs, seq len]
        logits: predicted logits, shape [bs, seq len, vocab size]
        loss_mask: optional bool mask, True where a target position should be INCLUDED in the loss.
    Return:
        cross entropy loss, single-item tensor
    """
    n_logits = logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()
    mask   = attention_mask[:, 1:].contiguous()

    if loss_mask is not None:
        # Only keep positions that are both non-pad AND flagged for loss.
        mask = mask & loss_mask[:, 1:].contiguous().to(mask.dtype).bool()

    # лосс покадрово без усреднения
    loss_tok = F.cross_entropy(
        n_logits.view(-1, n_logits.size(-1)),
        labels.view(-1),
        reduction="none",
    ).view_as(mask)

    valid = mask.float()
    loss = (loss_tok * valid).sum() / valid.sum().clamp_min(1.0)
    return loss