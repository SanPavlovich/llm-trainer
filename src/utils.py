import torch
import torch.nn.functional as F
from torch import Tensor


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


def cross_entropy_loss(input_ids: Tensor, attention_mask: Tensor, logits: Tensor) -> Tensor:
    """Calculate Cross-Entropy loss for Language Modeling task
    Under the hood:
    1. Create targtes based on input ids
    2. Masked out tokens corresponded to paddings
    3. Calculate cross entropy loss

    Args:
        input_ids: tensor with input ids, shape [bs, seq len]
        attention_mask: mask with zeros for pad tokens, shape [bs, seq len]
        logits: predicted logits, shape [bs, seq len, vocab size]
    Return:
        cross entropy loss, single-item tensor
    """
    n_logits = logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()
    mask   = attention_mask[:, 1:].contiguous()

    # лосс покадрово без усреднения
    loss_tok = F.cross_entropy(
        n_logits.view(-1, n_logits.size(-1)),
        labels.view(-1),
        reduction="none",
    ).view_as(mask)

    valid = mask.float()
    loss = (loss_tok * valid).sum() / valid.sum().clamp_min(1.0)
    return loss