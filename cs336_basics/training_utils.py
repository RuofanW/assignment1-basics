"""
Training utilities — CS336 assignment 1 section 4.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, Callable, Optional
from typing import Iterable, BinaryIO, IO, Union
import os
import torch


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross-entropy loss between logits and targets.
    logits: (..., d)
    targets: (...,)
    """
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    idx = torch.arange(logits.shape[0], device=logits.device)
    nll = -(logits[idx, targets] - torch.logsumexp(logits, dim=-1))
    return nll.mean()


def cross_entropy_loss_stable(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Alias for numerically stable cross-entropy (log-sum-exp)."""
    return cross_entropy_loss(logits, targets)


class AdamW(torch.optim.Optimizer):
    """AdamW matching PyTorch's decoupled weight decay and bias correction."""

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # Decoupled weight decay (same order as torch.optim.AdamW)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def learning_rate_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Linear warmup for it in [0, warmup_iters], then cosine decay from max to min
    for it in (warmup_iters, cosine_cycle_iters], then min thereafter.

    Cosine uses t = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters), so
    t = 0 at the end of warmup and t = 1 when it == cosine_cycle_iters (matches
    course tests with warmup_iters=7, cosine_cycle_iters=21).
    """
    if it < warmup_iters:
        return max_learning_rate * it / max(warmup_iters, 1)
    if it <= cosine_cycle_iters:
        span = max(cosine_cycle_iters - warmup_iters, 1)
        t = (it - warmup_iters) / span
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
            1 + math.cos(math.pi * t)
        )
    return min_learning_rate

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Global L2 norm clipping: same scaling as torch.nn.utils.clip_grad_norm_(..., 2)."""
    if max_l2_norm <= 0:
        return
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.linalg.vector_norm(
        torch.cat([g.detach().flatten() for g in grads]),
        ord=2,
    )
    if total_norm.isfinite() and total_norm > max_l2_norm:
        scale = max_l2_norm / total_norm
        for g in grads:
            g.mul_(scale)

def data_loading(data:np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load a batch of data from the dataset.
    data: np.ndarray
    batch_size: int
    context_length: int
    device: str
    """
    data_length = len(data)
    valid_starting_indices = torch.arange(data_length - context_length - 1)
    sampled_starting_indices = torch.randint(0, data_length - context_length, (batch_size,)).float()
    x = torch.stack([torch.Tensor(data[int(i):int(i)+context_length]) for i in sampled_starting_indices])
    y = torch.stack([torch.Tensor(data[(int(i)+1):(int(i)+context_length+1)]) for i in sampled_starting_indices])
    return x.to(device), y.to(device)


PathOrFile = Union[str, os.PathLike, BinaryIO, IO[bytes]]

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: PathOrFile
) -> None:
    """
    Save a training checkpoint containing model/optimizer state and iteration.

    Args:
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
        iteration: Current training iteration (step).
        out: File path or a binary file-like object.
    """
    obj = {
        "iteration": int(iteration),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(obj, out)    

def load_checkpoint(
    src: PathOrFile,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """
    Load a training checkpoint and restore model/optimizer state.

    Args:
        src: File path or a binary file-like object.
        model: torch.nn.Module to restore into.
        optimizer: torch.optim.Optimizer to restore into.

    Returns:
        The iteration (step) stored in the checkpoint.
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    ckpt = torch.load(src, map_location=device, weights_only=False)

    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint must be a dict.")
    
    if "model_state_dict" not in ckpt or "optimizer_state_dict" not in ckpt or "iteration" not in ckpt:
        raise KeyError("Checkpoint dict missing required keys.")

    model.load_state_dict(ckpt["model_state_dict"])    
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt["iteration"])