from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Config:
    """Training / model configuration. Override paths and scale for real runs."""

    data_path: str = ""
    train_data_path: str = ""
    val_data_path: str = ""
    model_path: str = "model.pt"
    device: str = "cpu"
    # dtype of token ids on disk (memmap); must match pretokenized file format
    dtype: np.dtype = np.dtype(np.uint16)
    # dtype for model weights / activations
    torch_dtype: torch.dtype = torch.float32
    batch_size: int = 32
    context_length: int = 128
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    d_ff: int = 1024
    theta: float = 10_000.0
    num_training_steps: int = 1000
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    vocab_size: int = 50257
    # Full training checkpoint (model + optimizer + step count)
    checkpoint_path: str = "checkpoint.pt"
    # If non-empty, load this checkpoint before training (optimizer + model + start step)
    resume_checkpoint_path: str = ""
    # Save checkpoint every N completed steps; 0 = only save at end of training
    checkpoint_interval: int = 0
    # >0: average CE over `eval_batches` val batches every N steps (and on last step if needed)
    eval_every: int = 100
    eval_batches: int = 32
