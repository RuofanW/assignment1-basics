from __future__ import annotations

import os

import numpy as np
import torch

from .config import Config
from .training_utils import (
    AdamW,
    cross_entropy_loss,
    data_loading,
    load_checkpoint,
    save_checkpoint,
)
from .transformer_lm import TransformerLM


def load_data(data_path: str, dtype: np.dtype) -> np.memmap:
    return np.memmap(data_path, dtype=dtype, mode="r")


def create_model(config: Config) -> torch.nn.Module:
    return TransformerLM(
        config.vocab_size,
        config.context_length,
        config.d_model,
        config.num_layers,
        config.num_heads,
        config.d_ff,
        config.theta,
        device=torch.device(config.device),
        dtype=config.torch_dtype,
    )


def create_optimizer(config: Config, model: torch.nn.Module) -> torch.optim.Optimizer:
    return AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas,
        eps=config.eps,
    )


@torch.no_grad()
def validate(model: torch.nn.Module, data_val: np.memmap, config: Config) -> float | None:
    if len(data_val) < config.context_length + 2 or config.eval_batches <= 0:
        return None
    was_training = model.training
    model.eval()
    try:
        total = 0.0
        for _ in range(config.eval_batches):
            x, y = data_loading(
                data_val,
                config.batch_size,
                config.context_length,
                config.device,
            )
            total += cross_entropy_loss(model(x.long()), y.long()).item()
        return total / config.eval_batches
    finally:
        if was_training:
            model.train()


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_train: np.memmap,
    data_val: np.memmap,
    config: Config,
    *,
    start_iteration: int = 0,
) -> int:
    model.train()
    for it in range(start_iteration, config.num_training_steps):
        x, y = data_loading(
            data_train, config.batch_size, config.context_length, config.device
        )
        x, y = x.long(), y.long()

        optimizer.zero_grad(set_to_none=True)
        loss = cross_entropy_loss(model(x), y)
        loss.backward()
        optimizer.step()

        step = it + 1
        if config.checkpoint_interval and step % config.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, step, config.checkpoint_path)

        if config.eval_every and (
            step % config.eval_every == 0 or step == config.num_training_steps
        ):
            v = validate(model, data_val, config)
            if v is not None:
                print(f"step {step}  val_loss={v:.4f}")

    save_checkpoint(model, optimizer, config.num_training_steps, config.checkpoint_path)
    return config.num_training_steps


def save_model(model: torch.nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def main() -> None:
    config = Config()
    data_train = load_data(config.train_data_path, config.dtype)
    data_val = load_data(config.val_data_path, config.dtype)

    model = create_model(config)
    optimizer = create_optimizer(config, model)

    resume = config.resume_checkpoint_path
    start = (
        load_checkpoint(resume, model, optimizer)
        if resume and os.path.isfile(resume)
        else 0
    )

    train_model(model, optimizer, data_train, data_val, config, start_iteration=start)
    save_model(model, config.model_path)


if __name__ == "__main__":
    main()
