from training_utils import AdamW, cross_entropy_loss, cross_entropy_loss_stable, learning_rate_cosine_schedule, gradient_clipping, data_loading, save_checkpoint, load_checkpoint
from config import Config
from transformer_lm import TransformerLM

import numpy as np

def load_data(data_path: str, dtype: np.dtype) -> np.memmap:
    # mode="r" = read-only; dtype must match how the file was written
    data = np.memmap(data_path, dtype=dtype mode="r")
    return data  # 1-D memmap, use like a normal 1d array for slicing / batching

def create_model(config: Config) -> torch.nn.Module:
    model = TransformerLM(
        config.vocab_size, config.context_length, config.d_model, config.num_layers, config.num_heads, config.d_ff,
        config.theta, device=config.device, dtype=config.dtype
    )
    return model

def create_optimizer(config: Config, model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas,
        eps=config.eps
    )
    return optimizer

def train_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, data_train: np.memmap, data_val: np.memmap, config: Config):
    for epoch in range(config.num_epochs):
        

def main():
    
    # Load config
    config = Config()
    # Load data
    data_train = load_data(config.train_data_path, config.dtype)
    data_val = load_data(config.val_data_path, config.dtype)

    # Create model
    model = create_model(config)

    # Create optimizer
    optimizer = create_optimizer(config, model)

    # Train model
    train_model(model, optimizer, scheduler, data_train, data_val, config)

if __name__ == "__main__":
    main()