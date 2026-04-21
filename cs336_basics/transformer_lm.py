"""
Implement a transformer LM, CS336 assignment 1 section 3.6
"""
from __future__ import annotations

import torch
from torch import nn
from cs336_basics.nn import Linear, Embedding, TransformerBlock, RMSNorm


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int,
        theta: float, eps: float = 1e-5,
        device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta

        self.emb = Embedding(vocab_size, d_model, device, dtype)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, theta, eps, device, dtype)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model, eps, device, dtype)
        self.linear = Linear(d_model, vocab_size, device, dtype)

    
    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        self.emb.set_weights(weights["token_embeddings.weight"])
        layer_keys = (
            "ln1.weight",
            "attn.q_proj.weight",
            "attn.k_proj.weight",
            "attn.v_proj.weight",
            "attn.output_proj.weight",
            "ln2.weight",
            "ffn.w1.weight",
            "ffn.w2.weight",
            "ffn.w3.weight",
        )
        for i, layer in enumerate(self.transformer_layers):
            prefix = f"layers.{i}."
            layer.set_weights({k: weights[f"{prefix}{k}"] for k in layer_keys})
        self.norm.set_weights(weights["ln_final.weight"])
        self.linear.set_weights(weights["lm_head.weight"])
    
    
    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            in_indices: LongTensor of shape (batch, seq_len)

        Returns:
            logits: Tensor of shape (batch, seq_len, vocab_size)
        """
        if in_indices.dim() != 2:
            raise ValueError(f"in_indices must have shape (batch, seq_len), got {tuple(in_indices.shape)}")

        batch, seq_len = in_indices.shape
        if seq_len > self.context_length:
            raise ValueError(f"seq_len={seq_len} exceeds context_length={self.context_length}")
        
        emb = self.emb(in_indices)  # (batch, seq_len, d_model)
        positions = torch.arange(seq_len, device=in_indices.device).unsqueeze(0).expand(batch, seq_len)
        x = emb
        for layer in self.transformer_layers:
            x = layer(x, positions)
        norm_out = self.norm(x)
        return self.linear(norm_out)



        

