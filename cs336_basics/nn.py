"""
Basic nn building blocks (Linear, etc.) — CS336 assignment 3.4.
"""
from __future__ import annotations

import torch
import math
import einops
from torch import nn


class Linear(nn.Module):
    """
    Bias-free linear layer: column-vector form y = W @ x (see PDF 3.3.1 / 3.4.2).

    Parameter `weight` has shape (out_features, in_features) — store W, not W.T.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # Allocate (out_features, in_features); fill in forward step after you add init below.
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        self.init_parameters()

    def init_parameters(self) -> None:
        """Truncated normal init per assignment §3.4.1 (Linear). Match mean, std, and a/b to the PDF."""
        fan_in, fan_out = self.weight.shape[1], self.weight.shape[0]
        sigma = math.sqrt(2.0 / (fan_in + fan_out))
        a, b = -3 * sigma, 3 * sigma
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=sigma, a=a, b=b)
    
    def set_weights(self, weights: torch.Tensor) -> None:
        self.weight.data = weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features)
        returns: (..., out_features)

        Do not use torch.nn.functional.linear or nn.Linear.
        PyTorch row-layout batches use: out = x @ W.T  with W shape (out, in).
        """
        return einops.einsum(self.weight, x, "o i, ... i -> ... o")
        

class Embedding(nn.Module):
    """
    A learnable embedding lookup table, equivalent to torch.nn.Embedding

    Parameter `embedding_matrix` has shape (num_embeddings, embedding_dim)
    The forward function uses the slice operation to get the embeddings for the token ids
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embedding_matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs))
        self.init_weights()
    
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (b, s)
        returns: (b, s, embedding_dim)
        """
        return self.embedding_matrix[token_ids] 
    
    def init_weights(self):
        torch.nn.init.trunc_normal_(self.embedding_matrix, mean=0, std=1, a=-3, b=3)

    def set_weights(self, weights: torch.Tensor) -> None:
        self.embedding_matrix.data = weights

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        self.g = nn.Parameter(torch.ones(d_model, **factory_kwargs))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # RMSNorm
        rms_a = torch.sqrt(torch.mean(x ** 2, dim=-1) + self.eps)
        result = x / rms_a.unsqueeze(-1)
        result = result * self.g
        # return in the original dtype
        return result.to(in_dtype)
    
    def set_weights(self, weights: torch.Tensor) -> None:
        self.g.data = weights
    

class SwiGLU(nn.Module):
    """
    Position-wise feed-forward network using the SwiGLU nonlinearity.

    The transformation is:
        FFN(x) = W2( SiLU(W1 x) ⊙ (W3 x) )

    where SiLU(z) = z * sigmoid(z), and ⊙ is elementwise multiplication.

    Shapes:
        input:  (..., d_model)
        W1, W3: (d_ff, d_model)   implemented as Linear(d_model -> d_ff)
        W2:     (d_model, d_ff)   implemented as Linear(d_ff -> d_model)
        output: (..., d_model)
    """

    def __init__(
        self, d_model: int, d_ff: int | None = None, *, multiple_of: int = 64,
        device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        factory_kwargs = {"device": device, "dtype": dtype}
        self.w1 = Linear(self.d_model, self.d_ff, **factory_kwargs)
        self.w3 = Linear(self.d_model, self.d_ff, **factory_kwargs)
        self.w2 = Linear(self.d_ff, self.d_model, **factory_kwargs)
    
    def set_w1_weights(self, weights: torch.Tensor) -> None:
        self.w1.set_weights(weights)

    def set_w2_weights(self, weights: torch.Tensor) -> None:
        self.w2.set_weights(weights)

    def set_w3_weights(self, weights: torch.Tensor) -> None:
        self.w3.set_weights(weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.w1(x)
        # SiLU(W1x)
        silu_w1_x = w1_x * torch.sigmoid(w1_x)
        w3_x = self.w3(x)
        mult = silu_w1_x * w3_x
        res = self.w2(mult)
        return res

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self, theta: float, d_k: int, max_seq_len: int, device=None
    ):
        super().__init__()
        assert d_k % 2 == 0
        assert max_seq_len > 0
        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)

        pairs = torch.arange(0, d_k, 2)
        seq = torch.arange(0, max_seq_len) #(seq_len)
        inv_freq = 1.0 / (theta ** (pairs.float() / d_k)) #(d_k / 2)
        freqs = einops.einsum(seq, inv_freq, "a, b -> a b")  # (seq_len, d_k / 2)

        # Interleaved layout: pair (x_{2i}, x_{2i+1}) shares one angle → repeat each cos/sin twice.
        cos = freqs.cos().repeat_interleave(2, dim=-1) # [c0, c1, c2, …] → [c0, c0, c1, c1, c2, c2, …]
        sin = freqs.sin().repeat_interleave(2, dim=-1) # [s0, s1, s2, …] → [s0, s0, s1, s1, s2, s2, …]
        self.register_buffer("cos", cos, persistent=False)  # (seq_len, d_k)
        self.register_buffer("sin", sin, persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        # Interleaved pairs (not first-half / second-half as in some LLaMA code paths).
        return torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).reshape(*x.shape)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_pos = self.cos[token_positions] # (batch_size, seq_len, d_k)
        sin_pos = self.sin[token_positions] # (batch_size, seq_len, d_k)
        return (x * cos_pos) + (self.rotate_half(x) * sin_pos)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        x (torch.Tensor): The input tensor to softmax.
        dim (int): The dimension to apply softmax to.
    """
    # return torch.nn.functional.softmax(x, dim=dim)
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_norm = x - x_max
    return torch.exp(x_norm) / torch.sum(torch.exp(x_norm), dim=dim, keepdim=True)

def scaled_dot_product_attention(k, q, v, mask_mat=None):
    """
    k: (batch_size, ..., seq_len, d_k)
    q: (batch_size, ..., seq_len, d_k)
    v: (batch_size, ..., seq_len, d_v)
    mask_mat: optional, (seq_len, seq_len), True or False

    return: (batch_size, ..., d_v)
    """
    score = q @ k.transpose(-2, -1) # (batch_size, ..., seq_len, seq_len)
    d_k = k.shape[-1]
    score_norm = score / torch.sqrt(torch.tensor(d_k, device=score.device, dtype=score.dtype))
    if mask_mat is not None:
        # broadcast
        num_lead_dim = k.ndim - mask_mat.ndim
        while mask_mat.ndim < score.ndim:
            mask_mat = mask_mat.unsqueeze(0)
        # (1, ... seq_len, seq_len)
        mask = mask_mat.expand_as(score)  # (batch_size, ..., seq_len, seq_len)
        mask_val = torch.full(mask.shape, float('-inf'), device=mask.device, dtype=torch.float32)
        mask_val[mask] = 0.0
        score_norm = score_norm + mask_val
    attn_scores = softmax(score_norm, -1)  # (batch_size, ..., seq_len, seq_len)
    return attn_scores @ v # (batch_size, ..., seq_len, d_v)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
        enable_rope: bool, theta: float = 10000, max_seq_len: int = -1, 
        device: torch.device | None = None, dtype: torch.dtype | None = None,):
        
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads") 
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(self.d_model, self.d_model, device, dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device, dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device, dtype)

        self.o_proj = Linear(self.d_model, self.d_model, device, dtype)

        # RoPE applies per head: last dim is d_k = d_model // num_heads (not d_model).
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len) if enable_rope else None

    def set_q_proj_weights(self, weights: torch.Tensor) -> None:
        self.q_proj.set_weights(weights)

    def set_k_proj_weights(self, weights: torch.Tensor) -> None:
        self.k_proj.set_weights(weights)

    def set_v_proj_weights(self, weights: torch.Tensor) -> None:
        self.v_proj.set_weights(weights)

    def set_o_proj_weights(self, weights: torch.Tensor) -> None:
        self.o_proj.set_weights(weights)
    
    
    def forward(self, x, positions=None):
        """
        x: (batch_size, ..., seq_len, d_model)
        returns: (batch_size, ..., seq_len, d_model)
        """
        if self.rope and positions is None:
            raise ValueError("positions must be provided if RoPE is enabled")

        q = self.q_proj(x) # (batch_size, ..., seq_len, num_heads * d_k)
        k = self.k_proj(x) # (batch_size, ..., seq_len, num_heads * d_k)
        v = self.v_proj(x) # (batch_size, ..., seq_len, num_heads * d_k)

        ori_shape = q.shape
        new_shape = q.shape[:-1] + (self.num_heads, self.d_k)
        q = q.view(new_shape).transpose(-2, -3)  # (batch_size, ..., num_heads, seq_len, d_k)
        k = k.view(new_shape).transpose(-2, -3)  # (batch_size, ..., num_heads, seq_len, d_k)
        v = v.view(new_shape).transpose(-2, -3)  # (batch_size, ..., num_heads, seq_len, d_k)

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

        if self.rope:        
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        attn = scaled_dot_product_attention(k, q, v, mask)  # (batch_size, ..., num_heads, seq_len, d_k)
        # print("attn ", attn.shape)
        attn_final = attn.transpose(-2, -3).reshape(ori_shape)

        out = self.o_proj(attn_final) # (batch_size, ..., seq_len, d_model)
        return out


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block.

    Structure (pre-norm):
        y = x + Attn(RMSNorm(x))
        z = y + FFN(RMSNorm(y))
    
    This block uses causal multi-head self-attention with RoPE
    """

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int,
        max_seq_len: int, theta: float, eps: float = 1e-5,
        device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.rmsnorm_1 = RMSNorm(d_model, eps, **factory_kwargs)
        self.mha = MultiHeadAttention(d_model, num_heads, True, theta, max_seq_len, **factory_kwargs)
        self.rmsnorm_2 = RMSNorm(d_model, eps, **factory_kwargs)
        self.ffn = SwiGLU(d_model, d_ff, **factory_kwargs)
    
    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        self.rmsnorm_1.set_weights(weights["ln1.weight"])
        self.mha.set_q_proj_weights(weights["attn.q_proj.weight"])
        self.mha.set_k_proj_weights(weights["attn.k_proj.weight"])
        self.mha.set_v_proj_weights(weights["attn.v_proj.weight"])
        self.mha.set_o_proj_weights(weights["attn.output_proj.weight"])
        self.rmsnorm_2.set_weights(weights["ln2.weight"])
        self.ffn.set_w1_weights(weights["ffn.w1.weight"])
        self.ffn.set_w2_weights(weights["ffn.w2.weight"])
        self.ffn.set_w3_weights(weights["ffn.w3.weight"])

    
    def forward(self, x, positions):
        """
        x: (batch_size, seq_len, d_model)
        positins: (batch_size, seq_len) or broadcast to it
        return: (batch_size, seq_len, d_model)
        """
        y = x + self.mha(self.rmsnorm_1(x), positions)
        z = y + self.ffn(self.rmsnorm_2(y))
        return z



    
