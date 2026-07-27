import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention block.
    Computes Q, K, V projections and applies scaled dot-product attention
    with a causal mask ensuring tokens can only look at past tokens.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout_p = config.dropout

        # Combine Query, Key, Value projections into one single linear layer for speed
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection back to n_embd
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask buffer: upper triangular matrix of -inf to mask future tokens
        # Register as buffer so it's moved to GPU/device automatically with the model
        bias_mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(
            1, 1, config.block_size, config.block_size
        )
        self.register_buffer("bias", bias_mask)

        # Saved attention weights for visualizer / analysis
        self.last_attention_weights = None

    def forward(self, x, return_attention: bool = False):
        B, T, C = x.size() # Batch size, Sequence length (T), Embedding dimension (C = n_embd)

        # Calculate Query, Key, Value for all heads in batch
        # qkv shape: (B, T, 3 * C) -> split into q, k, v each of shape (B, T, C)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape for multi-head attention: (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention calculation: (B, n_head, T, head_dim) x (B, n_head, head_dim, T) -> (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply causal mask: fill future token positions with -infinity before Softmax
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        if return_attention or not self.training:
            # Store last attention weights tensor (detach for safety)
            self.last_attention_weights = att.detach()

        att_dropped = self.attn_dropout(att)

        # Weighted aggregation of values: (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        y = att_dropped @ v

        # Concatenate all heads together: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection & residual dropout
        y = self.resid_dropout(self.c_proj(y))
        
        if return_attention:
            return y, att
        return y
