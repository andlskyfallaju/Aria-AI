import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Used in modern LLMs like LLaMA and Mistral.
    It simplifies standard LayerNorm by assuming zero-mean inputs,
    saving ~10-15% computation per layer on CPUs.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def get_norm(config: ModelConfig):
    """Factory helper to choose RMSNorm or standard LayerNorm."""
    if config.use_rms_norm:
        return RMSNorm(config.n_embd, eps=config.norm_eps)
    else:
        return nn.LayerNorm(config.n_embd, eps=config.norm_eps, bias=config.bias)

class MLP(nn.Module):
    """
    Feed-Forward Network (Position-wise MLP).
    Projects input dimension up by 4x, applies non-linearity (GELU),
    and projects back to embedding dimension d_model.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
