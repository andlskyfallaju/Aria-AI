import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig
from .layers import get_norm, MLP
from .attention import CausalSelfAttention

class TransformerBlock(nn.Module):
    """
    Standard Decoder Transformer Block with Pre-Layer/RMS Normalization.
    x -> Norm -> MultiHeadAttention -> + (Residual) -> Norm -> MLP -> + (Residual)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = get_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = get_norm(config)
        self.mlp = MLP(config)

    def forward(self, x, return_attention: bool = False):
        if return_attention:
            attn_out, attn_weights = self.attn(self.ln_1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, attn_weights
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

class GPT(nn.Module):
    """
    Full Decoder-Only GPT Transformer Model built from scratch.
    Capable of character or BPE token generation and fast CPU inference.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = get_norm(config),
        ))
        
        # Final language modeling head (maps hidden embedding -> vocab logits)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying: share weights between input token embedding and output LM head
        # This saves memory (~30% parameter reduction) and improves convergence
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights cleanly
        self.apply(self._init_weights)

        # Apply special scaled initialization to residual projections (GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True):
        """Return parameter count in human-readable format."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx, targets=None, return_attention: bool = False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"

        # Position indices: [0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Look up token embeddings & position embeddings
        tok_emb = self.transformer.wte(idx) # shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # shape (t, n_embd)

        x = self.transformer.drop(tok_emb + pos_emb)

        all_attentions = []
        for block in self.transformer.h:
            if return_attention:
                x, attn = block(x, return_attention=True)
                all_attentions.append(attn)
            else:
                x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training time: compute loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference time: compute logits for the last token position only
            logits = self.lm_head(x[:, -1:, :]) # shape (b, 1, vocab_size)
            loss = None

        if return_attention:
            return logits, loss, all_attentions
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = None, top_p: float = None, stop_tokens: list = None):
        """
        Autoregressively generate new tokens from input prompt `idx`.
        Supports Temperature, Top-K, Top-P sampling, and Stop-Token detection.
        """
        for _ in range(max_new_tokens):
            # Crop index sequence to max block size context window
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward model to get logits for last position
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 0.01)

            # Top-K filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Top-P (Nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above top_p threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift mask right to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                for batch_idx in range(logits.size(0)):
                    indices_to_remove = sorted_indices[batch_idx, sorted_indices_to_remove[batch_idx]]
                    logits[batch_idx, indices_to_remove] = -float('Inf')

            # Softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample next token from probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to running context
            idx = torch.cat((idx, idx_next), dim=1)

            # Early stop if stop_token is generated
            if stop_tokens and idx_next.item() in stop_tokens:
                break

        return idx
