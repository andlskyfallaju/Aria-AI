from dataclasses import dataclass

@dataclass
class ModelConfig:
    """
    Hyperparameters configuration for the Decoder-Only Transformer (LLM).
    Optimized for low-resource CPUs & integrated graphics while remaining
    fully scalable to larger GPT / LLaMA scale models.
    """
    block_size: int = 128     # Maximum context length (number of tokens in input window)
    vocab_size: int = 65      # Size of the vocabulary (65 characters for char-level TinyShakespeare)
    n_layer: int = 4          # Number of Transformer block layers
    n_head: int = 4           # Number of Attention heads
    n_embd: int = 128         # Embedding dimension (hidden state size d_model)
    dropout: float = 0.0      # Dropout rate for regularization (0.0 for small CPU models to speed up)
    bias: bool = False        # True: bias in Linears and LayerNorms (like GPT-2). False: no bias (like LLaMA / PaLM)
    norm_eps: float = 1e-5    # Epsilon value for normalization layers
    use_rms_norm: bool = True # Use RMSNorm (faster on CPU) instead of standard LayerNorm

    @classmethod
    def nano(cls, vocab_size: int = 65):
        """Ultra-fast ~0.8M parameter model. Trains on CPU in < 1 minute."""
        return cls(
            block_size=128,
            vocab_size=vocab_size,
            n_layer=4,
            n_head=4,
            n_embd=128,
            dropout=0.0,
            use_rms_norm=True
        )

    @classmethod
    def micro(cls, vocab_size: int = 65):
        """Balanced ~2.5M parameter model. High quality for laptop training."""
        return cls(
            block_size=256,
            vocab_size=vocab_size,
            n_layer=6,
            n_head=6,
            n_embd=192,
            dropout=0.1,
            use_rms_norm=True
        )

    @classmethod
    def mini(cls, vocab_size: int = 65):
        """More expressive ~10M parameter model."""
        return cls(
            block_size=256,
            vocab_size=vocab_size,
            n_layer=6,
            n_head=8,
            n_embd=384,
            dropout=0.1,
            use_rms_norm=True
        )
