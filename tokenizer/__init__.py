import json
from .char_tokenizer import CharTokenizer
from .bpe_tokenizer import BPETokenizer

def load_tokenizer(vocab_path: str):
    """Load CharTokenizer or BPETokenizer depending on vocab.json file schema."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("tokenizer_type") == "bpe":
        return BPETokenizer.load(vocab_path)
    return CharTokenizer.load(vocab_path)

__all__ = ["CharTokenizer", "BPETokenizer", "load_tokenizer"]
