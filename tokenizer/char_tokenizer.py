import json
from typing import List

class CharTokenizer:
    """
    Simple, transparent character-level tokenizer.
    Ideal for fast CPU training and learning how tokens map to integers.
    """
    def __init__(self, text: str = None):
        self.chars: List[str] = []
        self.stoi = {}
        self.itos = {}
        if text is not None:
            self.build_vocab(text)

    def build_vocab(self, text: str):
        """Extract unique characters and assign integer IDs."""
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> List[int]:
        """Convert string to list of token integer IDs."""
        return [self.stoi.get(c, 0) for c in text if c in self.stoi]

    def decode(self, tokens: List[int]) -> str:
        """Convert list of token integer IDs back to string."""
        return "".join([self.itos.get(t, "") for t in tokens])

    def save(self, filepath: str):
        """Save vocabulary to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str):
        """Load vocabulary from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.chars = data["chars"]
        tok.stoi = {ch: i for i, ch in enumerate(tok.chars)}
        tok.itos = {i: ch for i, ch in enumerate(tok.chars)}
        return tok
