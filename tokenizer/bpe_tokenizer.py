import json
import re
import collections
from typing import List, Dict, Tuple, Optional

# Reserved role/control tokens. These are ALWAYS treated as a single atomic
# token ID — never split into sub-word pieces by the BPE merge process.
# This matters a lot for a small model: without this, "<|user|>" gets
# shredded into several ordinary sub-tokens, wasting capacity and making it
# much harder for the model to learn a clean, reliable turn-boundary signal.
SPECIAL_TOKENS = ["<|user|>", "<|assistant|>", "<|endofturn|>"]


class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) Tokenizer implemented from first principles.
    Used in GPT-2, GPT-4, LLaMA, etc.
    Merges frequent byte pairs into subwords and full words.

    Also reserves a small set of atomic special/control tokens (see
    SPECIAL_TOKENS above) so chat-formatted data can carry clean,
    single-token role/turn markers instead of raw text.
    """
    def __init__(self, target_vocab_size: int = 512):
        self.target_vocab_size = target_vocab_size
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}  # initial 256 bytes
        self.special_tokens: Dict[str, int] = {}  # e.g. {"<|user|>": 700}
        self._special_pattern: Optional[re.Pattern] = None

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def itos(self) -> Dict[int, str]:
        """Map integer token IDs to human-readable strings for visualizer/inspection."""
        res = {}
        for idx, b in self.vocab.items():
            s = b.decode("utf-8", errors="replace")
            res[idx] = s
        return res

    @property
    def stoi(self) -> Dict[str, int]:
        """Map human-readable strings back to integer token IDs."""
        return {s: i for i, s in self.itos.items()}

    def get_special_id(self, token: str) -> Optional[int]:
        """Return the reserved atomic token ID for a special token, if registered."""
        return self.special_tokens.get(token)

    def _add_special_tokens(self, tokens: List[str] = None):
        """Reserve atomic IDs for special/control tokens, appended after the trained vocab."""
        tokens = tokens if tokens is not None else SPECIAL_TOKENS
        next_id = max(self.vocab.keys(), default=-1) + 1
        for tok in tokens:
            if tok in self.special_tokens:
                continue
            self.vocab[next_id] = tok.encode("utf-8")
            self.special_tokens[tok] = next_id
            next_id += 1
        self._rebuild_special_pattern()

    def _rebuild_special_pattern(self):
        if not self.special_tokens:
            self._special_pattern = None
            return
        # Longest-first so overlapping markers don't get mis-split
        escaped = sorted((re.escape(t) for t in self.special_tokens), key=len, reverse=True)
        self._special_pattern = re.compile("(" + "|".join(escaped) + ")")

    def _get_stats(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, text: str):
        """Train BPE merges on raw text input using fast word-frequency aggregation."""
        # Initial 256 byte vocabulary
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self.special_tokens = {}

        # Strip out special-token markers before counting byte-pair stats so
        # they never pollute the learned merges — they get reserved as their
        # own atomic IDs afterwards instead.
        clean_text = text
        for tok in SPECIAL_TOKENS:
            clean_text = clean_text.replace(tok, " ")

        # Split by spaces to preserve word boundaries, converting words to byte-int tuples
        raw_words = clean_text.split(" ")
        words = []
        for idx, w in enumerate(raw_words):
            # Prepend space to words after the first word (like GPT-2)
            word_str = (" " + w) if idx > 0 else w
            if word_str:
                words.append(tuple(word_str.encode("utf-8")))

        word_counts = collections.Counter(words)

        # Reserve room for the special tokens within the target vocab budget
        num_merges = self.target_vocab_size - 256 - len(SPECIAL_TOKENS)
        for i in range(num_merges):
            pair_counts = collections.defaultdict(int)
            for word_tuple, count in word_counts.items():
                for pair in zip(word_tuple, word_tuple[1:]):
                    pair_counts[pair] += count
            if not pair_counts:
                break

            top_pair = max(pair_counts, key=pair_counts.get)
            idx = 256 + i
            self.merges[top_pair] = idx
            self.vocab[idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

            # Replace top_pair in word_counts
            new_word_counts = {}
            for word_tuple, count in word_counts.items():
                new_word = []
                j = 0
                while j < len(word_tuple):
                    if j < len(word_tuple) - 1 and word_tuple[j] == top_pair[0] and word_tuple[j + 1] == top_pair[1]:
                        new_word.append(idx)
                        j += 2
                    else:
                        new_word.append(word_tuple[j])
                        j += 1
                new_word_counts[tuple(new_word)] = count
            word_counts = new_word_counts

        # Reserve atomic IDs for <|user|>, <|assistant|>, <|endofturn|> at the end
        self._add_special_tokens()

    def _encode_chunk(self, text: str) -> List[int]:
        """Encode a chunk of plain text (no special tokens inside) using learned BPE merges."""
        if not text:
            return []
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            # Find the pair with the lowest merge rank (first merged pair)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = self._merge(ids, pair, idx)
        return ids

    def encode(self, text: str) -> List[int]:
        """Encode text using learned BPE merges, keeping special tokens atomic."""
        if not text:
            return []

        if not self._special_pattern:
            return self._encode_chunk(text)

        # Split on special tokens, keeping them as their own pieces
        pieces = self._special_pattern.split(text)
        ids: List[int] = []
        for piece in pieces:
            if not piece:
                continue
            if piece in self.special_tokens:
                ids.append(self.special_tokens[piece])
            else:
                ids.extend(self._encode_chunk(piece))
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode sequence of token IDs into clean UTF-8 string."""
        tokens = b"".join(self.vocab.get(idx, b"") for idx in ids)
        return tokens.decode("utf-8", errors="replace")

    def save(self, filepath: str):
        """Save BPE merges, vocabulary & special tokens to JSON file."""
        merges_str = {f"{p[0]},{p[1]}": idx for p, idx in self.merges.items()}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "tokenizer_type": "bpe",
                "target_vocab_size": self.target_vocab_size,
                "merges": merges_str,
                "special_tokens": self.special_tokens,
            }, f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(target_vocab_size=data["target_vocab_size"])
        for pair_str, idx in data["merges"].items():
            p1, p2 = map(int, pair_str.split(","))
            pair = (p1, p2)
            tok.merges[pair] = idx
            tok.vocab[idx] = tok.vocab[pair[0]] + tok.vocab[pair[1]]

        special_tokens = data.get("special_tokens", {})
        if special_tokens:
            for tok_str, idx in special_tokens.items():
                tok.vocab[idx] = tok_str.encode("utf-8")
                tok.special_tokens[tok_str] = idx
            tok._rebuild_special_pattern()
        else:
            # Backwards compatibility with older vocab.json files saved before
            # special tokens existed: reserve them now, appended at the end.
            tok._add_special_tokens()
        return tok
