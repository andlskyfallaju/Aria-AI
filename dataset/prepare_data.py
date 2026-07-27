import os
import urllib.request
import torch
import sys

# Add parent directory to python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.char_tokenizer import CharTokenizer
from tokenizer.bpe_tokenizer import BPETokenizer

def prepare_dataset(dataset_name: str = "modern_english", data_dir: str = None, force: bool = False, custom_file: str = None, tokenizer_type: str = "bpe", target_vocab_size: int = 512):
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))
    
    os.makedirs(data_dir, exist_ok=True)
    
    if custom_file and os.path.exists(custom_file):
        print(f"Preparing dataset from custom file: '{custom_file}'...")
        input_file_path = custom_file
    elif dataset_name == "alpaca_chat":
        input_file_path = os.path.join(data_dir, "alpaca_chat.txt")
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    elif dataset_name == "modern_english":
        input_file_path = os.path.join(data_dir, "modern_english.txt")
        url = "https://www.gutenberg.org/files/1661/1661-0.txt"
        fallback_text = """To Sherlock Holmes she is always THE woman...""" * 100
    elif dataset_name == "fineweb_edu":
        # Real HuggingFaceFW/fineweb-edu data (filtered, high-quality web text),
        # pulled live via HF's public dataset-viewer /rows API. This is a
        # genuinely different, more diverse corpus than modern_english —
        # NOT an alias to the Sherlock Holmes text.
        input_file_path = os.path.join(data_dir, "fineweb_edu.txt")
        url = None
        fallback_text = """Photosynthesis is the process by which plants convert sunlight into chemical energy...""" * 100
    else:
        input_file_path = os.path.join(data_dir, "tinyshakespeare.txt")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        fallback_text = """First Citizen:\nBefore we proceed any further...""" * 100

    # Download dataset if not present or forced (unless custom file is provided)
    if not custom_file and (force or not os.path.exists(input_file_path)):
        print(f"Preparing dataset '{dataset_name}'...")
        try:
            if dataset_name == "fineweb_edu":
                content = _download_fineweb_edu_sample(target_chars=8_000_000)
                if not content:
                    raise RuntimeError("No rows returned from fineweb-edu API")
            else:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    raw_bytes = response.read()

                if dataset_name == "alpaca_chat":
                    import json
                    data_json = json.loads(raw_bytes.decode('utf-8'))
                    formatted_dialogues = []
                    for item in data_json:  # Use the FULL ~52,000 instruction-response pairs
                        inst = item.get("instruction", "").strip()
                        inp = item.get("input", "").strip()
                        out = item.get("output", "").strip()
                        if inp:
                            prompt = f"{inst}\nInput: {inp}"
                        else:
                            prompt = inst
                        # <|endofturn|> gives the model a learnable "stop here" signal
                        # so generation doesn't ramble into a fabricated new turn.
                        formatted_dialogues.append(f"<|user|>\n{prompt}\n<|assistant|>\n{out}\n<|endofturn|>\n")
                    content = "\n".join(formatted_dialogues)
                else:
                    content = raw_bytes.decode('utf-8-sig', errors='replace')

            with open(input_file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Successfully prepared '{dataset_name}' ({len(content):,} characters)!")
        except Exception as e:
            print(f"Network notice ({e}). Using built-in fallback corpus...")
            if dataset_name == "alpaca_chat":
                content = """<|user|>\nWho is Sherlock Holmes?\n<|assistant|>\nSherlock Holmes is a fictional private detective created by Sir Arthur Conan Doyle.\n<|endofturn|>\n""" * 200
            else:
                content = fallback_text
            with open(input_file_path, "w", encoding="utf-8") as f:
                f.write(content)

    # Read dataset
    with open(input_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Dataset length: {len(text):,} characters")

    # Build and save tokenizer (Reuse existing vocab.json if present to preserve token ID alignment)
    vocab_path = os.path.join(data_dir, "vocab.json")
    if os.path.exists(vocab_path) and not force:
        print(f"Loading existing Vocabulary from '{vocab_path}'...")
        from tokenizer import load_tokenizer
        tokenizer = load_tokenizer(vocab_path)
    else:
        if tokenizer_type == "bpe":
            print(f"Training BPE Subword Tokenizer (target vocab size: {target_vocab_size})...")
            tokenizer = BPETokenizer(target_vocab_size=target_vocab_size)
            tokenizer.train(text)
            tokenizer.save(vocab_path)
            print(f"BPE Vocabulary built! Total subwords/tokens: {tokenizer.vocab_size}")
        else:
            print("Building Character Tokenizer...")
            tokenizer = CharTokenizer(text)
            tokenizer.save(vocab_path)
            print(f"Character Vocabulary built! Unique characters: {tokenizer.vocab_size}")

    # Encode dataset. For chat-formatted data we also build a parallel LOSS
    # MASK (1 = train on this token, 0 = ignore) so the model only pays a
    # loss penalty for predicting the ASSISTANT's response — not for
    # "predicting" the user's question, which just wastes a small model's
    # limited capacity.
    if dataset_name == "alpaca_chat" and isinstance(tokenizer, BPETokenizer) and tokenizer.get_special_id("<|assistant|>") is not None:
        ids, mask = _build_chat_ids_and_mask(text, tokenizer)
        data = torch.tensor(ids, dtype=torch.long)
        mask_tensor = torch.tensor(mask, dtype=torch.long)
    else:
        ids = tokenizer.encode(text)
        data = torch.tensor(ids, dtype=torch.long)
        mask_tensor = torch.ones_like(data)  # plain LM: train on every token

    # 90% train, 10% val split (mask stays aligned with the token stream)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    train_mask, val_mask = mask_tensor[:n], mask_tensor[n:]

    train_path = os.path.join(data_dir, "train.pt")
    val_path = os.path.join(data_dir, "val.pt")
    train_mask_path = os.path.join(data_dir, "train_mask.pt")
    val_mask_path = os.path.join(data_dir, "val_mask.pt")

    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
    torch.save(train_mask, train_mask_path)
    torch.save(val_mask, val_mask_path)

    print(f"Saved {len(train_data):,} training tokens to {train_path}")
    print(f"Saved {len(val_data):,} validation tokens to {val_path}")
    if dataset_name == "alpaca_chat":
        pct_trained_on = 100.0 * mask_tensor.float().mean().item()
        print(f"Loss mask: training on {pct_trained_on:.1f}% of tokens (assistant responses only)")

    return tokenizer.vocab_size


def _download_fineweb_edu_sample(target_chars: int = 8_000_000, page_size: int = 100, max_pages: int = 400):
    """
    Pull real documents from HuggingFaceFW/fineweb-edu using HF's public
    dataset-viewer '/rows' API (no `datasets` library / parquet dependency
    needed — just plain HTTP+JSON, consistent with the rest of this file).

    We page through `sample-10BT` (a random subset of the full ~1T-token
    dataset) until we've collected roughly `target_chars` characters of
    real, filtered educational web text, then stop. This is intentionally
    a small slice of a huge dataset — enough to be a genuinely different,
    more diverse pretraining corpus than the single Sherlock Holmes novel,
    without trying to download gigabytes of parquet files onto a laptop.
    """
    import json
    import time

    base_url = "https://datasets-server.huggingface.co/rows"
    dataset = "HuggingFaceFW/fineweb-edu"
    config = "sample-10BT"
    split = "train"

    chunks = []
    total_chars = 0

    for page in range(max_pages):
        offset = page * page_size
        query = (
            f"{base_url}?dataset={dataset}&config={config}&split={split}"
            f"&offset={offset}&length={page_size}"
        )
        req = urllib.request.Request(query, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as e:
            print(f"  fineweb-edu page {page} failed ({e}), stopping pagination here.")
            break

        rows = payload.get("rows", [])
        if not rows:
            break

        for entry in rows:
            row = entry.get("row", {})
            text = (row.get("text") or "").strip()
            if text:
                chunks.append(text)
                total_chars += len(text)

        print(f"  fineweb-edu: fetched {offset + len(rows):,} rows, {total_chars:,} characters so far...")

        if total_chars >= target_chars:
            break

        time.sleep(0.15)  # be polite to the shared public API

    return "\n\n".join(chunks)


    """
    Re-parse the '<|user|>...<|assistant|>...<|endofturn|>' formatted chat
    text example-by-example so we know exactly where each assistant response
    starts. Returns a flat (ids, mask) pair, still safe to slice/batch like
    a normal token stream, but with mask==0 over the prompt/user portion.
    """
    ids: list = []
    mask: list = []
    examples = text.split("<|user|>")
    for chunk in examples:
        if not chunk.strip():
            continue
        if "<|assistant|>" not in chunk:
            continue
        prompt_part, response_part = chunk.split("<|assistant|>", 1)

        prefix_ids = tokenizer.encode("<|user|>" + prompt_part + "<|assistant|>")
        response_ids = tokenizer.encode(response_part)

        ids.extend(prefix_ids)
        mask.extend([0] * len(prefix_ids))
        ids.extend(response_ids)
        mask.extend([1] * len(response_ids))

    return ids, mask

# Backwards compatibility alias
def prepare_tinyshakespeare(data_dir: str = None):
    return prepare_dataset("tinyshakespeare", data_dir, tokenizer_type="char")

if __name__ == "__main__":
    prepare_dataset("modern_english", tokenizer_type="bpe", target_vocab_size=512)


