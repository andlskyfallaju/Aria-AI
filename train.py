import os
import time
import math
import argparse
import torch
import torch.nn as nn
from model.config import ModelConfig
from model.transformer import GPT
from tokenizer import load_tokenizer
from dataset.prepare_data import prepare_dataset

def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str, mask: torch.Tensor = None):
    """
    Sample a random batch of input contexts (x) and target labels (y).
    y is x shifted by 1 position (autoregressive target token).

    If `mask` is provided (aligned 1:1 with `data`, 1=train / 0=ignore),
    target positions where the NEXT token is masked out get replaced with
    -1, which model.forward()'s cross_entropy(..., ignore_index=-1) skips.
    This is how we make fine-tuning only learn from assistant responses
    instead of also being penalized for "predicting" the user's prompt.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    if mask is not None:
        y_mask = torch.stack([mask[i+1:i+1+block_size] for i in ix])
        y = y.masked_fill(y_mask == 0, -1)
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device, train_mask=None, val_mask=None):
    """Compute average loss over multiple validation iterations."""
    out = {}
    model.eval()
    for split, data, mask in [('train', train_data, train_mask), ('val', val_data, val_mask)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, block_size, batch_size, device, mask=mask)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(it: int, warmup_iters: int, lr_decay_iters: int, learning_rate: float, min_lr: float):
    """Cosine learning rate scheduler with warmup."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except Exception:
        pass
    return torch.device("cpu")

def train():
    parser = argparse.ArgumentParser(description="Train a Decoder-Only LLM from scratch on CPU/GPU")
    parser.add_argument("--preset", type=str, default="nano", choices=["nano", "micro", "mini"], help="Model preset size")
    parser.add_argument("--dataset", type=str, default="modern_english", choices=["modern_english", "alpaca_chat", "fineweb_edu", "tinyshakespeare"], help="Dataset to train on")
    parser.add_argument("--custom_file", type=str, default=None, help="Path to a custom plain text file or book to train on")
    parser.add_argument("--tokenizer", type=str, default="bpe", choices=["bpe", "char"], help="Tokenizer type: 'bpe' (Subword) or 'char' (Character)")
    parser.add_argument("--vocab_size", type=int, default=512, help="Target vocabulary size for BPE tokenizer")
    parser.add_argument("--force_dataset", action="store_true", help="Force re-preparing and downloading the dataset")
    parser.add_argument("--max_iters", type=int, default=1200, help="Total training iterations")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Max learning rate")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluate loss every N steps")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint file to resume/fine-tune from")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU training even if a GPU/DirectML device is available (use if the DirectML backend crashes on backward())")
    args = parser.parse_args()

    device = torch.device("cpu") if args.force_cpu else get_device()
    if device.type == "cpu":
        # Enable Intel MKL-DNN acceleration and all 12 CPU threads
        num_threads = os.cpu_count() or 4
        torch.set_num_threads(num_threads)
        if hasattr(torch.backends, "mkldnn"):
            torch.backends.mkldnn.enabled = True
        print(f"--- Training LLM on device: CPU (Optimized with Intel MKL-DNN & {num_threads} Threads) ---")
    else:
        print(f"--- Training LLM on device: {str(device).upper()} ---")

    # Data preparation
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    vocab_size = prepare_dataset(
        args.dataset,
        data_dir,
        force=args.force_dataset,
        custom_file=args.custom_file,
        tokenizer_type=args.tokenizer,
        target_vocab_size=args.vocab_size
    )

    train_path = os.path.join(data_dir, "train.pt")
    val_path = os.path.join(data_dir, "val.pt")
    train_mask_path = os.path.join(data_dir, "train_mask.pt")
    val_mask_path = os.path.join(data_dir, "val_mask.pt")
    vocab_path = os.path.join(data_dir, "vocab.json")

    train_data = torch.load(train_path)
    val_data = torch.load(val_path)

    # Loss mask (assistant-only training signal for chat data). Falls back
    # to None (train on every token) for plain-text datasets or older
    # pre-processed data that doesn't have mask files yet.
    train_mask = torch.load(train_mask_path) if os.path.exists(train_mask_path) else None
    val_mask = torch.load(val_mask_path) if os.path.exists(val_mask_path) else None
    if train_mask is not None:
        pct = 100.0 * train_mask.float().mean().item()
        print(f"Loss mask loaded: training on {pct:.1f}% of tokens (assistant responses only)")

    tokenizer = load_tokenizer(vocab_path)

    # Initialize model config
    if args.preset == "nano":
        config = ModelConfig.nano(vocab_size=vocab_size)
    elif args.preset == "micro":
        config = ModelConfig.micro(vocab_size=vocab_size)
    else:
        config = ModelConfig.mini(vocab_size=vocab_size)

    model = GPT(config).to(device)

    # Resume weights if specified (Fine-tuning mode)
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state'], strict=False)
        print(f"Resumed model weights from '{args.resume_checkpoint}' for fine-tuning!")

    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Context window: {config.block_size} tokens | Vocab size: {config.vocab_size}")

    # AdamW Optimizer (foreach=False for clean DirectML GPU execution)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1, foreach=False)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    start_time = time.time()

    print("\nBeginning training loop...")
    for it in range(args.max_iters):
        # Determine learning rate
        lr = get_lr(it, warmup_iters=50, lr_decay_iters=args.max_iters, learning_rate=args.learning_rate, min_lr=args.learning_rate/10)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluate loss periodically
        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, config.block_size, args.batch_size, eval_iters=20, device=device, train_mask=train_mask, val_mask=val_mask)
            elapsed = time.time() - start_time
            print(f"Step {it:4d}/{args.max_iters} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | lr: {lr:.2e} | Elapsed: {elapsed:.1f}s", flush=True)
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                ckpt_path = os.path.join(args.checkpoint_dir, "model_latest.pt")
                torch.save({
                    'model_state': model.state_dict(),
                    'config': config,
                    'iter': it,
                    'val_loss': losses['val']
                }, ckpt_path)
                tokenizer.save(os.path.join(args.checkpoint_dir, "vocab.json"))

        # Single training step
        X, Y = get_batch(train_data, config.block_size, args.batch_size, device, mask=train_mask)
        logits, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    print("\nTraining completed!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    
    # Generate sample text after training
    print("\n--- Generating Sample Text from Trained LLM ---")
    model.eval()
    stop_tokens = None
    if args.dataset == "alpaca_chat":
        sample_prompt = "<|user|>\nWhat is artificial intelligence?\n<|assistant|>\n"
        eot_id = tokenizer.get_special_id("<|endofturn|>") if hasattr(tokenizer, "get_special_id") else None
        if eot_id is not None:
            stop_tokens = [eot_id]
    else:
        sample_prompt = "The doctor " if args.dataset == "modern_english" else "ROMEO:"
    context = torch.tensor(tokenizer.encode(sample_prompt), dtype=torch.long, device=device).unsqueeze(0)
    generated_indices = model.generate(context, max_new_tokens=150, temperature=0.8, top_k=10, stop_tokens=stop_tokens)
    print(tokenizer.decode(generated_indices[0].tolist()))

if __name__ == "__main__":
    train()
