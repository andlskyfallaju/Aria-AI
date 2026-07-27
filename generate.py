import os
import sys
import argparse
import torch

from model.config import ModelConfig
from model.transformer import GPT
from tokenizer import load_tokenizer

def main():
    parser = argparse.ArgumentParser(description="Generate text using trained scratch LLM")
    parser.add_argument("--prompt", type=str, default="The doctor ", help="Input prompt string")
    parser.add_argument("--tokens", type=int, default=200, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0.1 = deterministic, 1.2 = creative)")
    parser.add_argument("--top_k", type=int, default=10, help="Top-K token sampling filter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P / Nucleus sampling threshold")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_latest.pt", help="Path to model checkpoint")
    args = parser.parse_args()

    from train import get_device
    device = get_device()

    checkpoint_dir = os.path.dirname(args.checkpoint)
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found.")
        print("Please train the model first by running: python train.py")
        sys.exit(1)

    # Load checkpoint & tokenizer
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt['config']
    model = GPT(config).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    tokenizer = load_tokenizer(vocab_path)

    print(f"Loaded LLM model ({model.get_num_params():,} parameters) from {args.checkpoint}")
    print(f"Prompt: {repr(args.prompt)}")
    print(f"Sampling config: Temperature={args.temperature}, Top-K={args.top_k}, Top-P={args.top_p}\n")
    print("-" * 50)

    encoded_prompt = tokenizer.encode(args.prompt)
    context = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)

    # If this tokenizer has a reserved <|endofturn|> token (chat-tuned models),
    # stop generation there instead of rambling into a fabricated new turn.
    stop_tokens = None
    if hasattr(tokenizer, "get_special_id"):
        eot_id = tokenizer.get_special_id("<|endofturn|>")
        if eot_id is not None:
            stop_tokens = [eot_id]

    out = model.generate(context, max_new_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, stop_tokens=stop_tokens)
    text = tokenizer.decode(out[0].tolist())
    print(text)
    print("-" * 50)

if __name__ == "__main__":
    main()
