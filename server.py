import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import torch
import torch.nn.functional as F

# Ensure root directory is in import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.config import ModelConfig
from model.transformer import GPT
from tokenizer import load_tokenizer, CharTokenizer

# Global model state
MODEL = None
TOKENIZER = None
CONFIG = None
from train import get_device
DEVICE = get_device()

def load_or_init_model():
    global MODEL, TOKENIZER, CONFIG
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "model_latest.pt")
    vocab_path = os.path.join(os.path.dirname(__file__), "checkpoints", "vocab.json")

    if os.path.exists(checkpoint_path) and os.path.exists(vocab_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        CONFIG = ckpt['config']
        TOKENIZER = load_tokenizer(vocab_path)
        MODEL = GPT(CONFIG).to(DEVICE)
        MODEL.load_state_dict(ckpt['model_state'])
        print(f"Loaded trained checkpoint: {MODEL.get_num_params():,} params, val loss: {ckpt.get('val_loss', 'N/A')}")
    else:
        # Fallback to untrained initialized Nano model for immediate visualizer demo
        sample_text = "First Citizen:\nBefore we proceed any further, hear me speak.\nAll:\nSpeak, speak.\n"
        TOKENIZER = CharTokenizer(sample_text)
        CONFIG = ModelConfig.nano(vocab_size=TOKENIZER.vocab_size)
        MODEL = GPT(CONFIG).to(DEVICE)
        print("Initialized nano model (untrained/demo state)")

    MODEL.eval()

class LLMVisualizerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        webui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui")
        super().__init__(*args, directory=webui_dir, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/info":
            self.handle_info()
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else "{}"
        try:
            data = json.loads(body)
        except Exception:
            data = {}

        if parsed.path == "/api/tokenize":
            self.handle_tokenize(data)
        elif parsed.path == "/api/attention":
            self.handle_attention(data)
        elif parsed.path == "/api/generate":
            self.handle_generate(data)
        else:
            self.send_error(404, "API Endpoint not found")

    def _send_json(self, response_dict):
        body = json.dumps(response_dict).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_info(self):
        response = {
            "num_params": MODEL.get_num_params(),
            "n_layer": CONFIG.n_layer,
            "n_head": CONFIG.n_head,
            "n_embd": CONFIG.n_embd,
            "block_size": CONFIG.block_size,
            "vocab_size": CONFIG.vocab_size,
            "device": str(DEVICE),
            "use_rms_norm": CONFIG.use_rms_norm
        }
        self._send_json(response)

    def handle_tokenize(self, data):
        text = data.get("text", "ROMEO:")
        token_ids = TOKENIZER.encode(text)
        tokens = [TOKENIZER.itos.get(idx, "?") for idx in token_ids]
        self._send_json({
            "text": text,
            "tokens": tokens,
            "token_ids": token_ids,
            "length": len(token_ids)
        })

    def handle_attention(self, data):
        prompt = data.get("prompt", "ROMEO:\nShall I hear more")
        token_ids = TOKENIZER.encode(prompt)
        if not token_ids:
            token_ids = [0]
        
        # Crop context to block size
        token_ids = token_ids[-CONFIG.block_size:]
        idx = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            logits, loss, attentions = MODEL(idx, return_attention=True)

        tokens = [TOKENIZER.itos.get(t, "?") for t in token_ids]

        # Process attentions into nested lists: [layer][head][q_pos][k_pos]
        # attentions is list of length n_layer, each tensor is (1, n_head, T, T)
        attn_data = []
        for layer_attn in attentions:
            layer_attn_cpu = layer_attn[0].cpu().tolist() # shape (n_head, T, T)
            # Round for low JSON payload
            layer_attn_rounded = [[[round(v, 4) for v in row] for row in head] for head in layer_attn_cpu]
            attn_data.append(layer_attn_rounded)

        self._send_json({
            "tokens": tokens,
            "token_ids": token_ids,
            "attentions": attn_data, # [layer][head][q][k]
            "n_layer": CONFIG.n_layer,
            "n_head": CONFIG.n_head
        })

    def handle_generate(self, data):
        prompt = data.get("prompt", "ROMEO:")
        max_tokens = min(data.get("max_tokens", 50), 200)
        temperature = float(data.get("temperature", 0.8))
        top_k = int(data.get("top_k", 10))

        token_ids = TOKENIZER.encode(prompt)
        if not token_ids:
            token_ids = [0]

        # Chat-tuned models get a reserved <|endofturn|> token — stop there
        # instead of continuing to hallucinate a brand new conversation turn.
        eot_id = TOKENIZER.get_special_id("<|endofturn|>") if hasattr(TOKENIZER, "get_special_id") else None

        idx = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
        
        step_probabilities = []
        with torch.no_grad():
            for _ in range(max_tokens):
                idx_cond = idx if idx.size(1) <= CONFIG.block_size else idx[:, -CONFIG.block_size:]
                logits, _ = MODEL(idx_cond)
                logits = logits[:, -1, :] / max(temperature, 0.01)

                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                
                # Get top 5 candidate tokens for visualizer step breakdown
                top_p_vals, top_p_indices = torch.topk(probs[0], k=min(5, CONFIG.vocab_size))
                candidates = []
                for p_val, p_idx in zip(top_p_vals.tolist(), top_p_indices.tolist()):
                    candidates.append({
                        "token": TOKENIZER.itos.get(p_idx, "?"),
                        "token_id": p_idx,
                        "probability": round(p_val, 4)
                    })

                idx_next = torch.multinomial(probs, num_samples=1)
                sampled_token = TOKENIZER.itos.get(idx_next.item(), "?")

                step_probabilities.append({
                    "step": len(step_probabilities) + 1,
                    "sampled_token": sampled_token,
                    "sampled_id": idx_next.item(),
                    "candidates": candidates
                })

                idx = torch.cat((idx, idx_next), dim=1)

                if eot_id is not None and idx_next.item() == eot_id:
                    break

        full_output_text = TOKENIZER.decode(idx[0].tolist())

        self._send_json({
            "prompt": prompt,
            "generated_text": full_output_text,
            "new_tokens_text": full_output_text[len(prompt):],
            "steps": step_probabilities
        })

def run_server(port=8000):
    load_or_init_model()
    server_address = ('', port)
    httpd = HTTPServer(server_address, LLMVisualizerHandler)
    print(f"\n[Scratch LLM Studio] Interactive Server running at: http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
