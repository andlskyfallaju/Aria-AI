# Aria AI

A decoder-only transformer language model **built from scratch in PyTorch**, with its own BPE tokenizer, a two-stage training pipeline (pre-training, then instruction fine-tuning) and an interactive web studio for looking inside the model as it generates text.

I built Aria AI to understand how modern language models work by implementing every piece myself, rather than loading a pre-built model. It is designed to train on a laptop CPU or an integrated GPU, so the models are small, but the architecture follows the same design as larger LLaMA-style models.

---

## Highlights

- **Transformer written from first principles:** token and position embeddings, multi-head causal self-attention with a fused Q/K/V projection, pre-norm residual blocks, a GELU feed-forward network and weight tying between the input embedding and output head.
- **RMSNorm** (as in LLaMA and Mistral) instead of standard LayerNorm, with a switch back to LayerNorm in the config.
- **Custom byte-pair encoding (BPE) tokenizer** with reserved special tokens (`<|user|>`, `<|assistant|>`, `<|endofturn|>`) that are never split apart, plus a character-level fallback.
- **Two-stage training pipeline:** language pre-training, then supervised fine-tuning on instruction pairs where the loss is masked on the prompt, so the model only learns from the assistant's replies.
- **Several data sources:** a modern English corpus, Alpaca instruction data, FineWeb-Edu (a sample fetched on demand), TinyShakespeare, or your own text file.
- **Interactive Web Studio:** see tokenisation, next-token probabilities and attention heatmaps for every layer and head, and generate text from the browser.
- **Runs on modest hardware:** automatic device selection for CUDA, Apple MPS, DirectML (Intel and AMD GPUs on Windows) or CPU.

---

## Architecture

| Component | Implementation |
| --- | --- |
| Model type | Decoder-only transformer (GPT-style) |
| Normalisation | RMSNorm, applied before attention and before the feed-forward layer |
| Attention | Multi-head causal self-attention, fused QKV projection, causal mask |
| Feed-forward | Linear (4x wider), GELU, Linear |
| Positions | Learned position embeddings |
| Output head | Linear layer tied to the token embedding weights |
| Biases | Off by default (LLaMA-style) |
| Sampling | Temperature, top-k and top-p (nucleus) sampling |

### Model presets

| Preset | Layers | Heads | Embedding size | Context length | Approx. parameters* |
| --- | :---: | :---: | :---: | :---: | :---: |
| `nano` | 4 | 4 | 128 | 128 | ~1M |
| `micro` | 6 | 6 | 192 | 256 | ~3M |
| `mini` | 6 | 8 | 384 | 256 | ~11M |

\*With a 2,048-token vocabulary. The count changes with vocabulary size.

---

## Training

Training is done with `train.py`:

- **Optimiser:** AdamW (betas 0.9 / 0.95, weight decay 0.1)
- **Schedule:** linear warm-up followed by cosine learning-rate decay
- **Stability:** gradient clipping at 1.0
- **Evaluation:** train and validation loss reported at a regular interval, with checkpoints saved to `checkpoints/`

### Datasets

| Name | Use |
| --- | --- |
| `modern_english` | Stage 1 pre-training corpus (included) |
| `alpaca_chat` | Stage 2 instruction fine-tuning (downloaded on first use) |
| `fineweb_edu` | A sample of FineWeb-Edu web text, fetched through the Hugging Face dataset viewer |
| `tinyshakespeare` | Small character-level test corpus |
| `--custom_file` | Any plain text file or book |

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/andlskyfallaju/Aria-AI.git
cd Aria-AI
pip install torch
```

For Intel or AMD GPU acceleration on Windows, also install DirectML:

```bash
pip install torch-directml
```

### 2. Run the tests

```bash
python -m unittest tests/test_model.py
```

The tests check the forward pass shape, loss computation, attention weight extraction, text generation, RMSNorm and a tokenizer round trip.

### 3. Train

**Stage 1: pre-train on modern English**

```bash
python train.py --dataset modern_english --preset micro --tokenizer bpe --vocab_size 2048 --max_iters 3000 --force_dataset
```

**Stage 2: fine-tune into a chat assistant**

```bash
python train.py --dataset alpaca_chat --preset micro --tokenizer bpe --vocab_size 2048 --max_iters 1500 --resume_checkpoint checkpoints/model_latest.pt
```

Use the same preset and vocabulary size in both stages. If the DirectML backend crashes during training, add `--force_cpu`.

### 4. Generate text

```bash
python generate.py --prompt "The doctor " --tokens 200 --temperature 0.8 --top_k 10 --top_p 0.9
```

### 5. Launch the Web Studio

```bash
python server.py 8080
```

Then open [http://localhost:8080](http://localhost:8080). The studio shows:

- BPE token chips for your prompt
- Next-token probability bars
- Attention heatmaps across every layer and head
- Live text generation with adjustable sampling settings

For the fine-tuned assistant, use the chat format:

```text
<|user|>
What is artificial intelligence?
<|assistant|>
```

The server also exposes a small JSON API: `GET /api/info`, `POST /api/tokenize`, `POST /api/attention` and `POST /api/generate`.

---

## Project Structure

```
Aria-AI/
├── model/
│   ├── config.py            Model presets (nano, micro, mini)
│   ├── layers.py            RMSNorm and feed-forward layers
│   ├── attention.py         Multi-head causal self-attention
│   └── transformer.py       Full decoder-only model and text generation
├── tokenizer/
│   ├── bpe_tokenizer.py     BPE tokenizer with special tokens
│   ├── char_tokenizer.py    Character-level fallback
│   └── __init__.py          Tokenizer loader
├── dataset/
│   ├── prepare_data.py      Dataset download, preparation and tokenizer building
│   ├── modern_english.txt   Pre-training corpus
│   └── vocab.json           Saved vocabulary and special tokens
├── webui/                   Web Studio (HTML, CSS, JavaScript)
├── tests/test_model.py      Unit tests
├── train.py                 Training script
├── generate.py              Command-line text generation
└── server.py                HTTP server and API for the Web Studio
```

---

## Status and Limitations

Aria AI is a learning project, so it is worth being clear about what it is:

- The models are small (roughly 1M to 11M parameters) and trained on small datasets, so they will not match large commercial models. The point is understanding and experimentation.
- The unit tests cover model mechanics, not the quality of the generated text.
- Output quality depends heavily on training time and data, and the 2-stage pipeline needs matching presets and vocabulary sizes between stages.

---

## License

Released under the MIT License. See `LICENSE`.

---

## Author

**Andile Makuyana**
BSc ICT (Software Engineering) student, Arrupe Jesuit University
GitHub: [@andlskyfallaju](https://github.com/andlskyfallaju)
