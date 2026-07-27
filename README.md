# 🧠 Aria AI — Scratch Decoder-Only LLM Studio

A modern, high-performance **Decoder-Only Transformer Language Model** built entirely from first principles using PyTorch and custom CUDA / DirectML Intel GPU acceleration.

Includes a custom **Byte-Pair Encoding (BPE) Subword Tokenizer**, a **2-Stage Transfer Learning / SFT Pipeline**, and a live **Interactive Web Studio Visualizer** for inspecting multi-head self-attention heatmaps, token probabilities, and embeddings.

---

## 🌟 Key Features

- 🏗️ **First-Principles Transformer Architecture**: Built ground-up with custom `RMSNorm`, fused Q,K,V `CausalSelfAttention`, and pre-computed lower-triangular causal masks.
- 🔤 **Byte-Pair Encoding (BPE) Tokenizer**: Fast BPE subword training with token preservation, integer-to-text decoding, and support for special tags (`<|user|>`, `<|assistant|>`, `<|endofturn|>`).
- ⚡ **DirectML & Intel GPU Acceleration**: Automated device discovery supporting DirectX 12 accelerators (Intel Iris Xe GPUs), CUDA, Apple MPS, and multi-threaded CPU fallbacks.
- 🎯 **2-Stage Transfer Learning Pipeline**:
  - **Stage 1 (Pre-Training)**: Continuous language pre-training on modern English corpora (`modern_english`).
  - **Stage 2 (Supervised Fine-Tuning / SFT)**: Prompt-masked instruction fine-tuning on `alpaca_chat` pairs (`ignore_index = -100`).
- 🎨 **Interactive Web Studio UI**: Built-in HTTP server (`server.py`) rendering an interactive visualizer for next-token probability bars, attention heatmaps across layers & heads, and BPE token chips.

---

## 📁 Repository Layout

```text
Aria AI/
├── dataset/
│   ├── prepare_data.py        # Automated dataset fetcher & BPE tokenizer builder
│   ├── modern_english.txt     # Modern English prose corpus
│   └── vocab.json             # Saved BPE vocabulary & special tokens mapping
├── model/
│   ├── attention.py           # Multi-Head Causal Self-Attention
│   ├── config.py              # ModelConfig presets (nano, micro, mini)
│   ├── layers.py              # RMSNorm, GELU, and MLP FeedForward layers
│   └── transformer.py         # Decoder-Only GPT Transformer architecture
├── tokenizer/
│   ├── bpe_tokenizer.py       # Fast BPE Subword Tokenizer implementation
│   ├── char_tokenizer.py      # Character-level fallback tokenizer
│   └── __init__.py            # Dynamic tokenizer loader
├── webui/
│   ├── index.html             # Web Studio layout & controls
│   ├── styles.css             # Glassmorphism dark-mode UI styling
│   └── app.js                 # API client & attention heatmap visualizer
├── generate.py                # Command-line text generation script
├── server.py                  # HTTP REST API server for Web Studio
├── train.py                   # PyTorch GPU/CPU training script
└── tests/
    └── test_model.py          # Comprehensive unit test suite
```

---

## 🚀 Quick Start Guide

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/aria-ai.git
cd aria-ai
pip install torch torch-directml
```

### 2. Run Unit Tests

Verify that all model layers and attention mechanisms pass validation:

```bash
python -m unittest tests/test_model.py
```

### 3. Train a Model

#### Stage 1: Base Pre-Training
Pre-train a 3.44M parameter **micro** model on Modern English:

```bash
python train.py --dataset modern_english --preset micro --tokenizer bpe --vocab_size 2048 --max_iters 3000 --force_dataset
```

#### Stage 2: Instruction Fine-Tuning (SFT)
Fine-tune the pre-trained model into a Q&A chat assistant:

```bash
python train.py --dataset alpaca_chat --preset micro --tokenizer bpe --vocab_size 2048 --max_iters 1500 --resume_checkpoint checkpoints/model_latest.pt
```

### 4. Launch Interactive Web Studio

Start the web studio server:

```bash
python server.py 8080
```

Open **[http://localhost:8080](http://localhost:8080)** in your browser to interact with your model!

---

## 📜 Prompt Format

When prompting your instruction fine-tuned assistant in the Web Studio, use the structured chat tags:

```text
<|user|>
What is artificial intelligence?
<|assistant|>
```

---

## 📄 License

MIT License — feel free to modify, extend, and build upon this codebase!
