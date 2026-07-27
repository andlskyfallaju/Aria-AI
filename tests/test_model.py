import unittest
import torch
import os
import sys

# Ensure parent directory is in import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.transformer import GPT
from model.layers import RMSNorm
from tokenizer.char_tokenizer import CharTokenizer

class TestLLMArchitecture(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig.nano(vocab_size=65)
        self.model = GPT(self.config)
        self.model.eval()

    def test_parameter_count(self):
        n_params = self.model.get_num_params()
        self.assertGreater(n_params, 0)
        print(f"Nano model parameter count: {n_params:,}")

    def test_forward_pass_shape(self):
        batch_size = 2
        seq_len = 16
        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        logits, loss = self.model(x)

        # In inference mode (targets=None), logits output shape is (B, 1, V)
        self.assertEqual(logits.shape, (batch_size, 1, self.config.vocab_size))
        self.assertIsNone(loss)

    def test_loss_computation(self):
        batch_size = 2
        seq_len = 16
        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        y = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        logits, loss = self.model(x, targets=y)

        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)

    def test_attention_weights_extraction(self):
        batch_size = 1
        seq_len = 8
        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        logits, loss, attentions = self.model(x, return_attention=True)

        self.assertEqual(len(attentions), self.config.n_layer)
        # Check attention matrix dimensions: (B, n_head, seq_len, seq_len)
        self.assertEqual(attentions[0].shape, (batch_size, self.config.n_head, seq_len, seq_len))

    def test_generation(self):
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        out = self.model.generate(prompt, max_new_tokens=5, temperature=1.0)
        self.assertEqual(out.shape, (1, 8))

    def test_rmsnorm(self):
        norm = RMSNorm(128)
        x = torch.randn(2, 8, 128)
        out = norm(x)
        self.assertEqual(out.shape, (2, 8, 128))

    def test_tokenizer_roundtrip(self):
        sample = "Hello LLM World!"
        tok = CharTokenizer(sample)
        ids = tok.encode(sample)
        decoded = tok.decode(ids)
        self.assertEqual(sample, decoded)

if __name__ == "__main__":
    unittest.main()
