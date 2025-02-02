import numpy as np
import torch
import torch.nn as nn
from torch.nn import Transformer
from data_utils import FormulaTokenizer


def generate_causal_mask(seq_len):
    """
    Generate a causal mask of shape (seq_len, seq_len).
    """
    return torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)


class Generator(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=8, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        # embedding + positional encoding
        x = self.embedding(x)  # Shape: (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        
        # Prepare the padding mask (if provided)
        src_key_padding_mask = None
        if mask is not None:
            # Convert mask to (batch_size, seq_len) boolean tensor where True indicates padding
            src_key_padding_mask = (mask == 0).transpose(0, 1)
        
        # Generate causal mask
        seq_len = x.size(0)  # Get sequence length
        causal_mask = generate_causal_mask(seq_len).to(x.device)  # Shape: (seq_len, seq_len)
        
        # Transformer processing with causal mask
        output = self.encoder(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        return self.fc_out(output)


# Needed positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), None, :]


class ValueModel(nn.Module):
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoderLayer(d_model, nhead=8)
        self.fc_out = nn.Linear(d_model, 1)
        
    def forward(self, sequence):
        x = self.embedding(sequence)
        x = self.transformer(x)
        return self.fc_out(x.mean(dim=1))

def test_generator_with_tokenizer():
    tokenizer = FormulaTokenizer(max_constants=3)
    generator = Generator(vocab_size=len(tokenizer.vocab))

    # Test case 1: Basic formula
    formula = "C0*x + C1"
    tokens = tokenizer.tokenize(formula)
    for _ in range(100):
        src = torch.tensor(tokens).unsqueeze(1)
        next_id = np.argmax(generator(src)[-1, 0, :].detach().numpy())
        if next_id == tokenizer.token_to_id['<END>']:
            print("End of sequence reached")
            break
        tokens += [next_id]
    print(f"Test 1 generated formula: {tokenizer.decode(tokens)}")

    # Test case 2: Batch processing with padding
    # padded_batch = [[10, 6, 3, 4, 11], [3, 4, 3, 0, 0]]
    # batch_tensor = torch.tensor(padded_batch).unsqueeze(0).transpose(0, 1)
    # output = generator(batch_tensor)
    # print(f"Test 2 output shape: {output.shape}")  # Should be [5, 2, 13]

    # # Test case 3: Check probabilities
    # logits = output[0, 0]
    # probabilities = torch.softmax(logits, dim=-1)
    # print(f"Test 3 sum of probabilities: {probabilities.sum().item():.4f}")

if __name__ == "__main__":
    test_generator_with_tokenizer()