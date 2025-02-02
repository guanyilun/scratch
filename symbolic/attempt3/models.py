import torch
import torch.nn as nn
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)  # Changed to include batch dimension
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, :x.size(1)]  # Broadcasting across batch dimension


def generate_square_subsequent_mask(size: int):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_padding_mask(seq, pad_idx=0):
    """Create a padding mask for a batch of sequences.
    
    Args:
        seq: Input tensor of shape (batch_size, seq_len)
        pad_idx: The index used for padding (default: 0)
    
    Returns:
        Padding mask of shape (batch_size, seq_len) where True values are positions
        that should be masked (padded positions)
    """
    padding_mask = seq == pad_idx
    return padding_mask


class MathTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, max_len=100, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Add layer normalization before transformer
        self.norm = nn.LayerNorm(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Add layer normalization before final projection
        self.final_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        padding_mask = create_padding_mask(x)
        causal_mask = generate_square_subsequent_mask(x.size(1)).to(x.device)
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.norm(x)
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = self.final_norm(x)
        return self.fc_out(x)
    

