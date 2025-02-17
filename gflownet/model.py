import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


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

class MathTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4, max_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.logZ = nn.Parameter(torch.ones(1))

        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        return self.fc_out(x)
    
class MathExpressionDataset(Dataset):
    def __init__(self, sequences, vocab, max_len):
        self.sequences = sequences
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Convert tokens to indices
        input_indices = [self.vocab[token] for token in seq[:-1]]
        target_index = self.vocab[seq[-1]]
        
        # Pad sequence if necessary
        if len(input_indices) < self.max_len - 1:
            padding_length = self.max_len - 1 - len(input_indices)
            input_indices = input_indices + [self.vocab['']] * padding_length
        else:
            # If sequence is too long, truncate it
            input_indices = input_indices[-(self.max_len - 1):]
            
        return {
            'input': torch.tensor(input_indices, dtype=torch.long),
            'target': torch.tensor(target_index, dtype=torch.long)
        }

def create_vocabulary():
    actions = ['', '+(', '+)', '-(', '-)', '*(', '*)', '/(', '/)'] + [str(i) for i in range(1, 14)]
    vocab = {token: idx for idx, token in enumerate(actions)}
    return vocab, actions

def train_model(model, train_loader, num_epochs, device, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)  # Shape: [batch_size, seq_len, vocab_size]
            # Only get the output for the last position
            last_output = outputs[:, -1, :]  # Shape: [batch_size, vocab_size]
            loss = criterion(last_output, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

def get_next_token_probabilities(model, sequence, vocab, max_len, device):
    """
    Get probability distribution over possible next tokens.
    
    Args:
        model: The trained transformer model
        sequence: List of tokens representing the current sequence
        vocab: Dictionary mapping tokens to indices
        max_len: Maximum sequence length
        device: Device to run the model on
    
    Returns:
        List of (token, probability) tuples, sorted by probability in descending order
    """
    # Convert sequence to indices
    input_indices = [vocab[token] for token in sequence]
    
    # Pad sequence if necessary
    if len(input_indices) < max_len - 1:
        input_indices = input_indices + [vocab['']] * (max_len - 1 - len(input_indices))
    else:
        # If sequence is too long, truncate it
        input_indices = input_indices[-(max_len - 1):]
    
    # Prepare input tensor
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output[0, -1], dim=0)
    
    # Convert to list of (token, probability) pairs
    idx_to_token = {idx: token for token, idx in vocab.items()}
    token_probs = [(idx_to_token[i], prob.item()) for i, prob in enumerate(probabilities)]
    
    # Sort by probability in descending order
    token_probs.sort(key=lambda x: x[1], reverse=True)
    
    return token_probs

def predict_next_token(model, sequence, vocab, max_len, device):
    """Get the most likely next token."""
    token_probs = get_next_token_probabilities(model, sequence, vocab, max_len, device)
    return token_probs[0][0]  # Return the token with highest probability

# Example usage
if __name__ == "__main__":
    # Configuration
    max_len = 50
    d_model = 128
    batch_size = 32
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create vocabulary
    vocab, actions = create_vocabulary()
    vocab_size = len(vocab)
    
    # Create model
    model = MathTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        max_len=max_len
    ).to(device)
    
    # Example training data (you would need to provide your own sequences)
    example_sequences = [
        ['1', '+(', '2', '+)', '3'],
        ['2', '*(', '3', '*)', '6'],
        # Add more training sequences here
    ]
    
    # Create dataset and dataloader
    dataset = MathExpressionDataset(example_sequences, vocab, max_len)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train model
    train_model(model, train_loader, num_epochs, device)
    
    # Example prediction
    test_sequence = ['1', '+(', '2']
    predicted_token = predict_next_token(model, test_sequence, vocab, max_len, device)
    print(f'Input sequence: {test_sequence}')
    print(f'Predicted next token: {predicted_token}')