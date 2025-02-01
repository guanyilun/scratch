import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import MathTransformer
from data_utils import MathExpressionDataset, Tokenizer


def train_model(model, train_loader, num_epochs, device, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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


def get_next_token_probabilities(model, sequence, tokenizer, max_len, device):
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
    input_indices = tokenizer.encode(sequence)
    seq_len = len(input_indices)

    # Pad sequence if necessary
    if len(input_indices) < max_len - 1:
        input_indices = input_indices + tokenizer.encode(['']) * (max_len - 1 - seq_len)
    else:
        # If sequence is too long, truncate it
        input_indices = input_indices[-(max_len - 1):]

    # Prepare input tensor
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)

    padding_mask = torch.zeros((1, max_len - 1), dtype=torch.bool).to(device)
    padding_mask[0, seq_len:] = True  # Mark padding positions

    # Get model prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor, mask=padding_mask)
        probabilities = torch.softmax(output[0, seq_len-1], dim=0)  # YG: dim needs to be checked

    # Convert to list of (token, probability) pairs
    token_probs = [(a, p) for (a, p) in zip(tokenizer.actions, probabilities)]
    token_probs.sort(key=lambda x: x[1], reverse=True)
    return token_probs


def predict_next_token(model, sequence, vocab, max_len, device):
    """Get the most likely next token."""
    token_probs = get_next_token_probabilities(model, sequence, vocab, max_len, device)
    return token_probs[0][0]  # Return the token with highest probability


# Example usage
if __name__ == "__main__":
    max_len = 50
    d_model = 128
    batch_size = 32
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer()

    torch.manual_seed(42)

    # Create model
    model = MathTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        max_len=max_len
    ).to(device)

    # Example training data (you would need to provide your own sequences)
    example_sequences = [
        ['1', '+(', '2', '+)', '3'],
        ['2', '*(', '3', '*)', '6'],
    ]

    # Create dataset and dataloader
    dataset = MathExpressionDataset(example_sequences, tokenizer, max_len)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train model
    train_model(model, train_loader, num_epochs, device)

    # Example prediction
    test_sequence = ['1', '+(', '2']
    predicted_token = predict_next_token(model, test_sequence, tokenizer, max_len, device)
    print(f'Input sequence: {test_sequence}')
    print(f'Predicted next token: {predicted_token}')