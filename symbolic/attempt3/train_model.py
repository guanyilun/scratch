import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import MathTransformer
from data_utils import MathExpressionDataset, Tokenizer


def train_model(model, train_loader, num_epochs, device, learning_rate=0.0001):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Add warmup and cosine decay
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1  # 10% warmup
    )
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')


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
        input_indices = input_indices + [tokenizer.vocab['<pad>']] * (max_len - 1 - seq_len)
    else:
        # If sequence is too long, truncate it
        input_indices = input_indices[-(max_len - 1):]

    # Prepare input tensor
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)

    # Get model prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        seq_len = min(seq_len, max_len - 1)
        probabilities = torch.softmax(output[0, seq_len-1], dim=0)  # YG: dim needs to be checked

    # Convert to list of (token, probability) pairs
    token_probs = [(a, p) for (a, p) in zip(tokenizer.actions, probabilities)]
    token_probs.sort(key=lambda x: x[1], reverse=True)
    return token_probs


def predict_next_token(model, sequence, vocab, max_len, device):
    """Get the most likely next token."""
    token_probs = get_next_token_probabilities(model, sequence, vocab, max_len, device)
    return token_probs[0][0]  # Return the token with highest probability


def rollout(model, sequence, tokenizer, max_len, device):
    """
    Generate a sequence of tokens by iteratively predicting the next token.

    Args:
        model: The trained transformer model
        sequence: List of tokens representing the initial sequence
        vocab: Dictionary mapping tokens to indices
        max_len: Maximum sequence length
        device: Device to run the model on
        num_tokens: Number of tokens to generate

    Returns:
        List of tokens representing the generated sequence

    """
    generated_sequence = sequence.copy()
    for _ in range(max_len - len(sequence)):
        next_token = predict_next_token(model, generated_sequence, tokenizer, max_len, device)
        if next_token == '<end>':
            break
        generated_sequence.append(next_token)
    return generated_sequence


# Example usage
if __name__ == "__main__":
    config = {
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6,
        'max_len': 100,
        'batch_size': 64,
        'learning_rate': 3e-4,
        'num_epochs': 50,
        'dropout': 0.1
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer()
    mode = 'train'

    # torch.manual_seed(42)

    if mode == 'train': 
        # Create model
        model = MathTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=config['d_model'],
            max_len=config['max_len'],
            num_layers=config['num_layers'],
        ).to(device)

        # Example training data (you would need to provide your own sequences)
        with open('data/math_expressions.txt', "r") as f:
            lines = f.readlines()
        sequences = [line.strip() for line in lines]

        # Create dataset and dataloader
        dataset = MathExpressionDataset(sequences, tokenizer, config['max_len'])
        train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

        # Train model
        train_model(model, train_loader, config['num_epochs'], device, learning_rate=1e-3)

        # Save model
        torch.save(model.state_dict(), 'data/math_transformer.pt')

    if mode == 'test': 
        model = MathTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=config['d_model'],
            max_len=config['max_len'],
            num_layers=config['num_layers']
        ).to(device)
        model.load_state_dict(torch.load('data/math_transformer.pt'))
    

        # Example prediction
        test_sequence = ['1', '+', '(', '2']
        predicted_token = predict_next_token(model, test_sequence, tokenizer, max_len, device)
        print(f'Input sequence: {test_sequence}')
        # roll out
        generated_sequence = rollout(model, test_sequence, tokenizer, max_len, device)
        print(f'Generated sequence: {generated_sequence}')