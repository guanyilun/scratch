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
    """Get probability distribution over possible next tokens."""
    input_indices = tokenizer.encode(sequence, add_start_end=True)  # Make sure to add start token
    seq_len = len(input_indices)
    
    # Handle sequence length
    if seq_len >= max_len:
        input_indices = input_indices[-(max_len-1):]  # Keep room for next prediction
    else:
        input_indices = input_indices + [tokenizer.vocab['<pad>']] * (max_len - 1 - seq_len)
    
    # Prepare input tensor
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        # Get probabilities for the next token
        logits = output[0, -1]  # Take the last position
        probabilities = torch.softmax(logits, dim=0)
    
    # Filter out special tokens if needed
    token_probs = [(token, prob.item()) 
                   for token, prob in zip(tokenizer.actions, probabilities)
                   if token not in ['<pad>', '<start>']]  # Filter special tokens
    
    return sorted(token_probs, key=lambda x: x[1], reverse=True)


def predict_next_token(model, sequence, tokenizer, max_len, device, temperature=0.3):
    """Get the next token using temperature sampling."""
    token_probs = get_next_token_probabilities(model, sequence, tokenizer, max_len, device)
    
    # Apply temperature
    probs = torch.tensor([prob for _, prob in token_probs])
    probs = torch.softmax(torch.log(probs) / temperature, dim=0)
    
    # Sample from the distribution
    next_token_idx = torch.multinomial(probs, 1).item()
    return token_probs[next_token_idx][0]


def rollout(model, sequence, tokenizer, max_len, device, temperature=1.0, top_k=None):
    """Generate a sequence with more controlled generation."""
    generated_sequence = sequence.copy()
    
    for _ in range(max_len - len(sequence)):
        next_token = predict_next_token(
            model, 
            generated_sequence,
            tokenizer, 
            max_len, 
            device,
            temperature=temperature
        )
        
        # Stop conditions
        if next_token == '<end>' or next_token == '<pad>':
            break
            
        # Validate mathematical expression
        generated_sequence.append(next_token)
        
        # Optional: Add basic validation
        if len(generated_sequence) > 2:
            # Check for consecutive operators
            if (generated_sequence[-1] in tokenizer.operators and 
                generated_sequence[-2] in tokenizer.operators):
                generated_sequence.pop()  # Remove invalid token
                continue
    
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
    mode = 'test'

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
        predicted_token = predict_next_token(model, test_sequence, tokenizer, config['max_len'], device)
        print(f'Input sequence: {test_sequence}')

        # roll out
        generated_sequence = rollout(model, test_sequence, tokenizer, config['max_len'], device)
        print(f'Generated sequence: {generated_sequence}')