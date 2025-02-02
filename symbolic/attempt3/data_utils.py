import torch
from torch.utils.data import Dataset


def create_vocabulary():
    operators = ['+', '-', '*', '/']
    digits = [str(i) for i in range(1, 14)]
    others = ['<start>', '<end>', '(', ')']
    
    actions = [''] + operators + digits + others
    vocab = {token: idx for idx, token in enumerate(actions)}
    return vocab, actions


class Tokenizer:
    def __init__(self):
        self.vocab, self.actions = create_vocabulary()
        self.idx_to_token = {token: idx for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.pad_token = ''
        
    def encode(self, sequence, add_start_end=True):
        indices = [self.vocab[token] for token in sequence]
        if add_start_end:
            indices = [self.vocab['<start>']] + indices + [self.vocab['<end>']]
        return indices
    
    def decode(self, indices, skip_special_tokens=True):
        if skip_special_tokens:
            indices = [idx for idx in indices if idx not in [self.vocab['<start>'], self.vocab['<end>']]]
        return [self.idx_to_token[idx] for idx in indices]


class MathExpressionDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_len):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_indices = self.tokenizer.encode(seq)

        # Pad sequence if necessary
        if len(input_indices) < self.max_len - 1:
            padding_length = self.max_len - 1 - len(input_indices)
            input_indices = input_indices + [self.tokenizer.vocab[self.tokenizer.pad_token]] * padding_length
        else:
            # If sequence is too long, truncate it
            input_indices = input_indices[-(self.max_len - 1):]

        # Prepare input and target tensors
        target_indices = input_indices[1:]
        input_indices = input_indices[:-1]
            
        return {
            'input': torch.tensor(input_indices, dtype=torch.long),
            'target': torch.tensor(target_indices, dtype=torch.long)
        }


import random

def generate_math_expression(max_depth=3):
    """
    Generate a random mathematical expression using specified operators and numbers.
    
    Args:
        max_depth (int): Maximum depth of nested expressions to prevent excessive complexity
        
    Returns:
        str: A valid mathematical expression
    """
    # Available operations and numbers
    operators = ['+', '-', '*', '/']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
    
    def generate_subexpression(depth):
        if depth >= max_depth:
            # Return a random number when max depth is reached or randomly
            return random.choice(numbers)
        
        # Decide whether to generate a simple expression or a parenthesized one
        if random.random() < 0.1:
            # Generate a simple expression without parentheses
            return random.choice(numbers)
        
        # Generate a complex expression
        left = generate_subexpression(depth + 1)
        operator = random.choice(operators)
        right = generate_subexpression(depth + 1)
        
        # Decide whether to wrap in parentheses
        if random.random() < 0.4:
            return f"({left} {operator} {right})"
        return f"{left} {operator} {right}"
    
    return generate_subexpression(0)


if __name__ == '__main__':
    vocab = create_vocabulary()
    print(vocab)

    for _ in range(100):
        expression = generate_math_expression(max_depth=4)
        print(f"Generated expression: {expression}")
