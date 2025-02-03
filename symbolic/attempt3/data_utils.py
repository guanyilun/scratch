import torch
from torch.utils.data import Dataset
import random


def create_vocabulary():
    operators = ['+', '-', '*', '/']
    digits = [str(i) for i in range(1, 14)]
    others = ['<start>', '<end>', '(', ')']
    
    actions = [''] + operators + digits + others
    vocab = {token: idx for idx, token in enumerate(actions)}
    return vocab, actions

class Tokenizer:
    def __init__(self):
        # Define the special tokens
        self.special_tokens = {
            '<pad>': 0,
            '<start>': 1,
            '<end>': 2,
        }
        
        # Define operators, numbers, and parentheses
        self.operators = ['+', '-', '*', '/']
        self.numbers = [str(i) for i in range(1, 14)]  # '1' through '13'
        self.parentheses = ['(', ')']
        
        # Create the vocabulary
        self.vocab = {}
        # First add special tokens
        self.vocab.update(self.special_tokens)
        # Then add operators, numbers, and parentheses
        current_idx = len(self.special_tokens)
        for token in self.operators + self.numbers + self.parentheses:
            self.vocab[token] = current_idx
            current_idx += 1
            
        # Create reverse mapping (index to token)
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.pad_token = '<pad>'
        self.actions = list(self.vocab.keys())
        
    def encode(self, sequence, add_start_end=True, pad_to_length=None):
        """
        Encode a sequence of tokens into indices.
        
        Args:
            sequence: List of tokens or space-separated string
            add_start_end: Whether to add start/end tokens
            pad_to_length: Optional length to pad to
            
        Returns:
            List of indices
        """
        # If input is a string, split it into tokens
        if isinstance(sequence, str):
            # Split by spaces but keep parentheses as separate tokens
            tokens = []
            current_token = ''
            for char in sequence:
                if char in [' ']:
                    if current_token:
                        tokens.append(current_token)
                        current_token = ''
                elif char in ['(', ')']:
                    if current_token:
                        tokens.append(current_token)
                        current_token = ''
                    tokens.append(char)
                else:
                    current_token += char
            if current_token:
                tokens.append(current_token)
        else:
            tokens = sequence
            
        # Convert tokens to indices
        try:
            indices = [self.vocab[token] for token in tokens]
        except KeyError as e:
            raise ValueError(f"Unknown token found: {e}. Valid tokens are: {list(self.vocab.keys())}")
            
        # Add start and end tokens if requested
        if add_start_end:
            indices = [self.vocab['<start>']] + indices + [self.vocab['<end>']]
            
        # Pad sequence if requested
        if pad_to_length is not None:
            indices = indices + [self.vocab[self.pad_token]] * (pad_to_length - len(indices))
            
        return indices
    
    def decode(self, indices, skip_special_tokens=True):
        """
        Decode a sequence of indices back into tokens.
        
        Args:
            indices: List of indices
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            List of tokens
        """
        special_token_ids = set() if not skip_special_tokens else {
            self.vocab[token] for token in ['<start>', '<end>', '<pad>']
        }
        
        tokens = []
        for idx in indices:
            if idx not in special_token_ids:
                tokens.append(self.idx_to_token[idx])
                
        return tokens
    
    def decode_to_string(self, indices, skip_special_tokens=True):
        """
        Decode indices to a formatted expression string.
        
        Args:
            indices: List of indices
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Formatted expression string
        """
        tokens = self.decode(indices, skip_special_tokens)
        # Add spaces around operators but not around parentheses
        formatted = ''
        for i, token in enumerate(tokens):
            if token in self.operators:
                formatted += f' {token} '
            elif token in self.parentheses:
                formatted += token
            else:
                formatted += token
        return formatted.strip()
    
    def get_vocab(self):
        """Return the vocabulary dictionary."""
        return self.vocab.copy()
    
    def get_vocab_size(self):
        """Return the size of the vocabulary."""
        return self.vocab_size


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
    # vocab = create_vocabulary()
    # print(vocab)

    from tqdm import tqdm
    tokenizer = Tokenizer()

    expressions = []
    for _ in tqdm(range(100_000)):
        expression = generate_math_expression(max_depth=4)
        expressions.append(expression)
    
    # save to file
    with open('data/math_expressions.txt', 'w') as f:
        for expr in expressions:
            f.write(expr + '\n')
    
    # # Test cases
    # expressions = [
    #     "1 + 2",
    #     "(3 * 4) / 5",
    #     "13 - (11 + 2)",
    #     "7 * (8 + 9)"
    # ]
    
    # print("Testing tokenizer:")
    # print("-" * 50)
    # for expr in expressions:
    #     print(f"\nOriginal expression: {expr}")
    #     # Encode
    #     encoded = tokenizer.encode(expr)
    #     print(f"Encoded: {encoded}")
    #     # Decode
    #     decoded = tokenizer.decode(encoded)
    #     print(f"Decoded tokens: {decoded}")
    #     # Decode to formatted string
    #     formatted = tokenizer.decode_to_string(encoded)
    #     print(f"Formatted expression: {formatted}")
