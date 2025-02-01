import torch
from torch.utils.data import Dataset


def create_vocabulary():
    actions = ['', '+(', '+)', '-(', '-)', '*(', '*)', '/(', '/)'] + [str(i) for i in range(1, 14)]
    vocab = {token: idx for idx, token in enumerate(actions)}
    return vocab, actions


class Tokenizer:
    def __init__(self):
        self.vocab, self.actions = create_vocabulary()
        self.idx_to_token = {token: idx for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
    def encode(self, sequence):
        return [self.vocab[token] for token in sequence]
    
    def decode(self, indices):
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
        input_indices = self.tokenizer.encode(seq[:-1])
        target_index = self.tokenizer.decode([seq[-1]])[0]
        
        # Pad sequence if necessary
        if len(input_indices) < self.max_len - 1:
            padding_length = self.max_len - 1 - len(input_indices)
            input_indices = input_indices + self.tokenizer.encode(['']) * padding_length
        else:
            # If sequence is too long, truncate it
            input_indices = input_indices[-(self.max_len - 1):]
            
        return {
            'input': torch.tensor(input_indices, dtype=torch.long),
            'target': torch.tensor(target_index, dtype=torch.long)
        }


if __name__ == '__main__':
    vocab = create_vocabulary()
    print(vocab)
