"""various data loading utilities in general"""
import jax.numpy as np
from typing import NamedTuple, Any
from tokenizers import Tokenizer as HFTokenizer


class Tokenizer(NamedTuple):
    """Thin wrapper around HFTokenizer, mostly to get rid of the `.ids`"""
    tokenizer: HFTokenizer
    @classmethod
    def from_file(cls, tokenizer_file):
        tokenizer = HFTokenizer.from_file(tokenizer_file)
        return cls(tokenizer=tokenizer)
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    def decode(self, ids):
        return self.tokenizer.decode(ids)


class JSONLoader(NamedTuple):
    """jsonl dataloader"""
    file: str
    tokenizer: Tokenizer

    @classmethod
    def from_file(cls, json_file, tokenizer=None):
        if tokenizer is None:
            tokenizer = Tokenizer.from_file("20B_tokenizer.json")
        return cls(file=json_file, tokenizer=tokenizer)

    def get_dataloader(self, batch_size=8, block_size=256):
        def data_generator():
            with open(self.file, 'r') as f:
                x_batch = []
                y_batch = []
                l_batch = []
                for line in f:
                    tokens = self.tokenizer.encode(line.strip())
                    prev, next = tokens[:-1], tokens[1:]
                    x, l = fold_into_blocks(prev, block_size)
                    y, _ = fold_into_blocks(next, block_size)
                    for (x_, y_, l_) in zip(x, y, l):
                        x_batch.append(x_)
                        y_batch.append(y_)
                        l_batch.append(l_)
                        if len(x_batch) == batch_size:
                            yield np.array(x_batch), np.array(y_batch), np.array(l_batch)
                            x_batch = []
                            y_batch = []
                            l_batch = []
        return data_generator()

def fold_into_blocks(arr, block_size, pad_value=0):
    """fold a 1D array into blocks of size block_size, padding `pad_value` if necessary.
    returns a tuple of (folded_arr, lengths) where lengths is a 1D array of the lengths of each block before
    padding."""
    arr = np.array(arr)
    nrows = int(np.ceil(len(arr) / block_size))
    last_row_length = len(arr) % block_size if len(arr) % block_size != 0 else block_size
    padded_arr = np.pad(arr, (0, block_size-last_row_length), mode='constant', constant_values=pad_value)
    folded_arr = padded_arr.reshape(nrows, block_size)
    lengths = np.array([block_size] * (nrows-1) + [last_row_length])
    return folded_arr, lengths
