#%%
from jax import numpy as np
import jax
from jax.nn.initializers import uniform
from jax import lax

from pathlib import Path
from typing import NamedTuple

from s5.dataloading import Datasets, DataLoader

#%%
def init_weight_info(n_vocab, n_channel, n_layer, n_ffn):
    info = {
        'emb': {'weight': (n_vocab, n_channel)},
        'blocks': {},
        'ln_out': {'weight': (n_channel,), 'bias': (n_channel,)},
        'head': {'weight': (n_vocab, n_channel)},
    }
    for i in range(n_layer):
        block = {
            'att': {
                'o_proj': (n_channel, n_channel),
                'k_proj': (n_channel, n_channel),
                'v_proj': (n_channel, n_channel),
                'r_proj': (n_channel, n_channel),
                'time_mix_r': (n_channel,),
                'time_mix_k': (n_channel,),
                'time_mix_v': (n_channel,),
                'time_decay': (n_channel,),
                'time_first': (n_channel,),
            },
            'ffn': {
                'k_proj': (n_ffn, n_channel),
                'v_proj': (n_channel, n_ffn),
                'r_proj': (n_channel, n_channel),
                'time_mix_k': (n_channel,),
                'time_mix_r': (n_channel,),
            },
            'ln1': {'weight': (n_channel,), 'bias': (n_channel,)},
            'ln2': {'weight': (n_channel,), 'bias': (n_channel,)},
        }
        # convention in rwkv: ln0 is in first block
        if i == 0: block['ln0'] = {'weight': (n_channel,), 'bias': (n_channel,)}
        info['blocks'][i] = block
    return info

def init_weights(weight_info, key, init_fn):
    return jax.tree_map(lambda x: init_fn(key, x), weight_info, is_leaf=lambda x: isinstance(x, tuple))

def init_uniform(key, shape, a=-1e-4, b=1e-4):
    return uniform(scale=b-a)(key, shape, dtype=np.float32) + a

#%%
class LRABatchConfig(NamedTuple):
    block_size: int
    batch_size: int
    s5_dataloaders: DataLoader
    train_size: int
    n_classes_in: int
    n_classes_out: int

    @classmethod
    def from_s5(cls, batch_size: int, cache_path: Path, dataset_name: str, seed: int = 0):
        create_dataset_fn = Datasets[dataset_name]
        trainloader, valloader, testloader, aux_dataloaders, n_classes, seq_len, in_dim, train_size = create_dataset_fn(
            cache_path, seed=seed, bsz=batch_size)
        return cls(block_size=seq_len, batch_size=batch_size,
                   s5_dataloaders={'train': trainloader, 'val': valloader, 'test': testloader}, train_size=train_size,
                     n_classes_in=in_dim, n_classes_out=n_classes)

    @property
    def samplers(self):
        def get_sampler(loader):
            loader_sampler = iter(loader)
            def sampler():
                x, y, l = next(loader_sampler)
                x = trim_or_pad(np.array(x), self.block_size)
                return x, np.array(y), np.array(l['lengths'])
            return sampler
        return {k: get_sampler(v) for k, v in self.s5_dataloaders.items()}

    @property
    def dataloaders(self):
        def get_dataloader(loader: DataLoader):
            def data_generator():
                loader_iter = iter(loader)
                while True:
                    x, y, l = next(loader_iter)
                    x = trim_or_pad(np.array(x), self.block_size)
                    yield x, np.array(y), np.array(l['lengths'])
            return data_generator()
        return {k: get_dataloader(v) for k, v in self.s5_dataloaders.items()}

def trim_or_pad(x, max_length):
    if x.shape[-1] >= max_length:
        return x[..., :max_length]
    else:
        return lax.pad(x, 0, ((0,0,0),(0,max_length-x.shape[-1],0)))
# %%
# lra = LRABatchConfig.from_s5(24, "lra_benchmarks", "listops-classification")
# sampler = lra.samplers['train']
# for i in range(10):
#     x, y, l = sampler()
#     print(x.shape)
# %%
