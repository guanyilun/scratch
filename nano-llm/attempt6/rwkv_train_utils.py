#%%
from jax import numpy as np
import jax
from jax.nn.initializers import uniform


def init_weight_info(n_vocab, n_channel, n_layer, n_ffn, n_vocab_out=None):
    # default to the same vocab size for output
    n_vocab_out = n_vocab_out or n_vocab
    info = {
        'emb': {'weight': (n_vocab, n_channel)},
        'blocks': {},
        'ln_out': {'weight': (n_channel,), 'bias': (n_channel,)},
        'head': {'weight': (n_vocab_out, n_channel)},
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

def init_uniform(key, shape, a=-1e-4, b=1e-4, dtype=np.float32):
    # uniform in [a, b) range, default to [-1e-4, 1e-4) following rwkv recommendation
    return uniform(scale=b-a)(key, shape, dtype=dtype) + a
