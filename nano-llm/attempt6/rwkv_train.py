"""train rwkv using long-range arena benchmark dataset"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import logging

import jax
from jax import jit, numpy as np
from jax.nn.initializers import zeros, glorot_normal
import optax
import wandb
import os.path as op

from rwkv_basic import rwkv_net
from rwkv_batch import rwkv_net_batch
from rwkv_utils import get_tokenizer, rnn_generate
from rwkv_train_utils import init_weight_info, init_weights, init_uniform, KeyGen, get_loss_fn, get_acc_fn
from data_utils import load_npy_as_dataloader

use_wandb = True
adam_params = {
    'learning_rate': 1e-4,
    'b1': 0.9,
    'b2': 0.999,
    'eps': 1e-8,
}
lion_params = {
    'learning_rate': 1e-4,
    'b1': 0.95,
    'b2': 0.98,
    'weight_decay': 0.01
}
run_config = {
    'name': 'rwkv-shakespeare',
    'data': 'data/shakespeare.npy',
    'n_epoch': 100,
    'batch_size': 4,
    'eval_freq': 1000,
    'n_channel': 768,
    'n_layer': 12,
    'n_ffn': 3072,
    # 'opt': 'adam',
    # 'opt_params': adam_params,
    'opt': 'lion',
    'opt_params': lion_params,
    'block_size': 256,
    'save_step': 10000,
}

if use_wandb:
    wandb_run = wandb.init(
        project="inside-transformer",
        config=run_config,
    )

tokenizer = get_tokenizer()

# initialize weights
logging.info("initializing weights...")
winfo = init_weight_info(
    tokenizer.get_vocab_size(),
    run_config['n_channel'],
    run_config['n_layer'],
    run_config['n_ffn'],
)

keygen = KeyGen()

# option 1:
# all zero init but head and embedding
weights = init_weights(winfo, keygen, zeros)  # key is not required for zeros init
weights['emb']['weight'] = init_uniform(keygen(), winfo['emb']['weight'], a=-1e-4, b=1e-4)
weights['head']['weight'] = init_uniform(keygen(), winfo['head']['weight'], a=-1e-4, b=1e-4)

# option 2:
# glorot_normal for all 2d matrices and zero for all 1d vectors
# w_init_fn = lambda key, shape: glorot_normal()(key, shape) if len(shape) == 2 else zeros(key, shape)
# weights = init_weights(winfo, keygen, w_init_fn)
logging.info("weights initialized")

# initialize optimizers
logging.info("initializing optimizer...")
optimizer = {'lion': optax.lion, 'adam': optax.adam}[run_config['opt']](**run_config['opt_params'])
opt_state = optimizer.init(weights)
logging.info("optimizer initialized")

#%%
# setup loss, its grad, accuracy and validation
loss_fn = get_loss_fn(rwkv_net_batch)
loss_fn_grad = jax.value_and_grad(loss_fn)
acc_fn = get_acc_fn(rwkv_net_batch)

def get_validation_results(weights):
    prompt = "The quick brown fox jumps over the lazy"
    output = rnn_generate(rwkv_net, weights, prompt, 50, tokenizer)
    res = {'output': output}
    return res

@jit
def make_step(weights, opt_state, batch):
    loss_val, grads = loss_fn_grad(weights, batch)
    updates, opt_state = optimizer.update(grads, opt_state, weights)
    weights = optax.apply_updates(weights, updates)
    return weights, opt_state, loss_val

i_step = 0
logging.info("start training...")
for _ in range(run_config['n_epoch']):
    trainloader = load_npy_as_dataloader(run_config['data'], batch_size=run_config['batch_size'], block_size=run_config['block_size'])
    for batch in trainloader:
        weights, opt_state, loss_val = make_step(weights, opt_state, batch)
        if i_step % 10 == 0:
            print(f"step: {i_step}, batch loss: {loss_val}")
        if i_step % run_config['eval_freq'] == 0:
            print(f"step: {i_step}, batch loss: {loss_val}")
            res = get_validation_results(weights)
            if use_wandb:
                wandb.log({
                    "batch_loss": loss_val,
                    "n_tokens_trained": i_step * run_config['batch_size'] * run_config['block_size'],
                    "generated": wandb.Html(res['output'])
                })
        if "n_train_step" in run_config and i_step >= run_config['n_train_step']:
            break
        if i_step % run_config['save_step'] == 0:
            ofile = op.join(wandb_run.dir, f"rwkv_weights_{i_step}.npy") if use_wandb else f"rwkv_weights_{i_step}.npy"
            np.save(ofile, weights)
        i_step += 1

ofile = op.join(wandb_run.dir, "rwkv_weights.npy") if use_wandb else "rwkv_weights.npy"
np.save(ofile, weights)

if use_wandb: wandb.finish()

# example loading saved weights with np
# res = np.load("rwkv_weights.npy", allow_pickle=True).item()
