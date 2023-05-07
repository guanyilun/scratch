#%%
import jax
from jax import jit, numpy as np
from jax.nn.initializers import zeros
import optax

from rwkv_batch import rwkv_net_batch
from rwkv_train_utils import init_weight_info, init_weights, init_uniform
from lra_utils import LRABatchConfig

#%%
adam_params = {
    'learning_rate': 1e-4,
    'beta1': 0.9,
    'beta2': 0.999,
    'eps': 1e-8,
}
lion_params = {
    'learning_rate': 1e-4,
    'b1': 0.95,
    'b2': 0.98,
    'weight_decay': 0.01
}

run_config = {
    'n_epoch': 3,
    'batch_size': 32,
    'eval_freq': 50,
    'n_train_step': 5000,
    'n_channel': 512,
    'n_layer': 4,
    'n_ffn': 1024,
    'opt': 'lion',
    'opt_params': lion_params,
}

#%%
cache_path = "lra_benchmarks"
lra_config = LRABatchConfig.from_s5(run_config['batch_size'], cache_path, "listops-classification")

#%%
# initialize weights
key = jax.random.PRNGKey(0)
winfo = init_weight_info(
    lra_config.n_classes_in,
    run_config['n_channel'],
    run_config['n_layer'],
    run_config['n_ffn'],
    n_vocab_out=lra_config.n_classes_out
)
weights = init_weights(winfo, None, zeros)  # key is not required for zeros init
weights['head']['weight'] = init_uniform(key, winfo['head']['weight'], a=-1e-4, b=1e-4)

# initialize optimizers
optimizer = {'lion': optax.lion, 'adam': optax.adam}[run_config['opt']](**run_config['opt_params'])
opt_state = optimizer.init(weights)

#%%
def loss_fn(weights, batch):
    x, y, lengths = batch
    y_pred = rwkv_net_batch(x, **weights)
    return optax.softmax_cross_entropy_with_integer_labels(y_pred[np.arange(x.shape[0]), lengths], y).mean()

loss_fn_grad = jax.value_and_grad(loss_fn)

@jit
def make_step(weights, opt_state, batch):
    loss_val, grads = loss_fn_grad(weights, batch)
    updates, opt_state = optimizer.update(grads, opt_state, weights)
    weights = optax.apply_updates(weights, updates)
    return weights, opt_state, loss_val

i_step = 0
for _ in range(run_config['n_epoch']):
    trainloader = lra_config.get_dataloader('train')
    for batch in trainloader:
        weights, opt_state, loss_val = make_step(weights, opt_state, batch)
        if i_step % run_config['eval_freq'] == 0:
            print(f"step {i_step}, loss {loss_val:.4f}")
        i_step += 1

#%%
np.save("rwkv_weights.npy", weights)
