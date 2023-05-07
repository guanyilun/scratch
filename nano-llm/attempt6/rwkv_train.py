#%%
import jax
# jax.config.update('jax_platform_name', 'cpu')  # debug
from jax import jit, numpy as np
from jax.nn.initializers import zeros
import optax

from lra_benchmarks.listops.configs.base_listops_config import get_config

from rwkv_batch import rwkv_net_batch
from rwkv_train_utils import init_weight_info, init_weights, init_uniform, LRABatchConfig

#%%
config = get_config()

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


#%%
lra = LRABatchConfig.from_s5(config.batch_size, "lra_benchmarks", "listops-classification")
sampler = lra.samplers['train']

#%%
# initialize weights
key = jax.random.PRNGKey(0)
winfo = init_weight_info(lra.n_classes_in, config.emb_dim, config.num_layers, config.mlp_dim)
weights = init_weights(winfo, None, zeros)  # key is not required for all_zeros
weights['head']['weight'] = init_uniform(key, winfo['head']['weight'], a=-1e-4, b=1e-4)

# initialize optimizers
optimizer = optax.lion(**lion_params)
opt_state = optimizer.init(weights)

#%%
@jax.value_and_grad
def loss_fn(weights, batch):
    x, y, lengths = batch
    y_pred = rwkv_net_batch(x, **weights)
    return optax.softmax_cross_entropy_with_integer_labels(y_pred[np.arange(x.shape[0]), lengths], y).mean()

@jit
def make_step(weights, opt_state, batch):
    loss_val, grads = loss_fn(weights, batch)
    updates, opt_state = optimizer.update(grads, opt_state, weights)
    weights = optax.apply_updates(weights, updates)
    return weights, opt_state, loss_val

for i in range(config.num_train_steps):
    batch = sampler()
    weights, opt_state, loss_val = make_step(weights, opt_state, batch)
    if i % 10 == 0:
        print(f"Step {i}: loss = {loss_val:.4f}")

np.save("rwkv_weights.npy", weights)
# %%
