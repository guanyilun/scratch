#%%
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from jax.scipy.stats import norm

slope_true = 2
intercept_true = 1
sigma_true = 1
x = jnp.linspace(0, 10, 1000)
key = jax.random.PRNGKey(0)
y = slope_true*x + intercept_true + sigma_true*jax.random.normal(key, shape=x.shape)

plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')

# %%
def lnposterior(theta, x, y):
    slope, intercept = theta
    y_model = slope*x + intercept
    # prior = norm.logpdf(slope, 0, 10) + norm.logpdf(intercept, 0, 10)  # Example priors
    # prior = -0.5*jnp.sum(theta**2)
    prior = -0.5*jnp.sum(theta**2)
    likelihood = -0.5*jnp.sum((y - y_model)**2)
    return prior + likelihood

lnposterior_v = jax.vmap(lnposterior, in_axes=(0, None, None))

def mf_sample(key, nsamps, mu, log_sigma):
    sigma = jnp.exp(log_sigma) + 1e-5
    eps = jax.random.normal(key, shape=(nsamps, mu.shape[0]))
    return eps * sigma + mu

def mf_lnp(theta, mu, log_sigma):
    sigma = jnp.exp(log_sigma) + 1e-5
    return norm.logpdf(theta, mu, sigma).sum()

mf_lnp_v = jax.vmap(mf_lnp, in_axes=(0, None, None))

def loss_fn(model, x, y, key):
    nsamps = 100
    x_samples = mf_sample(key, nsamps, **model)
    return -jnp.mean(lnposterior_v(x_samples, x, y) - mf_lnp_v(x_samples, model['mu'], model['log_sigma']))

# %%
from optax import adam
import optax

nsamps = 100
max_iter = 100
key = jax.random.PRNGKey(0)
optimizer = adam(learning_rate=0.05)

model = {
    'mu': jnp.array([0., 0.]),
    'log_sigma': jnp.array([0., 0.]),
}

opt_state = optimizer.init(model)

@jax.jit
def train_step(opt_state, model, x, y, key):
    loss, grad = jax.value_and_grad(loss_fn)(model, x, y, key)
    update, opt_state = optimizer.update(grad, opt_state)
    model = optax.apply_updates(model, update)
    return opt_state, model, loss 

    
for i in range(max_iter):
    _, key = jax.random.split(key)
    opt_state, model, loss = train_step(opt_state, model, x, y, key)
    if i % 100 == 0:
        print(f'Iteration {i}, Loss: {loss}; {model["mu"]}')

print(model)
# %%
plt.scatter(x, y)
y_true = slope_true*x + intercept_true
plt.plot(x, y_true, color='black', label='Truth')
y_pred = model['mu'][0]*x + model['mu'][1]
plt.plot(x, y_pred, color='red', label='Prediction')
plt.legend()

# %%
