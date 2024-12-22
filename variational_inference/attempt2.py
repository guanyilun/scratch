#%%
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from jax.scipy.stats import norm
import optax

from typing import NamedTuple

class MeanField(NamedTuple):
    mu: jnp.ndarray
    log_sigma: jnp.ndarray
    
    def sample(self, key, nsamps):
        sigma = jnp.exp(self.log_sigma) + 1e-5
        eps = jax.random.normal(key, shape=(nsamps, self.mu.shape[0]))
        return eps * sigma + self.mu
    
    def logpdf(self, z):
        sigma = jnp.exp(self.log_sigma) + 1e-5
        return norm.logpdf(z, self.mu, sigma).sum()

class VIConfig(NamedTuple):
    nsamps: int
    max_iter: int
    optimizer: optax.GradientTransformation
    lnposterior: callable

    def train(self, key, model, tol=None, print_every=100):
        lnposterior = jax.vmap(self.lnposterior, in_axes=(0,))
        lnq = jax.vmap(model.logpdf, in_axes=(0,))
        def loss_fn(model, key):
            z = model.sample(key, self.nsamps)
            eblo = jnp.mean(lnposterior(z) - lnq(z))
            return -eblo

        @jax.jit
        def train_step(opt_state, model, key):
            loss, grad = jax.value_and_grad(loss_fn)(model, key)
            update, opt_state = self.optimizer.update(grad, opt_state)
            model = optax.apply_updates(model, update)
            return opt_state, model, loss

        opt_state = self.optimizer.init(model)
        for i in range(self.max_iter):
            _, key = jax.random.split(key)
            opt_state, model, loss = train_step(opt_state, model, key)
            if i % print_every == 0:
                print(f'Iteration {i}, Loss: {loss}; {model.mu}')
            # check for convergence
            if tol is not None and (loss < tol):
                print(f'Converged at iteration {i}: Loss: {loss}')
                break
        print(f"Reached max iterations: Loss: {loss}")
        return model

if __name__ == '__main__':
    # simulate data
    slope_true = 5
    intercept_true = 1
    sigma_true = 1
    x = jnp.linspace(0, 10, 1000)
    key = jax.random.PRNGKey(0)
    y = slope_true*x + intercept_true + sigma_true*jax.random.normal(key, shape=x.shape)

    # define the likelihood
    def lnposterior(theta):
        slope, intercept = theta
        y_model = slope*x + intercept
        prior = -0.5*jnp.sum(theta**2)
        likelihood = -0.5*jnp.sum((y - y_model)**2)
        return prior + likelihood
    
    # define the model
    model = MeanField(
        mu=jnp.array([0., 0.]),
        log_sigma=jnp.array([0., 0.])
    )
    
    # define the config
    vi_config = VIConfig(
        nsamps=100,
        max_iter=500,
        optimizer=optax.adam(learning_rate=0.05),
        lnposterior=lnposterior
    )
    
    # train the model
    model = vi_config.train(key, model, tol=1e-5)
    print(model)
    
    # plot the results
    plt.scatter(x, y)
    y_true = slope_true*x + intercept_true
    plt.plot(x, y_true, color='black', label='Truth')
    y_pred = model.mu[0]*x + model.mu[1]
    plt.plot(x, y_pred, color='red', label='Prediction')
    plt.legend()
    plt.show()
# %%
