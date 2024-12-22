#%%
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
import chex
import optax
from typing import Protocol, NamedTuple
from functools import partial

class ProbabilityDistribution(Protocol):
    def sample(self, key: jax.random.PRNGKey, nsamps: int) -> jnp.ndarray: ...
    def logpdf(self, z: jnp.ndarray) -> float: ...
    
@chex.dataclass
class MeanField(ProbabilityDistribution):
    """Gaussian mean-field approximation with parameter management"""
    mu: jnp.ndarray
    log_sigma: jnp.ndarray
    
    @property
    def sigma(self):
        return jnp.exp(self.log_sigma) + 1e-5
    
    def sample(self, key, nsamps):
        noise = jax.random.normal(key, shape=(nsamps, self.mu.shape[0]))
        return self.sigma * noise + self.mu
    
    def logpdf(self, z):
        return norm.logpdf(z, self.mu, self.sigma).sum()
    
    def __repr__(self):
        return f"MeanField(μ={self.mu}, σ={self.sigma})"

class VITrainer(NamedTuple):
    """Variational Inference Trainer with monitoring capabilities"""
    nsamps: int
    max_iter: int
    optimizer: optax.GradientTransformation
    lnposterior: callable

    @partial(jax.jit, static_argnums=(0,))
    def compute_elbo(self, model: ProbabilityDistribution, key: jax.random.PRNGKey) -> float:
        """Compute Evidence Lower BOund"""
        z = model.sample(key, self.nsamps)
        lnp = jax.vmap(self.lnposterior)(z)
        lnq = jax.vmap(model.logpdf)(z)
        return jnp.mean(lnp - lnq)

    def train(self, key, initial_model, convergence_tol=None, verbose=True, print_every=100):
        @jax.jit
        def train_step(opt_state, model, key):
            loss, grads = jax.value_and_grad(lambda m: -self.compute_elbo(m, key))(model)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            model = optax.apply_updates(model, updates)
            return opt_state, model, loss
            
        opt_state = self.optimizer.init(initial_model)
        model = initial_model
        
        for i in range(self.max_iter):
            _, key = jax.random.split(key)
            opt_state, model, loss = train_step(opt_state, model, key)
            if verbose and i % print_every == 0:
                print(f'Iteration {i}, Loss: {loss}; {model.mu}')
            if convergence_tol is not None and loss < convergence_tol:
                break
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
    
    trainer = VITrainer(
        nsamps=100,
        max_iter=5000,
        optimizer=optax.adam(learning_rate=0.05),
        lnposterior=lnposterior
    )
    
    trained_model = trainer.train(key, model)
    print(trained_model)
        
# %%
