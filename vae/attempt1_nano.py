import jax
import jax.numpy as np

def linear(x, w, b):
    return x @ w + b

def relu(x):
    return np.maximum(x, 0)

def encoder(x, mlp_weights, mu_weights, logvar_weights):
    x = linear(x, **mlp_weights)
    x = relu(x)
    mu = linear(x, **mu_weights)
    logvar = linear(x, **logvar_weights)
    return mu, logvar

def decoder(z, mlp_weights, output_weights):
    z = linear(z, **mlp_weights)
    z = relu(z)
    return linear(z, **output_weights)

def sample(mu, logvar, key):
    return mu + np.exp(0.5 * logvar) * jax.random.normal(key, mu.shape)

def vae(x, encoder_weights, decoder_weights, key):
    mu, logvar = encoder(x, **encoder_weights)
    z = sample(mu, logvar, key)
    return decoder(z, **decoder_weights), mu, logvar

def loss(x, x_recon, mu, logvar):
    recon_loss = np.sum((x - x_recon)**2)
    kl_loss = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
    return recon_loss + kl_loss

class PNGKeyGen:
    def __init__(self, key):
        self.key = key
    def __call__(self):
        self.key, subkey = jax.random.split(self.key)
        return subkey
    

def init_params(in_dim, hidden_dim, latent_dim, keygen):
    params = {
        'encoder_weights': {
            'mlp_weights': {
                'w': jax.random.normal(keygen(), (in_dim, hidden_dim)),
                'b': jax.random.normal(keygen(), (hidden_dim,))
            },
            'mu_weights': {
                'w': jax.random.normal(keygen(), (hidden_dim, latent_dim)),
                'b': jax.random.normal(keygen(), (latent_dim,))
            },
            'logvar_weights': {
                'w': jax.random.normal(keygen(), (hidden_dim, latent_dim))/100,  # logvar should be small
                'b': jax.random.normal(keygen(), (latent_dim,))/100             
            }
        },
        'decoder_weights': {
            'mlp_weights': {
                'w': jax.random.normal(keygen(), (latent_dim, hidden_dim)),
                'b': jax.random.normal(keygen(), (hidden_dim,))
            },
            'output_weights': {
                'w': jax.random.normal(keygen(), (hidden_dim, in_dim)),
                'b': jax.random.normal(keygen(), (in_dim,))
            }
        }
    }
    return params

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    keygen = PNGKeyGen(key)

    batch_size = 100
    data_dim = 500
    hidden_dim = 100
    latent_dim = 10

    params = init_params(data_dim, hidden_dim, latent_dim, keygen)
    x = jax.random.normal(keygen(), (batch_size, data_dim))
    x_recon, mu, logvar = vae(x, key=keygen(), **params)
    print(loss(x, x_recon, mu, logvar))