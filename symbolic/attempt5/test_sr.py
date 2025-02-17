#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sympy
import jax
import jax.numpy as jnp
import optax
from functools import partial

X = 2 * np.random.randn(5, 100)
# y = 2.5382 * np.cos(X[3]) + X[0] ** 2 - 0.5
y = 51.5 * X[0] + 33.1*X[1] + 2.1*X[2] + 1.3*X[3] + 0.5*X[4]

expr = "x1 + 4*x2 + 3*x3 + 2*x4 + 5*x5"
expr = sympy.sympify(expr)

def define_problem(X, y):
    n_var = X.shape[0]
    x = sympy.symbols("x1:{}".format(n_var + 1))
    y = sympy.symbols("y")
    return x, y

X_sym, y_sym = define_problem(X, y)

# suppose an expression is generated
expr_str = "3*x1 + 4*x2 + 3*x3 + 2*x4 + 5*x5"
expr = sympy.sympify(expr_str)
expr.evalf()


#%%
# now that we have parsed the expression, next we generate a jax module
expr_mod = sympy2jax.SymbolicModule(expr.evalf())

#%%
# get all tunable parameters
params, static = eqx.partition(expr_mod, eqx.is_array)

#%%
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

def loss_fn(params):
    mod = eqx.combine(params, static)
    y_pred = mod(**{str(k): v for k, v in zip(X_sym, X)})
    loss = optax.l2_loss(y_pred, y).mean()
    return loss

@jax.jit
def make_step(params, opt_state):
    loss_val, grad = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val

#%%
optimizer = optax.sgd(1e-3)
opt_state = optimizer.init(params)
params, static = eqx.partition(expr_mod, eqx.is_array)

#%%
niter = 1000
for i in range(niter):
    params, opt_state, loss_val = make_step(params, opt_state)
    if np.isclose(loss_val, 0):
        break
    if i % 100 == 0:
        print(f"{i:3d}: loss = {loss_val:.3f}: {jax.tree_leaves(params)}")
    
#%%
eqx.combine(params, static).sympy()
# %%