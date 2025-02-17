"""
compared to sr2, this implements a shorter loss function calculation, and will
optimize over free parameters specified as c0,.... This is useful to reduce the
number of parameters fitting (adding relations between them)

another thing to consider is to use standard scipy.minimize instead of optimizer.
"""
#%%
import numpy as np
import sympy
import jax
import optax
import sympy2jax

#%%
def get_evaluator(X):
    n_var = X.shape[0]
    X_sym = sympy.symbols("x:{}".format(n_var))
    mapping = {str(k): v for k, v in zip(X_sym, X)}
    def evaluater(mod, params):
        bindings = {
            f"c{i}": v for i, v in enumerate(params)
        } | mapping
        return mod(**bindings)
    return evaluater


def get_n_free_pars(mod):
    n_pars = len(set([x for x in jax.tree.leaves(mod) if isinstance(x, str) and x.startswith('c')]))
    return n_pars


def optimize_eq_params(
    expr, 
    X, 
    y, 
    optimizer=None, 
    niter=1000, 
    atol=1e-4, 
    log_step=100, 
    seed=0
):
    expr_mod = sympy2jax.SymbolicModule(expr)
    eval_mod = get_evaluator(X)
    n_free = get_n_free_pars(expr_mod)
    key = jax.random.PRNGKey(seed)
    
    # get all tunable parameters
    params = jax.random.normal(key, (n_free,))

    if optimizer is None:
        optimizer = optax.sgd(1e-1)

    opt_state = optimizer.init(params)

    def loss_fn(params):
        y_pred = eval_mod(expr_mod, params)
        loss = optax.l2_loss(y_pred, y).mean()
        return loss

    @jax.jit
    def make_step(params, opt_state):
        loss_val, grad = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    for i in range(niter):
        params, opt_state, loss_val = make_step(params, opt_state)
        if (i+1) % log_step == 0:
            print(f"> step {i+1}: loss = {loss_val}")
        if loss_val < atol:
            print(f"> step {i+1}: loss = {loss_val}")
            print(f"> Reached convergence: {loss_val} < {atol} (atol)")
            break
    else:
        print(f"-> maximum number of iterations reached: {niter}")
    expr_out = expr_mod.sympy().subs({f"c{i}": v for i, v in enumerate(params)})
    return expr_out, loss_val
    

if __name__ == '__main__':

    X = 2 * np.random.randn(5, 100)
    y = 2.5382 * np.cos(X[3]) + X[0] ** 2 - 0.5

    # suppose an expression is generated
    expr_str = "c0 * cos(c2*x3) + x0**2 - c1"
    expr = sympy.sympify(expr_str)
    print(f"input: {expr}")

    expr_out, loss_val = optimize_eq_params(expr, X, y)
    print(f"output: {expr_out}")


# %%
