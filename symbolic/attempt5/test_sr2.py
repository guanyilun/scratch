"""
Perform a symbolic regression by optimizing the numerics in an expression.
See test_sr3 for when the numerics are explicitly specified with variables
like c1, and c2. The loss function is also considerably simplified in test_sr3
compared to this.

"""
#%%
import numpy as np
import sympy
import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import optax
import equinox as eqx
import sympy2jax

def get_evaluator(X):
    n_var = X.shape[0]
    X_sym = sympy.symbols("x:{}".format(n_var))
    mapping = {str(k): v for k, v in zip(X_sym, X)}
    return lambda mod: mod(**mapping)

def optimize_eq_params(expr, X, y, optimizer=None, niter=1000, atol=1e-3, log_step=100, verbose=True):
    expr_mod = sympy2jax.SymbolicModule(expr.evalf())
    eval_mod = get_evaluator(X)
    
    # get all tunable parameters
    params, static = eqx.partition(expr_mod, lambda x: eqx.is_array(x) and x.dtype != jnp.int32)

    if optimizer is None:
        optimizer = optax.sgd(1e-2)

    opt_state = optimizer.init(params)

    def loss_fn(params):
        mod = eqx.combine(params, static)
        y_pred = eval_mod(mod)
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
            if verbose: print(f"{i+1:3d}: loss = {loss_val:.3f}: {eqx.combine(params, static).sympy()}")
        if loss_val < atol:
            if verbose: print(f"{i+1:3d}: loss = {loss_val:.3f}: {eqx.combine(params, static).sympy()}")
            if verbose: print("Converged")
            break
    expr_out = eqx.combine(params, static).sympy()
    return expr_out, loss_val
    
from scipy.optimize import minimize
def optimize_eq_params_scipy(expr, X, y):
    numbers = np.array(sorted(list((expr.atoms(sympy.Number))))).astype(float)
    print(f"numbers: {numbers}")

    def loss_fn(params):
        new_expr = expr.subs({k: v for k, v in zip(numbers, params)})
        # get y pred with new_expr with X as input
        f = sympy.lambdify(sympy.symbols("x:5"), new_expr)
        y_pred = f(*X)
        loss = np.mean((y_pred - y) ** 2)
        return loss

    #Use the scipy.optimize.minimize function
    result = minimize(loss_fn, x0=numbers, method='L-BFGS-B',
                      options={
                        'maxiter': 1000,
                        'ftol': 1e-6,
                     })

    #Update the parameters with the optimized values
    if result.success:
        params_optimized_values = result.x
        expr_out = expr.subs({k: v for k, v in zip(numbers, params_optimized_values)})
        loss = result.fun
    else:
        expr_out = expr
        loss = np.inf
    return expr_out, loss

if __name__ == '__main__':
    X = 2 * np.random.randn(5, 100)
    y = 2.5382 * np.cos(X[3]) + X[0] ** 2 - 0.5

    # suppose an expression is generated
    expr_str = "1.6 * cos(1.5*x3) + x0**2 - 2.0"
    expr = sympy.sympify(expr_str)
    print(f"input: {expr.evalf()}")

    expr_out, loss_val = optimize_eq_params(expr, X, y)
    print(f"output: {expr_out}")

    expr_out, loss_val = optimize_eq_params_scipy(expr, X, y)
    print(f"output: {expr_out}")
# %%
