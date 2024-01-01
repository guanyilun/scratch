"""
Attempt 1: find the lagrangian for a harmonic oscillator with
symbolic regression
"""
#%%
import jax
import numpy as np
import jax.numpy as jnp
import sympy as sp
from genetic import GeneticAlgorithm
from symbolic import LLMFuzzer

def euler_lagrangian(expr, backend='numpy'):
    """Given a lagrangian expression, return the euler-lagrangian
    equation.
    """
    a, x, y, t = sp.symbols("a x y t")
    rules = {
        'add_t': [
            (x, sp.sympify("x(t)")),
            (y, sp.sympify("y(t)")),
        ],
        'dx_to_v': [
            (sp.diff(sp.sympify("x(t)"), t), sp.sympify("y(t)")),
        ],
        'drop_t': [
            (sp.sympify("x(t)"), x),
            (sp.sympify("y(t)"), y),
        ],
        'dv_to_a': [
            (sp.diff(sp.sympify("y(t)"), t), a),
        ]
    }    
    expr = sp.sympify(expr)
    lhs = sp.diff(sp.diff(expr, y).subs(rules['add_t']), t)
    lhs = lhs.subs(rules['dx_to_v']).subs(rules['dv_to_a']).subs(rules['drop_t'])
    rhs = sp.diff(expr, x)
    # print(f"EXPR: {expr}\tLHS: {lhs}\tRHS: {rhs}")
    if sp.simplify(lhs - rhs).is_constant():
        raise ValueError("Euler-Lagrange solution trivially satisfied")
    lhs_fun = sp.lambdify((x, y, a), lhs, backend)
    rhs_fun = sp.lambdify((x, y), rhs, backend)
    return lhs_fun, rhs_fun

class LagrangianRegGA(GeneticAlgorithm):
    """A regression genetic algorithm for finding the lagrangian
    
    Parameters
    ----------
    fuzzer: Fuzzer
        A fuzzer object to generate random expressions
    conjugates: list of tuples
        A list of tuples of the form (x, v) where x and v are
        the symbols for the position and velocity respectively.
    data: dict
        A dictionary of the form {x: [x1, x2, ...], v: [v1, v2, ...]}
        where x and v are the symbols for the position and velocity
        respectively.
    """
    def __init__(self, fuzzer, data, **kwargs):
        super().__init__(**kwargs)
        self.fuzzer = fuzzer
        self.data = data

    def w_func(self, fitness):
        # higher precision to avoid annoying tolerance problem 
        # with np.random.choice
        fitness = np.asarray(fitness, dtype=np.float64)
        fitness[np.isnan(fitness)] = -np.inf  # nan -> -inf

        # straightly positive
        weights = fitness - fitness[~(np.isneginf(fitness))].min() + 1

        # abandon bad individuals
        weights[np.isneginf(fitness)] = 0

        # quadratic weighting
        # weights **= 2

        # probability adds up to 1
        return weights / weights.sum()

    def init_population(self, n_pop):
        return self.fuzzer.rand_expr(n_expr=n_pop)
       
    def evaluate_fitness(self, expr):
        """Given a lagrangian expression, evaluate its fitness by
        solving Euler-Lagrange equation and comparing the solution
        with the given data.
        """
        try:
            lhs, rhs = euler_lagrangian(expr)
        except:
            return -np.inf
        lhs_data = lhs(self.data['x'], self.data['v'], self.data['a'])
        rhs_data = rhs(self.data['x'], self.data['v'])
        rms = np.sqrt(np.sum((lhs_data - rhs_data)**2))
        fitness = -np.log(rms + 1e-10)  # regularize to avoid pole
        return fitness

    def mutate(self, child):
        return child
    
    def crossover(self, p1, p2):
        return self.fuzzer.gen_expr_similar(p1, p2, show_pbar=False)

    def is_converged(self, fitness):
        if fitness.max() > 5:
            return True
        return False
    
#%% simulate data    
m = 1 
k = 10
A = 3
phi = 1

def x_fun(t):
    return A * jnp.sin(jnp.sqrt(k/m) * t + phi)

v_fun = jax.grad(x_fun)
a_fun = jax.grad(v_fun)

t = jnp.linspace(0, 10, 100)
x_data = jax.vmap(x_fun)(t)
v_data = jax.vmap(v_fun)(t)
a_data = jax.vmap(a_fun)(t)

# from matplotlib import pyplot as plt
# plt.plot(t, x_data, label="x")
# plt.plot(t, v_data, label="v")
# plt.plot(t, a_data, label="a")
# plt.xlabel("t")
# plt.legend()

#%%
# make a fuzzer to produce random expressions
fuzzer = LLMFuzzer(
    symbols=["x", "y"], 
    operators=["+", "-", "*", "/"],
)

# build symbolic regression engine
lr = LagrangianRegGA(
    fuzzer=fuzzer,
    data={'x': x_data, 'v': v_data, 'a': a_data},
)

# execute genetic algorithm
lr.run(n_pop=100, max_gen=100)
