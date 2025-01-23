"""generate random sum of fibbonacci numbers"""

# %%
# earlier attempt
# from sympy.unify.rewrite import rewriterule
# n = sp.Symbol("n")
# rule = rewriterule(fib(n), fib(n-1) + fib(n-2), [n], lambda n: n.is_integer and n > 1)
# list(rule(expr))
# [sp.simplify(x) for x in rule(expr)]

import json
import numpy as np
from typing import NamedTuple
from copy import deepcopy
from tqdm import tqdm

class Fib(NamedTuple):
    n: int
    def compress(self):
        return int(self.n)

def create_init_fib(n, low=0, high=100):
    return [Fib(n) for n in np.random.randint(0, 100, 3)]

def expand_recurse(fib_n):
    if fib_n.n > 1:
        return [Fib(fib_n.n - 1), Fib(fib_n.n - 2)]
    else:
        return [fib_n]

def rand_expand_n(expr_list, n=100):
    expr_list = deepcopy(expr_list)
    for i in range(n):
        i_expand = np.random.choice(len(expr_list))
        expr = expr_list.pop(i_expand)
        expr_list.extend(expand_recurse(expr))
    np.random.shuffle(expr_list)
    return expr_list

def make_pair(n_start=3, n_expand=100):
    fibs = create_init_fib(n_start)
    outputs = rand_expand_n(fibs, n_expand)
    to_save = {
        "input": [fib.compress() for fib in fibs],
        "output": [fib.compress() for fib in rand_expand_n(outputs)],
    }
    return to_save

# %%
n_train = 10000
n_validate = 1000
n_test = 1000

data_cfg = 
    "n_start": 3,
    "n_expand": 100,
}
np.random.seed(42)

with open("train.json", "w") as f:
    for i in tqdm(range(n_train)):
        sample = make_pair(**data_cfg)
        line = json.dumps(sample) 
        f.write(line + "\n")

with open("validate.json", "w") as f:
    for i in tqdm(range(n_validate)):
        sample = make_pair(**data_cfg)
        line = json.dumps(sample) 
        f.write(line + "\n")

with open("test.json", "w") as f:
    for i in tqdm(range(n_test)):
        sample = make_pair(**data_cfg)
        line = json.dumps(sample) 
        f.write(line + "\n")
# %%
