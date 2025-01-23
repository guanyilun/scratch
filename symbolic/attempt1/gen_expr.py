"""Generate expressions based on a given set of primitives
and a maximum depth.
"""

#%%
from typing import NamedTuple, Any
from functools import partial

class W3J(NamedTuple):
    l1: int
    l2: int
    l3: int
    m1: int
    m2: int
    m3: int
    def __repr__(self):
        return f"(w3j {self.l1} {self.l2} {self.l3} {self.m1} {self.m2} {self.m3})"

class Add(NamedTuple):
    x: Any
    y: Any
    def __repr__(self):
        return f"(+ {self.x} {self.y})" 

class Sub(NamedTuple):
    x: Any
    y: Any 
    def __repr__(self):
        return f"(- {self.x} {self.y})" 

class Mul(NamedTuple):
    x: Any
    y: Any
    def __repr__(self):
        return f"(* {self.x} {self.y})" 

class Div(NamedTuple):
    x: Any
    y: Any
    def __repr__(self):
        return f"(/ {self.x} {self.y})" 

class Pow(NamedTuple):
    x: Any
    y: Any
    def __repr__(self):
        return f"(^ {self.x} {self.y})" 
    
#%%
# build expression tree
import random

def gen_random_w3j():
    l1 = random.randint(0, 10)
    l2 = random.randint(0, 10)
    l3 = random.randint(0, 10)
    m1 = random.randint(-l1, l1)
    m2 = random.randint(-l2, l2)
    m3 = random.randint(-l3, l3)
    return W3J(l1, l2, l3, m1, m2, m3)

def gen_2arg_expr(op):
    nexpr = min(random.randint(0, 2), len(expr_pool))
    nnum = 2 - nexpr
    exprs = [random.choice(expr_pool) for _ in range(nexpr)]
    lits = [random.randint(-10, 10) for _ in range(nnum)]
    args = exprs + lits
    random.shuffle(args)
    return op(*args)


#%%
expr_pool = []
operator_pool = [Add, Sub, Mul, Div, Pow, W3J]
custom_generators = {
    W3J: gen_random_w3j,
}
max_depth = 1000
for i in range(max_depth):
    op = random.choice(operator_pool)
    if op in custom_generators:
        generator = custom_generators[op]
    else:
        generator = partial(gen_2arg_expr, op)
    expr = generator()
    expr_pool.append(expr)
print(expr_pool)

#%%
w3j = W3J(1,2,3,4,5,6)
# %%
len(w3j)
# %%
# %%
