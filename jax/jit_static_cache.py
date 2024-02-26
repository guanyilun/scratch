"""
I had an idea that static arguments in jax jit can be used as a mechanism
for caching calculation results. It also helps getting non-jittable code
into jax. Here is a concrete example:

"""

#%%
# mimic some expensive external class that we cannot jit
import time
class Expensive:
    def __init__(self, i):
        self.i = i
    def compute(self):
        time.sleep(self.i+1)
        return self.i

#%%
import jax

# this will not work because tracing time.sleep is not possible
@jax.jit
def f(x, i):
    return x + Expensive(i).compute()
f(10, 3)

#%%
from functools import partial
# this will work because `i` will not be traced
@partial(jax.jit, static_argnums=(1,))
def f(x, i):
    return x + Expensive(i).compute()
f(10, 3)  # this call will take 3 sec
#%%
# subsequent calls will be fast: Expensive computation is cached for `i=3`
from timeit import timeit
%timeit f(14, 3)

# output:
# 10.4 us per loop

#%%
# change the static argument will recompute the result
f(14, 4)  # this call will take 4 sec

%timeit f(14, 4)
%timeit f(14, 3)

# both of these are fast because they are cached
