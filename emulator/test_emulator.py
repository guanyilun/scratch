#%%
import numpy as np
from core import emulate

def func(a=1, b=1, c=1):
    return a+2*b+3*c

print("True answer:", func(a=1.5, b=2.2, c=2.1))
# 12.20

# emulate the same function
@emulate(samples={'a': np.random.randn(1000), 'b': np.random.randn(1000), 'c': np.random.randn(1000)})
def func(a=1, b=1, c=1):
    return a+2*b+3*c

print("Emulated answer:", func(a=1.5, b=2.2, c=2.1))
# 12.11