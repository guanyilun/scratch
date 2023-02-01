from core import emulate

def func(a=1, b=1, c=1):
    return a+b+c

print("True answer:", func(a=1.5, b=2.2, c=2.1))
# 5.8

# emulate the same function
@emulate(args={'a': np.linspace(0,3,10), 'b': np.linspace(0,3,10), 'c': np.linspace(0,3,10)}, epoches=1000)
def func(a=1, b=1, c=1):
    return a+b+c

print("Emulated answer:", func(a=1.5, b=2.2, c=2.1))
# 5.787
