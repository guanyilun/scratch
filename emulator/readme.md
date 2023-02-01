# Emulator

The idea of this work is to build a simple to use emulator in python that can emulate generic numerical functions regardless of its implementation details. 

Here's my idea of how one should interact with this program, suppose we have the following function, and we want to build an emulator for it,

```python
def calculation(a=1,b=2,c=3):
    return a+b+c
```

we should just be able to do

```python
@emulate
def calculation(a=1,b=2,c=3):
    return a+b+c
```

and expect the calculation function is an emulated version of it instead of its actual implementation. The philosophy is to make the interaction with emulator as seamless as possible.