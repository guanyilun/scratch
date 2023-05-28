#%% base
from typing import NamedTuple, Type
from contextlib import contextmanager
from abc import ABC, abstractmethod
from dataclasses import dataclass

class Primitive(NamedTuple):
    name: str

add_p = Primitive('add')
mul_p = Primitive('mul')

def add(x, y):
    return bind(add_p, x, y)

def mul(x, y):
    return bind(mul_p, x, y)

#%%
@dataclass
class Tracer:
    """each tracer should carry an abstract value (aval) that will
    be used to represent its value during tracing"""
    @property
    def aval(self): return None
    def __add__(self, x):
        return add(self.aval, x)
    def __mul__(self, x):
        return mul(self.aval, x)
    def __radd__(self, x):
        return add(x, self.aval)
    def __rmul__(self, x):
        return mul(x, self.aval)
    def __iadd__(self, x):
        return self.__add__(x)
    def __imul__(self, x):
        return self.__mul__(x)
    

@dataclass 
class Trace(ABC):
    """Trace defines the transformation rules."""
    interpreter: "Interpreter"
    def lift(self, x):
        """lift a value into a trace"""
        return x
    def lower(self, x):
        """lower a value from a trace"""
        return x
    @abstractmethod
    def process_primitive(self, p, *args):...

class Interpreter(NamedTuple):
    trace_type: Type[Trace]

interpreter_stack = []

@contextmanager
def new_interpreter(trace_type):
    interpreter = Interpreter(trace_type)
    interpreter_stack.append(interpreter)
    yield interpreter
    interpreter_stack.pop()

def get_current_trace():
    interpreter = interpreter_stack[-1]
    return interpreter.trace_type(interpreter)

def bind(p: Primitive, *args):
    trace = get_current_trace()
    args = [trace.lift(arg) for arg in args]
    out = trace.process_primitive(p, *args)
    return trace.lower(out)

#%%
class EvalTrace(Trace): 
    def lift(self, x): 
        """evaluation don't need to lift value to a tracer"""
        return x
    def lower(self, x):
        return x
    def process_primitive(self, p, *args):
        implementations = {
            add_p: lambda x, y: x + y,
            mul_p: lambda x, y: x * y,
        }
        out = implementations[p](*args)
        return out

def make_eval_trace(fun):
    def _transformed(*args):
        with new_interpreter(EvalTrace):
            return fun(*args)
    return _transformed

def test(a, b):
    c = add(a, b)
    d = mul(c, c)
    return add(d, 1)
    
make_eval_trace(test)(1, 2)
#%%
class Expr(NamedTuple):
    op: Primitive
    args: list
    def __repr__(self):
        return f'{self.op.name}({", ".join(map(str, self.args))})'

@dataclass
class GraphTracer(Tracer):
    expr: Expr 
    val: object
    @property
    def aval(self): return self.val
    def __repr__(self):
        return f'GraphTracer({self.expr})'
        
class MakeGraph(Trace):
    def lift(self, x):
        if isinstance(x, Tracer): return x
        return GraphTracer(x, x)
    def lower(self, x):
        return x
    def process_primitive(self, p, *args):
        implementations = {
            add_p: lambda x, y: (Expr(add_p, [x.expr, y.expr]), x.aval+y.aval),
            mul_p: lambda x, y: (Expr(mul_p, [x.expr, y.expr]), x.aval*y.aval),
        }
        expr, val = implementations[p](*args)
        return GraphTracer(expr, val)

def make_graph_trace(fun):
    def _transformed(*args):
        with new_interpreter(MakeGraph):
            return fun(*args)
    return _transformed

make_graph_trace(test)(1, 2)

# output:
# GraphTracer(add(mul(add(1, 2), add(1, 2)), 1))
