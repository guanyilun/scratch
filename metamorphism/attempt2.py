#%% base
from typing import NamedTuple, Type
from contextlib import contextmanager
from abc import ABC, abstractmethod
from dataclasses import dataclass

class Primitive(NamedTuple):
    name: str
    fun: "callable"
    def __hash__(self) -> int:
        return hash(self.name)

#%%
@dataclass
class Tracer:
    """each tracer should carry an abstract value (aval) that will
    be used to represent its value during tracing"""
    @property
    def aval(self): return None

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
primitives = []

def morphosis(**kwargs):
    # make primitives
    for name, fun in kwargs.items():
        primitive = Primitive(name, fun)
        primitives[primitive] = primitive
    primitives.append(kwargs)
    try:
        yield
    finally:
        primitives.pop()

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

