#%%
from sympy import *
from sympy.unify.core import Compound
from sympy.unify.usympy import unify
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class Rule:
    lhs: Compound
    rhs: Compound
    slots: list[Symbol]

    @classmethod
    def parse(cls, rule_str):
        lhs, rhs = rule_str.split("->")
        # find all slot variables that start with "?"
        q = re.compile(r"\?([a-zA-Z0-9]+)")
        slots_lhs = re.findall(q, lhs)
        slots_rhs = re.findall(q, rhs)
        if not set(slots_rhs).issubset(set(slots_lhs)):
            raise ValueError("more slots on rhs than lhs")
        # make symbols for each slot
        slot_symbols = [Symbol(f"slot_{s}") for s in slots_lhs]
        # replace slot variables with slot symbols
        lhs = q.sub(r"slot_\1", lhs)
        rhs = q.sub(r"slot_\1", rhs)
        return cls(
            lhs=sympify(lhs, evaluate=False),
            rhs=sympify(rhs, evaluate=False),
            slots=slot_symbols
        )
    def __call__(self, expr):
        matches = unify(expr, self.lhs, {}, variables=self.slots)
        # TODO: think about how to handle multiple matches
        # for now, just return the first match
        matches = list(matches)
        if len(matches) == 0:
            return None
        return self.rhs.xreplace(matches[0])
        
@dataclass(frozen=True)
class Prewalk:
    """Apply a rule to an expression in a pre-order traversal (top-down).
    Automatically passthrough mismatched expressions."""
    rule: Rule 
    def __call__(self, expr):
        with evaluate(False):
            if (r:=self.rule(expr)) is not None: return r
            if (not expr.is_Symbol) and (not expr.is_Number):
                return expr.func(*[self(arg) for arg in expr.args])
            else:
                return expr

@dataclass(frozen=True)
class Postwalk:
    """Apply a rule to an expression in a post-order traversal (bottom-up).
    Automatically passthrough mismatched expressions."""
    rule: Rule 
    def __call__(self, expr):
        with evaluate(False):
            if (not expr.is_Symbol) and (not expr.is_Number):
                new_expr = expr.func(*[self(arg) for arg in expr.args])
                return r if (r:=self.rule(new_expr)) is not None else new_expr
            else:
                return r if (r:=self.rule(expr)) is not None else expr


if __name__ == "__main__":
    rule = Rule.parse("?a + ?b + c -> ?b * ?a + c")
    expr = sympify("x + y + c")
    print(rule(expr))

    rule_pre = Prewalk(rule)
    print(rule_pre(expr))

    rule_post = Postwalk(rule)
    print(rule_post(expr))


# old codes that I don't want to delete yet
# expr_decon = deconstruct(expr)

# def traverse(expr):
#     if isinstance(expr, Compound):
#         print(f"op: {expr.op}\t args: {expr.args}")
#     else:
#         assert expr.is_Symbol or expr.is_Number
#         print(f"val: {expr}, type: {type(expr)}")
#     for arg in expr.args:
#         traverse(arg)

# expr = sympify("(x + 1) * (x + 2)")
# traverse(expr_decon)
