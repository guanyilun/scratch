#%%
from typing import Dict, Optional 
from gen_expr_custom import Symbol, Expr, Op, Number

def unify(expr1: Expr, expr2: Expr, bindings: Optional[Dict[Symbol, Expr]] = None) -> Optional[Dict[Symbol, Expr]]:
    """
    Unify two expressions and return a dictionary of variable bindings.
    If unification fails, return None.
    """
    if bindings is None:
        bindings = {}

    # If the expressions are identical, return the current bindings
    if expr1 == expr2:
        return bindings

    # If one of the expressions is a variable, bind it to the other expression
    if isinstance(expr1, Symbol):
        return _unify_variable(expr1, expr2, bindings)
    if isinstance(expr2, Symbol):
        return _unify_variable(expr2, expr1, bindings)

    # If both expressions are operations, recursively unify their arguments
    if isinstance(expr1, Op) and isinstance(expr2, Op):
        if expr1.op != expr2.op:
            return None  # Operations must match
        # Unify the left and right arguments
        bindings = unify(expr1.left, expr2.left, bindings)
        if bindings is None:
            return None
        bindings = unify(expr1.right, expr2.right, bindings)
        return bindings

    # If one expression is a number and the other is not, unification fails
    if isinstance(expr1, Number) or isinstance(expr2, Number):
        return None

    # If none of the above cases apply, unification fails
    return None

def _unify_variable(var: Symbol, expr: Expr, bindings: Dict[Symbol, Expr]) -> Optional[Dict[Symbol, Expr]]:
    """
    Unify a variable with an expression.
    """
    if var in bindings:
        # If the variable is already bound, check if the bound value matches the expression
        return unify(bindings[var], expr, bindings)
    else:
        # Bind the variable to the expression
        bindings[var] = expr
        return bindings

        
if __name__ == '__main__':
    # Define some expressions
    x = Symbol('x')
    y = Symbol('y')
    expr1 = Op('+', x, y)
    expr2 = Op('+', Number(2), y)

    # Unify the expressions
    bindings = unify(expr1, expr2)
    print(bindings)  # Output: {x: Number(2)}
# %%
