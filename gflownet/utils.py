from typing import List
import sympy
from dataclasses import dataclass
from sympy.parsing.sympy_parser import parse_expr
import numpy as np

@dataclass
class ParsingError(Exception):
    """Custom exception for parsing errors"""
    message: str
    position: int
    expression: str

    def __str__(self):
        return f"{self.message} at position {self.position}: '{self.expression}'"

def parse_to_sympy(actions: List[str]) -> sympy.Expr:
    """
    Convert a list of actions into a sympy expression.

    Args:
        actions: List of strings representing mathematical operations and numbers

    Returns:
        sympy.Expr: The parsed mathematical expression

    Raises:
        ParsingError: If the expression is invalid or cannot be parsed
    """
    try:
        expression = []
        stack = []
        last_token_type = None

        for i, action in enumerate(actions):
            if not action:
                continue

            if action == 'end':
                if last_token_type == 'operator':
                    raise ParsingError("Expression ends with operator", i-1, ''.join(expression))
                if stack:
                    raise ParsingError("Unclosed parenthesis", stack[-1][1], ''.join(expression))
                break

            current_token = ""

            if action.endswith('('):
                if not action.startswith(('+', '-', '*', '/')):
                    raise ParsingError("Invalid opening parenthesis format", i, ''.join(expression))
                if last_token_type == 'number':
                    raise ParsingError("Missing operator before parenthesis", i, ''.join(expression))
                current_token = '('
                stack.append(('(', i))
                if len(action) > 1:
                    if last_token_type is None:
                        if action[0] in ('*', '/'):
                            raise ParsingError("Invalid starting operator", i, ''.join(expression))
                    expression.append(action[0])
                last_token_type = 'operator'

            elif action.endswith(')'):
                if not stack:
                    raise ParsingError("Unmatched closing parenthesis", i, ''.join(expression))
                if last_token_type == 'operator':
                    raise ParsingError("Operator before closing parenthesis", i, ''.join(expression))
                stack.pop()
                current_token = ')'
                last_token_type = 'number'

            elif action in ['+', '-', '*', '/']:
                if last_token_type != 'number':
                    raise ParsingError("Invalid operator placement", i, ''.join(expression))
                current_token = action
                last_token_type = 'operator'

            elif action.isdigit():
                if last_token_type == 'number':
                    raise ParsingError("Adjacent numbers without operator", i, ''.join(expression))
                current_token = action
                last_token_type = 'number'

            else:
                raise ParsingError(f"Invalid token: {action}", i, ''.join(expression))

            if current_token:
                expression.append(current_token)

        if not expression:
            raise ParsingError("Empty expression", 0, '')

        expr_str = ''.join(expression)
        try:
            return parse_expr(expr_str, evaluate=False)
        except (sympy.SympifyError, SyntaxError) as e:
            raise ParsingError(f"Invalid mathematical expression: {str(e)}", 0, expr_str)

    except Exception as e:
        if isinstance(e, ParsingError):
            raise
        raise ParsingError(f"Unexpected error: {str(e)}", 0, ''.join(expression))

def run_test_case(actions: List[str], expected_success: bool = True, expected_result: str = None):
    """Helper function to run test cases"""
    try:
        result = parse_to_sympy(actions)
        if expected_success:
            if expected_result is not None:
                result_str = str(result)
                assert result_str == expected_result, f"Expected {expected_result}, got {result_str}"
            print(f"✓ Success: {actions} -> {result}")
            return True
        else:
            print(f"✗ Test failed: Expected failure but got success: {result}")
            return False
    except ParsingError as e:
        if not expected_success:
            print(f"✓ Expected error caught: {e}")
            return True
        print(f"✗ Unexpected error: {e}")
        return False
    except AssertionError as e:
        print(f"✗ Test failed: {e}")
        return False

def count_operators(expr):
    """
    Count the number of operators in a SymPy expression.

    Args:
        expr: A SymPy expression

    Returns:
        int: The number of operators in the expression
    """
    if expr.is_Atom:
        return 0
    count = 1
    for arg in expr.args:
        count += count_operators(arg)
    return count

def reward(actions: List[str]) -> float:
    try:
        expr = parse_to_sympy(actions)
        if float(expr) == 24:
            reward = 2
        else:
            return 0
        return np.exp(reward - 0.1 * count_operators(expr))
    except (ParsingError, sympy.SympifyError):
        return 0

def test_operator_counting():
    test_cases = [
        {"expr_str": "4*6", "expected_ops": 1},
        {"expr_str": "4*6+0", "expected_ops": 2},
        {"expr_str": "4+5*6", "expected_ops": 2},
        {"expr_str": "(4+5)*6", "expected_ops": 2},
        {"expr_str": "4*6/2", "expected_ops": 2},
        {"expr_str": "4", "expected_ops": 0},
        {"expr_str": "4*6*2", "expected_ops": 1}
    ]

    for i, test_case in enumerate(test_cases):
        expr = sympy.parse_expr(test_case["expr_str"], evaluate=False)
        ops = count_operators(expr)
        print(f"\nTest case {i+1}:")
        print(f"Expression: {test_case['expr_str']}")
        print(f"Expected operators: {test_case['expected_ops']}, Got: {ops}")
        print(f"Pass: {ops == test_case['expected_ops']}")
        print(f"Evaluates to: {float(expr)}")

def test_reward_function():
    test_cases = [
        {"input": ["4", "*", "6"], "expected_result": np.exp(2 - 0.1 * 1)},
        {"input": ["4", "*", "6", "+", "0"], "expected_result": np.exp(2 - 0.1 * 2)},
        {"input": ["1", "+", "2"], "expected_result": 0},
        {"input": ["4", "*", "6", "/", "1"], "expected_result": np.exp(2 - 0.1 * 2)}
    ]

    for i, test_case in enumerate(test_cases):
        result = reward(test_case["input"])
        print(f"\nTest case {i+1}:")
        print(f"Expression: {''.join(test_case['input'])}")
        print(f"Expected reward: {test_case['expected_result']}, Got: {result}")
        print(f"Pass: {abs(result - test_case['expected_result']) < 1e-10}")

if __name__ == "__main__":
    print("Running test cases...")
    test_cases = [
        {'actions': ['1', '+', '2', 'end'], 'expected_success': True, 'expected_result': '1 + 2'},
        {'actions': ['+(', '1', '+', '2', '+)', '*', '3', 'end'], 'expected_success': True, 'expected_result': '3*(1 + 2)'},
        {'actions': ['1', 'end'], 'expected_success': True, 'expected_result': '1'},
        {'actions': ['1', '+', 'end'], 'expected_success': False},
        {'actions': ['+(', '1', '+', '2', 'end'], 'expected_success': False},
        {'actions': ['end'], 'expected_success': False},
        {'actions': ['1', '+', '2', '*', '3'], 'expected_success': True, 'expected_result': '1 + 2*3'},
        {'actions': ['+(', '1', '+', '2', '+)', '+', '3'], 'expected_success': True, 'expected_result': '(1 + 2) + 3'},
        {'actions': ['1', '2', '+', '3'], 'expected_success': False},
        {'actions': ['1', '+', '+', '2'], 'expected_success': False}
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        run_test_case(**test)

    print("\nTesting operator counting...")
    test_operator_counting()

    print("\nTesting reward function...")
    test_reward_function()
