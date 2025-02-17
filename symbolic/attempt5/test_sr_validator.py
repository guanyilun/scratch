import pytest
from train_grpo_sr import is_valid_format

# Test fixture for common test configuration
@pytest.fixture
def answer():
    return {
        'variables': ['x0', 'x1', 'x2', 'x3', 'x4'],
        'operators': ['+', '-', '*', '/', 'sin', 'cos', '**']
    }

def test_valid_basic_expressions(answer):
    """Test valid basic arithmetic expressions."""
    assert is_valid_format("x0 + x1", answer) == True
    assert is_valid_format("x2 * x3", answer) == True
    assert is_valid_format("x4 - x0", answer) == True
    assert is_valid_format("x1 / x2", answer) == True
    assert is_valid_format("x3 ** 2", answer) == True

def test_valid_complex_expressions(answer):
    """Test valid complex expressions with multiple operations."""
    assert is_valid_format("sin(x0) + cos(x1)", answer) == True
    assert is_valid_format("x0 * x1 + x2 ** 2", answer) == True
    assert is_valid_format("(x3 + x4) / (x0 - x1)", answer) == True
    assert is_valid_format("sin(x2 ** 2) + cos(x3 / x4)", answer) == True

def test_invalid_variables(answer):
    """Test expressions with invalid variables."""
    assert is_valid_format("y + x0", answer) == False
    assert is_valid_format("x5 * x1", answer) == False
    assert is_valid_format("a * b", answer) == False
    assert is_valid_format("x0 + z", answer) == False

def test_invalid_operators(answer):
    """Test expressions with invalid operators."""
    assert is_valid_format("x0 % x1", answer) == False
    assert is_valid_format("tan(x0)", answer) == False
    assert is_valid_format("sqrt(x1)", answer) == True  # not considered a function internally in sympy
    assert is_valid_format("x0 && x1", answer) == False

def test_invalid_syntax(answer):
    """Test expressions with invalid syntax."""
    assert is_valid_format("x0 +", answer) == False
    assert is_valid_format("* x1", answer) == False
    assert is_valid_format("x0 x1", answer) == False
    assert is_valid_format("(x0 + x1", answer) == False
    assert is_valid_format("sin()", answer) == False

def test_whitespace_handling(answer):
    """Test expressions with different whitespace patterns."""
    assert is_valid_format("x0+x1", answer) == True
    assert is_valid_format("  x0  +  x1  ", answer) == True
    assert is_valid_format("sin( x0 )", answer) == True
    assert is_valid_format("\tx0\t+\tx1\t", answer) == True

def test_decimal_numbers(answer):
    """Test expressions with decimal numbers."""
    assert is_valid_format("2.5 * x0", answer) == True
    assert is_valid_format("x1 + 3.14", answer) == True
    assert is_valid_format("x2 ** 2.0", answer) == True

def test_negative_numbers(answer):
    """Test expressions with negative numbers."""
    assert is_valid_format("-x0", answer) == True
    assert is_valid_format("-1 * x2", answer) == True