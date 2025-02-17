import itertools
import random
from operator import add, sub, mul, truediv

def solve_24(numbers):
    OPERATORS = [(add, '+'), (sub, '-'), (mul, '*'), (truediv, '/')]
    epsilon = 1e-6

    for perm in itertools.permutations(numbers):
        for ops in itertools.product(OPERATORS, repeat=3):
            op1, op2, op3 = ops
            op1_func, op1_sym = op1
            op2_func, op2_sym = op2
            op3_func, op3_sym = op3
            a, b, c, d = perm

            # Structure 1: ((a OP1 b) OP2 c) OP3 d
            try:
                val = op1_func(a, b)
                val = op2_func(val, c)
                val = op3_func(val, d)
                if abs(val - 24) < epsilon:
                    return f"(({a} {op1_sym} {b}) {op2_sym} {c}) {op3_sym} {d}"
            except ZeroDivisionError:
                pass

            # Structure 2: (a OP1 (b OP2 c)) OP3 d
            try:
                val = op2_func(b, c)
                val = op1_func(a, val)
                val = op3_func(val, d)
                if abs(val - 24) < epsilon:
                    return f"({a} {op1_sym} ({b} {op2_sym} {c})) {op3_sym} {d}"
            except ZeroDivisionError:
                pass

            # Structure 3: a OP1 ((b OP2 c) OP3 d)
            try:
                val = op2_func(b, c)
                val = op3_func(val, d)
                val = op1_func(a, val)
                if abs(val - 24) < epsilon:
                    return f"{a} {op1_sym} (({b} {op2_sym} {c}) {op3_sym} {d})"
            except ZeroDivisionError:
                pass

            # Structure 4: a OP1 (b OP2 (c OP3 d))
            try:
                val = op3_func(c, d)
                val = op2_func(b, val)
                val = op1_func(a, val)
                if abs(val - 24) < epsilon:
                    return f"{a} {op1_sym} ({b} {op2_sym} ({c} {op3_sym} {d}))"
            except ZeroDivisionError:
                pass

            # Structure 5: (a OP1 b) OP2 (c OP3 d)
            try:
                val1 = op1_func(a, b)
                val2 = op3_func(c, d)
                val = op2_func(val1, val2)
                if abs(val - 24) < epsilon:
                    return f"({a} {op1_sym} {b}) {op2_sym} ({c} {op3_sym} {d})"
            except ZeroDivisionError:
                pass

    return None

def generate_and_solve():
    while True:
        numbers = [random.randint(1, 13) for _ in range(4)]
        solution = solve_24(numbers)
        if solution:
            print(f"Generated numbers: {numbers}")
            print(f"Solution: {solution}")
            break

def generate_problems(n_probs):
    probs = []
    while len(probs) < n_probs:
        numbers = [random.randint(1, 13) for _ in range(4)]
        solution = solve_24(numbers)
        if solution:
            probs.append(numbers)
    return probs

if __name__ == "__main__":
    n_probs = 10000
    probs = generate_problems(n_probs)
    # save
    import json
    with open("probs.json", "w") as f:
        json.dump(probs, f)