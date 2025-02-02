class FormulaTokenizer:
    def __init__(self, max_constants=3):
        self.vocab = [
            '<PAD>', '<START>', '<END>', 'x', '+', '-', '*', '/', '(', ')'
        ]
        # Add parameter tokens: C0, C1, ..., C_{max_constants-1}
        self.vocab += [f'C{i}' for i in range(max_constants)]  
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.max_constants = max_constants

    def tokenize(self, formula):
        tokens = []
        buffer = ''
        for c in formula:
            if c == 'C':
                buffer = 'C'
            elif buffer.startswith('C') and c.isdigit():
                buffer += c
                # Check if the constructed token (e.g., C0) exists
                if buffer in self.token_to_id:
                    tokens.append(self.token_to_id[buffer])
                    buffer = ''
                else:
                    # Invalid constant, reset buffer (ignore invalid token)
                    buffer = ''
            else:
                if buffer:
                    # Incomplete C token (e.g., C without a digit), reset buffer
                    buffer = ''
                if c in self.token_to_id:
                    tokens.append(self.token_to_id[c])
        # Handle any remaining buffer (though unlikely with current logic)
        if buffer and buffer in self.token_to_id:
            tokens.append(self.token_to_id[buffer])
        return tokens

    def decode(self, tokens):
        return ''.join([self.id_to_token[t] for t in tokens])

if __name__ == "__main__":
    tokenizer = FormulaTokenizer()
    test_cases = [
        ("C0*x + C1", [10, 6, 3, 4, 11]),
        ("x + 1", [3, 4]),
        ("C2*x - C1", [12, 6, 3, 5, 11]),
        ("x + x", [3, 4, 3]),
        ("C0 + C1 * x", [10, 4, 11, 6, 3]),
        (" (x + x) ", [8, 3, 4, 3, 9]),
        ("C0", [10]),
        ("", []),
        ("x", [3]),
        ("x+x", [3, 4, 3]),
        ("C1/x", [11, 7, 3]),
        ("C0*C1", [10, 6, 11]),
        ("C0 + C1 - C2", [10, 4, 11, 5, 12]),
        ("((x))", [8, 8, 3, 9, 9]),
    ]

    for formula, expected_tokens in test_cases:
        tokens = tokenizer.tokenize(formula)
        assert tokens == expected_tokens, f"Test case failed for formula '{formula}': Expected {expected_tokens}, got {tokens}"
        print(f"Test case passed for formula '{formula}'")