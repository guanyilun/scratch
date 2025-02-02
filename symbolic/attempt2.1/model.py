from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct", device_map="auto", trust_remote_code=True)

def generate_math_solution(prompt):
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    # Decode and return response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
if __name__ == "__main__":
    # math_prompt = "Solve the equation: 2x + 5 = 13"
    math_prompt = '''Generate an expression for a given list of operators and vectors.
    
    valid functions: 
      +(number, number)->number
      -(number, number)->number
      *(number, number)->number
      /(number, number)->number
      /(vector, number)->vector
      *(vector, number)->vector
      *(number, vector)->vector
      +(vector, vector)->vector
      -(vector, vector)->vector
      cross(vector, vector)->vector
      dot(vector, vector)->number
      modulus(vector)->number
      
    Allowed vectors (total of 3) denoted as: r1, r2, r3
    
    Valid number expression:
    
    dot(r1 + r2, r3)
    dot(r1, r3) + dot(r2, r3) + dot(r1, r2)
    mod(r1) + mod(r2) + mod(r3)
    dot(r1/mod(r1), r2/mod(r2)

    Please generate another valid number expression, wrap your answer in <answer></answer> tags.'''
    solution = generate_math_solution(math_prompt)
    print(f"Problem: {math_prompt}")
    print(f"Solution: {solution}")
