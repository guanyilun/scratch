import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import numpy as np
from typing import List, Dict
import json
from datetime import datetime
import logging
from sympy import sympify, Number
import re

# Disable VLLM's progress bars
logging.getLogger("vllm").setLevel(logging.WARNING)

# Constants from training script
R1_STYLE_SYSTEM_PROMPT = """A conversation between User and Assistant. The user provides four numbers 
between 1 and 13, and the Assistant generates a math expression using these numbers and operators 
+, -, *, / and parentheses () that evaluates to 24. Each number must be used exactly once.
The assistant first thinks about the reasoning process in the mind and then provides the expression.
The reasoning process and answer are enclosed within <reasoning> </reasoning> and
<answer> </answer> tags, respectively, i.e., <reasoning> reasoning process here </reasoning>
<answer> answer here </answer>."""

TASK_SPECIFIC_INSTRUCTIONS = "The answer must be a math expression using the given number and operators +, -, *, / and parentheses () that evaluates to 24. Use each number exactly once."

def is_valid_format(expression: str, numbers: List[int]) -> bool:
    """Check if the expression uses valid numbers and operators."""
    try:
        # Remove parentheses for number checking
        clean_expr = re.sub(r'[()*/+\-]', ' ', expression)
        expr_numbers = [int(n) for n in clean_expr.split()]

        sorted_numbers = sorted(expr_numbers)
        if sorted_numbers != sorted(numbers):
            return False

        # Check for valid characters
        if not all(c in '0123456789()+-*/' for c in expression):
            return False
            
        # Check parentheses balance
        if expression.count('(') != expression.count(')'):
            return False
            
        return True
    except:
        return False

def evaluate_expression(expression: str) -> float:
    """Safely evaluate a mathematical expression string using sympy."""
    try:
        # Remove spaces if any
        expr = expression.replace(' ', '')
        
        # Evaluate using sympy
        result = float(sympify(expr).evalf())
        
        # Check if result is a valid number
        if not isinstance(result, (int, float)) or not isinstance(sympify(expr), Number):
            return float('nan')
                
        return result
    except:
        return float('nan')

def extract_xml_answer(text: str) -> str:
    try:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
        return answer
    except IndexError:
        return ""

def evaluate_model(
    model_path: str,
    problems: List[List[int]],
    batch_size: int = 4,
    num_samples: int = None,
    save_results: bool = True,
    gpu_memory_utilization: float = 0.3,
) -> Dict:
    print("Initializing evaluation...")

    # Initialize VLLM with progress indicator
    with tqdm(total=2, desc="Loading model components") as pbar:
        llm = LLM(
            model=model_path,
            dtype="half",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=768,
            device="cuda:0",
            enable_chunked_prefill=True,
        )
        pbar.update(1)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=768,
            padding_side='right',
            truncation_side='right'
        )
        pbar.update(1)

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,  # Matching max_completion_length from training
        stop_token_ids=[tokenizer.eos_token_id],
    )

    # Process problems
    print("Processing problems...")
    if num_samples:
        problems = problems[:num_samples]
    total_samples = len(problems)
    print(f"Processing {total_samples} problems")

    results = []
    correct = 0
    total = 0

    # Create progress bar
    progress_bar = tqdm(
        total=total_samples,
        desc="Processing samples",
        unit="examples",
        dynamic_ncols=True,
    )

    progress_bar.set_postfix({
        'acc': '0.00%',
        'correct': '0',
    })

    # Process in batches
    for i in range(0, total_samples, batch_size):
        batch_problems = problems[i:i + batch_size]
        current_batch_size = len(batch_problems)

        # Prepare prompts using same format as training
        prompts = [
            [
                {'role': 'system', 'content': R1_STYLE_SYSTEM_PROMPT + "\n" + TASK_SPECIFIC_INSTRUCTIONS},
                {'role': 'user', 'content': "Generate a math expression using the numbers " + str(prob) + " and operators +,-,*,/ and parentheses () that equals 24. Use each number exactly once."},
            ] for prob in batch_problems
        ]

        # Convert to chat format
        formatted_prompts = [
            tokenizer.apply_chat_template(
                p,
                tokenize=False,
                add_generation_prompt=True
            )
            for p in prompts
        ]

        # Generate responses
        outputs = llm.generate(
            formatted_prompts,
            sampling_params,
        )

        # Process responses
        for j, output in enumerate(outputs):
            response = output.outputs[0].text
            answer = extract_xml_answer(response)
            
            # Evaluate correctness
            is_valid = is_valid_format(answer, batch_problems[j])
            value = evaluate_expression(answer) if is_valid else float('nan')
            is_correct = is_valid and abs(value - 24) < 1e-10

            # Store result
            result = {
                'numbers': batch_problems[j],
                'generated_answer': answer,
                'full_response': response,
                'is_valid_format': is_valid,
                'evaluated_value': None if np.isnan(value) else value,
                'correct': is_correct
            }
            results.append(result)

            # Update metrics
            if is_correct:
                correct += 1
            total += 1

        # Update progress
        progress_bar.update(current_batch_size)
        progress_bar.set_postfix({
            'acc': f'{(correct/total)*100:.2f}%',
            'correct': f'{correct}/{total}',
        })

    progress_bar.close()

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    metrics = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'model_path': model_path,
        'timestamp': datetime.now().isoformat()
    }

    # Save results
    if save_results:
        save_path = f"game24_eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(save_path, 'w') as f:
            json.dump({
                'metrics': metrics,
                'results': results
            }, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return metrics

# Example evaluation run
if __name__ == "__main__":
    print("Starting 24 Game evaluation...")
    from game24 import generate_problems
    
    checkpoint_path = "outputs/Qwen2.5-0.5B-Instruct-GRPO/checkpoint-620"  # Update path as needed
    test_problems = generate_problems(10)  # Generate 100 test problems
    
    metrics = evaluate_model(
        model_path=checkpoint_path,
        problems=test_problems,
        batch_size=4,
        num_samples=None,
        save_results=True,
        gpu_memory_utilization=0.3,
    )

    print("\nFinal Evaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")