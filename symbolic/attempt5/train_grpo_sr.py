import os
import re
import argparse
import torch
import random
import numpy as np
from sympy import sympify, Number, Symbol, Function
from test_sr2 import optimize_eq_params, optimize_eq_params_scipy
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.trainer import GRPOConfig, GRPOTrainer


R1_STYLE_SYSTEM_PROMPT = """A conversation between User and Assistant. The user will provide
a list of variables, such as [x0, x1, x2, x3] and a list of operators such as [+, -, *, /],
the assistant will generate a math expression using the given variables, operators, numbers and 
parathesis ().

Possible list of operators and example use includes:

+: x + y
-: x - y
*: x * y
/: x / y
sin: sin(x)
cos: cos(x)
**: x ** y

The assistant first thinks about the reasoning process in the mind and then provides the expression.
The reasoning process and answer are enclosed within <reasoning> </reasoning> and
<answer> </answer> tags, respectively, i.e., <reasoning> reasoning process here </reasoning>
<answer> answer here </answer>.
"""

TASK_SPECIFIC_INSTRUCTIONS = "The expression must be based on the provided list of variable and operators with number parentheses ()."

def generate_dataset(num_samples=1000):
    # generate 4 random integers from 1-13
    prompts = [{
        'prompt': [
            {'role': 'system', 'content': R1_STYLE_SYSTEM_PROMPT + "\n" + TASK_SPECIFIC_INSTRUCTIONS},
            {'role': 'user', 'content': "variables: [x0, x1, x2, x3, x4], operators: [+, -, *, /, sin, cos, **]. Please generate an expression only using the given variables, operators, numbers and parathesis ()."},
        ],
        'answer': {
            'variables': ['x0', 'x1', 'x2', 'x3', 'x4'],
            'operators': ['+', '-', '*', '/', 'sin', 'cos', '**'],
        } # for now, we don't know valid solutions upfront
    } for _ in range(num_samples)]
    return Dataset.from_list(prompts)


def is_valid_format(expression: str, answer: list[int]) -> bool:
    """Check if the expression uses valid numbers and operators."""
    # Define allowed variables and operators
    allowed_vars = answer['variables']
    allowed_ops = answer['operators']
    
    try:
        # Remove spaces if any
        expr = expression.replace(' ', '')
        
        # Try parsing with sympy
        parsed = sympify(expr, evaluate=False)
        
        # Get all symbols and functions used
        symbols_used = {str(s) for s in parsed.free_symbols}
        functions_used = {str(f).split('(')[0] for f in parsed.atoms(Function)}
        # print(f"parsed: {parsed}")
        # print(f"parsed symbols: {parsed.free_symbols}")
        # print(f"parsed functions: {parsed.atoms(Function)}")
        
        # Check all variables are allowed
        if not all(v in allowed_vars for v in symbols_used):
            # print(f"Not all variables are allowed: {symbols_used}")
            return False
            
        # Check all functions are allowed
        if not all(f in allowed_ops for f in functions_used):
            # print(f"Not all functions are allowed: {functions_used}")
            return False
        
        # Check operators used in expression
        ops_used = re.findall(r'[\+\-\*/]{1,2}', expr)  # Match single or double operators
        if not all(op in allowed_ops for op in ops_used):
            # print(f"Not all operators are allowed: {ops_used}")
            return False
        
        # Check if expression can be evaluated with test values
        test_vals = {var: 1.0 for var in allowed_vars}
        parsed.subs(test_vals)
        # print(f"Valid expression: {expression}") 
        # print(f"Symbols used: {symbols_used}")
        # print(f"Functions used: {functions_used}")
        return True
    except Exception as e:
        print(f"Invalid expression: {expression}")
        import traceback
        traceback.print_exc()
        return False


def extract_xml_answer(text: str) -> str:
    try:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
        return answer
    except IndexError:
        return ""

def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has the correct format."""
    pattern = r"^<reasoning>(?:(?!</reasoning>).)*</reasoning>\n<answer>(?:(?!</answer>).)*</answer>$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.match(pattern, r.strip())) for r in responses]
    return [1.0 if match else 0.0 for match in matches]

def reward_format(prompts, completions, answer, **kwargs):
    """Reward function for the prompt-completion pair."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [1.0 if is_valid_format(res, ans) else 0.0 for (res, ans) in zip(extracted_responses, answer)]

# Generate test data similar to test_sr2.py
X_test = 2 * np.random.randn(5, 100)
y_test = 2.5382 * np.cos(X_test[3]) + X_test[0] ** 2 - 0.5

def reward_fit_data(prompts, completions, answer, **kwargs):
    """Reward function that evaluates expressions based on their fit to test data."""
    responses = [completion[0]["content"] for completion in completions]
    responses = [extract_xml_answer(r) for r in responses]
    
    rewards = []
    valids = []
    for res in responses:
        try:
            if not is_valid_format(res, answer[0]):
                rewards.append(0.0)
                valids.append(False)
                continue

            # Parse and optimize expression
            expr = sympify(res)
            # _, loss = optimize_eq_params(expr, X_test, y_test, log_step=10000, verbose=False)  # Reduce logging
            _, loss = optimize_eq_params_scipy(expr, X_test, y_test)  # Reduce logging
            # Convert loss to reward (higher reward for lower loss)
            # Use exponential decay to map loss -> [0,1]
            # reward = np.exp(-loss)
            # reward = 0.1/(loss.item() + 1e-9)
            reward = 1/(loss + 1)
            rewards.append(reward)
            valids.append(True)
          
        except Exception as e:
            import traceback
            traceback.print_exc()
            rewards.append(0.0)
            
    for i in range(5):
        print(f"Expression: {responses[i]}, Valid: {valids[i]} Reward: {rewards[i]}")

    rewards = [0.0 if np.isnan(r) else r for r in rewards]
    return rewards

if __name__ == '__main__':

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    output_dir = f"outputs/{model_name.split('/')[-1]}-GRPO-SR"
    run_name = f"{model_name.split('/')[-1]}-sr"

    # Set memory-related environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    max_prompt_length=256
    max_completion_length=512

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=1e-5,
        beta=0.005,
        optim="adamw_8bit",
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=4,
        num_generations=4,  # group size
        gradient_accumulation_steps=4,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        num_train_epochs=10,
        save_steps=100,
        max_grad_norm=0.1,
        report_to="wandb",
        log_on_each_node=False,
        use_vllm=True,
        vllm_init_kwargs={
            "device": "cuda:0",
            "gpu_memory_utilization": 0.3,
            "max_model_len": max_prompt_length + max_completion_length,
            "dtype": "half",
        },
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logit_computation_mini_batch_size=1,
        enable_profiling=False
    )

    # Parse command line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpoint', type=str, help='Path to checkpoint directory to resume training from')
    # args = parser.parse_args()

    # Load model
    # model_path = args.checkpoint if args.checkpoint else model_name
    model_path = model_name
    # model_path = "outputs/Qwen2.5-0.5B-Instruct-GRPO-SR/checkpoint-200"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # device_map={"": "cuda:1"}
        device_map="auto"
    )

    # Load tokenizer from original model to ensure consistency
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=training_args.max_completion_length,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    dataset = generate_dataset(num_samples=1000)

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_format,
            reward_fit_data
        ],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
