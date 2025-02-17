import os
import re
import argparse
import torch
import random
from sympy import sympify, Number
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.trainer import GRPOConfig, GRPOTrainer
from game24 import generate_problems

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

R1_STYLE_SYSTEM_PROMPT = """A conversation between User and Assistant. The user provides four numbers 
between 1 and 13, and the Assistant generates a math expression using these numbers and operators 
+, -, *, / and parentheses () that evaluates to 24. Each number must be used exactly once.
The assistant first thinks about the reasoning process in the mind and then provides the expression.
The reasoning process and answer are enclosed within <reasoning> </reasoning> and
<answer> </answer> tags, respectively, i.e., <reasoning> reasoning process here </reasoning>
<answer> answer here </answer>."""

TASK_SPECIFIC_INSTRUCTIONS = "The answer must be a math expression using the given number and operators +, -, *, / and parentheses () that evaluates to 24. Use each number exactly once."

def generate_dataset(num_samples=10000):
    # generate 4 random integers from 1-13
    prompts = []
    probs = generate_problems(num_samples)
    prompts = [{
        'prompt': [
            {'role': 'system', 'content': R1_STYLE_SYSTEM_PROMPT + "\n" + TASK_SPECIFIC_INSTRUCTIONS},
            {'role': 'user', 'content': "Generate a math expression using the numbers " + str(prob) + " and operators +,-,*,/ and parentheses () that equals 24. Use each number exactly once."},
        ],
        'answer': prob  # for now, we don't know valid solutions upfront
    } for prob in probs]
    return Dataset.from_list(prompts)

def is_valid_format(expression: str, answer: list[int]) -> bool:
    """Check if the expression uses valid numbers and operators."""
    try:
        # Remove parentheses for number checking
        clean_expr = re.sub(r'[()*/+\-]', ' ', expression)
        numbers = [int(n) for n in clean_expr.split()]

        sorted_numbers = sorted(numbers)
        if sorted_numbers != sorted(answer):
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
        
        # Also validate the numbers used are <=13 by extracting them
        # clean_expr = re.sub(r'[()*/+\-]', ' ', expr)
        # numbers = [int(n) for n in clean_expr.split()]
        # # Double check numbers are between 1 and 13
        # if not all(1 <= n <= 13 for n in numbers):
        #     return float('nan')
        
        return result
    except:
        return float('nan')

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

def expression_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function that checks if the expression is valid."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [1.0 if is_valid_format(r, a) else 0.0 for r, a in zip(extracted_responses, answer)]

# def correctness_reward_func(prompts, completions, **kwargs) -> list[float]:
#     """Reward function that checks if the expression evaluates to 24."""
#     responses = [completion[0]['content'] for completion in completions]
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     values = [evaluate_expression(r) for r in extracted_responses]
#     print(f"Expression: {extracted_responses[0]}, Value: {values[0]}")
#     return [1.0 if abs(v - 24) < 1e-10 else 0.0 for v in values]

def correctness_reward_func(prompts, completions, answer, **kwargs):
    """Combimed reward function"""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    valid_rewards = [1.0 if is_valid_format(r, a) else 0.0 for r, a in zip(extracted_responses, answer)]
    values = [evaluate_expression(r) for r in extracted_responses]
    value_rewards = [1.0 if abs(v - 24) < 1e-10 else 0.0 for v in values]
    for i in range(5):
        print(f"Expression: {extracted_responses[i]}, Value: {values[i]}")
    rewards = [v * c for v, c in zip(valid_rewards, value_rewards)]
    return rewards

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"

output_dir = f"outputs/{model_name.split('/')[-1]}-GRPO"
run_name = f"{model_name.split('/')[-1]}-24game"

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
# model_path = model_name
model_path = "outputs/Qwen2.5-0.5B-Instruct-GRPO/checkpoint-620"
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
dataset = generate_dataset(num_samples=10000)

# Initialize trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        format_reward_func,
        expression_reward_func,
        correctness_reward_func
    ],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
