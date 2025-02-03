import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.distributions

from train_model import rollout


class MathExpressionReward:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def check_parentheses(self, expression):
        """Verify parentheses are balanced and properly nested."""
        stack = []
        for char in expression:
            if char == '(':
                stack.append(char)
            elif char == ')':
                if not stack:
                    return False
                stack.pop()
        return len(stack) == 0
    
    def check_operator_placement(self, tokens):
        """Verify operators are properly placed between operands."""
        for i in range(len(tokens)):
            if tokens[i] in self.tokenizer.operators:
                # Check if operator is at start or end
                if i == 0 or i == len(tokens) - 1 or tokens[i+1] in self.tokenizer.operators:
                    return False
                # Check if operator is between two valid operands
                prev_valid = tokens[i-1] in self.tokenizer.numbers or tokens[i-1] == ')'
                next_valid = tokens[i+1] in self.tokenizer.numbers or tokens[i+1] == '('
                if not (prev_valid and next_valid):
                    return False
        return True
    
    def calculate_reward(self, expression_tokens):
        """Calculate reward based on multiple criteria."""
        try:
            # Base reward for a non-empty expression
            if not expression_tokens:
                return -1.0
                
            # Convert tokens to string for parentheses checking
            expr_str = ' '.join(expression_tokens)
            
            rewards = {
                'length': min(len(expression_tokens) / 10, 1.0),  # Reward longer valid expressions
                'parentheses': float(self.check_parentheses(expr_str)),
                'operators': float(self.check_operator_placement(expression_tokens)),
                'completion': float('<end>' in expression_tokens)  # Reward for proper completion
            }
            
            # Calculate final reward
            # Expression must be valid to get any positive reward
            if not all([rewards['parentheses'], rewards['operators']]):
                return -1.0
                
            return sum(rewards.values()) / len(rewards)
            
        except Exception as e:
            # Any error in evaluation results in negative reward
            return -1.0



class PPOTrainer:
    def __init__(self, model, tokenizer, reward_model, 
                 learning_rate=1e-5, epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        self.epsilon = epsilon  # PPO clipping parameter
        self.value_coef = value_coef  # Value loss coefficient
        self.entropy_coef = entropy_coef  # Entropy coefficient
        
        # Add value head to the model
        self.value_head = nn.Linear(model.d_model, 1).to(next(model.parameters()).device)
        
    def generate_experience(self, initial_prompt, num_rollouts=64):
        """Generate experience using current policy."""
        experiences = []
        
        for _ in range(num_rollouts):
            # Generate sequence
            with torch.no_grad():
                sequence = initial_prompt.copy()
                log_probs = []
                values = []
                
                for _ in range(self.model.max_len - len(sequence)):
                    # Get model predictions
                    inputs = torch.tensor(
                        self.tokenizer.encode(sequence), 
                        dtype=torch.long
                    ).unsqueeze(0).to(next(self.model.parameters()).device)
                    
                    outputs = self.model(inputs)
                    value = self.value_head(outputs[0, -1])
                    probs = F.softmax(outputs[0, -1], dim=-1)
                    
                    # Sample action
                    action_dist = torch.distributions.Categorical(probs)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)
                    
                    # Convert action to token and append
                    token = self.tokenizer.idx_to_token[action.item()]
                    sequence.append(token)
                    
                    log_probs.append(log_prob)
                    values.append(value)
                    
                    if token == '<end>':
                        break
                
                # Calculate reward
                reward = self.reward_model.calculate_reward(sequence)
                
                experiences.append({
                    'sequence': sequence,
                    'log_probs': torch.stack(log_probs),
                    'values': torch.stack(values),
                    'reward': reward
                })
        
        return experiences
    
    def train_step(self, experiences):
        """Perform one PPO update step."""
        # Prepare advantages and returns
        advantages = []
        returns = []
        
        for exp in experiences:
            # Calculate returns and advantages
            reward = exp['reward']
            values = exp['values']
            
            # Simple advantage estimation
            advantage = reward - values.detach()
            advantages.append(advantage)
            returns.append(torch.ones_like(values) * reward)
        
        advantages = torch.cat(advantages)
        returns = torch.cat(returns)
        old_log_probs = torch.cat([exp['log_probs'] for exp in experiences])
        
        # PPO update
        for _ in range(4):  # Multiple epochs of training
            for sequence, log_probs in zip(experiences, old_log_probs):
                # Get current policy predictions
                outputs = self.model(sequence)
                values = self.value_head(outputs)
                probs = F.softmax(outputs, dim=-1)
                
                # Calculate policy loss
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(sequence)
                
                ratio = torch.exp(new_log_probs - log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss = F.mse_loss(values.squeeze(-1), returns)
                
                # Calculate entropy bonus
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.value_coef * value_loss - 
                       self.entropy_coef * entropy)
                
                # Update model
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()


def train_with_rlhf(model, tokenizer, num_episodes=1000, episode_length=100):
    """Train the model using RLHF."""
    device = next(model.parameters()).device
    reward_model = MathExpressionReward(tokenizer)
    ppo_trainer = PPOTrainer(model, tokenizer, reward_model)
    
    # Training loop
    for episode in range(num_episodes):
        # Generate experience with initial prompts
        initial_prompts = [
            ['1'],  # Start with single number
            ['('],  # Start with opening parenthesis
            ['1', '+'],  # Start with partial expression
        ]
        
        all_experiences = []
        for prompt in initial_prompts:
            experiences = ppo_trainer.generate_experience(prompt)
            all_experiences.extend(experiences)
        
        # Train on collected experiences
        ppo_trainer.train_step(all_experiences)
        
        # Log training progress
        avg_reward = sum(exp['reward'] for exp in all_experiences) / len(all_experiences)
        print(f"Episode {episode}, Average Reward: {avg_reward:.4f}")
        
        # Sample and print some generated expressions
        if episode % 10 == 0:
            print("\nSample generations:")
            for prompt in initial_prompts:
                sequence = rollout(model, prompt, tokenizer, model.max_len, device)
                print(f"Prompt: {' '.join(prompt)}")
                print(f"Generated: {' '.join(sequence)}")
                print(f"Reward: {reward_model.calculate_reward(sequence):.4f}\n")