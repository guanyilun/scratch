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


class GRPOTrainer:
    def __init__(self, model, tokenizer, reward_model, 
                 learning_rate=1e-5, temperature=1.0, rank_epsilon=0.05):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        self.temperature = temperature  # Temperature for reward scaling
        self.rank_epsilon = rank_epsilon  # Tolerance for rank comparisons
        
    def compute_rank_weights(self, rewards):
        """Compute importance weights based on reward ranking."""
        # Sort rewards and get indices
        sorted_indices = torch.argsort(rewards)
        ranks = torch.zeros_like(rewards)
        ranks[sorted_indices] = torch.arange(len(rewards), dtype=torch.float32)
        
        # Convert ranks to weights using soft rank transformation
        weights = torch.exp(ranks / self.temperature)
        weights = weights / weights.sum()  # Normalize weights
        
        return weights
        
    def generate_experience(self, initial_prompt, num_rollouts=64):
        """Generate experience using current policy."""
        experiences = []
        
        for _ in range(num_rollouts):
            with torch.no_grad():
                sequence = initial_prompt.copy()
                log_probs = []
                
                # Generate sequence using current policy
                for _ in range(self.model.max_len - len(sequence)):
                    inputs = torch.tensor(
                        self.tokenizer.encode(sequence), 
                        dtype=torch.long
                    ).unsqueeze(0).to(next(self.model.parameters()).device)
                    
                    outputs = self.model(inputs)
                    probs = F.softmax(outputs[0, -1], dim=-1)
                    
                    # Sample action and compute log probability
                    action_dist = torch.distributions.Categorical(probs)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)
                    
                    token = self.tokenizer.idx_to_token[action.item()]
                    sequence.append(token)
                    log_probs.append(log_prob)
                    
                    if token == '<end>':
                        break
                
                # Calculate reward for the sequence
                reward = self.reward_model.calculate_reward(sequence)
                
                experiences.append({
                    'sequence': sequence,
                    'log_probs': torch.stack(log_probs),
                    'reward': reward
                })
        
        return experiences
    
    def train_step(self, experiences):
        """Perform one GRPO update step."""
        # Collect rewards and compute rank-based weights
        rewards = torch.tensor([exp['reward'] for exp in experiences])
        weights = self.compute_rank_weights(rewards)
        
        # Compute weighted policy gradient
        policy_loss = 0
        entropy_loss = 0
        
        for exp, weight in zip(experiences, weights):
            # Get current policy predictions
            outputs = self.model(torch.tensor(
                self.tokenizer.encode(exp['sequence']),
                dtype=torch.long
            ).unsqueeze(0).to(next(self.model.parameters()).device))
            
            probs = F.softmax(outputs, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            # Compute weighted negative log-likelihood loss
            policy_loss -= weight * exp['log_probs'].sum()
            
            # Add entropy regularization
            entropy_loss -= weight * dist.entropy().mean()
        
        # Total loss with entropy regularization
        total_loss = policy_loss + 0.01 * entropy_loss
        
        # Update model
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return total_loss.item()

def train_with_grpo(model, tokenizer, num_episodes=1000, episode_length=100):
    """Train the model using GRPO."""
    device = next(model.parameters()).device
    reward_model = MathExpressionReward(tokenizer)
    grpo_trainer = GRPOTrainer(model, tokenizer, reward_model)
    
    for episode in range(num_episodes):
        # Generate experience with various initial prompts
        initial_prompts = [
            ['1'],
            ['('],
            ['1', '+'],
        ]
        
        all_experiences = []
        for prompt in initial_prompts:
            experiences = grpo_trainer.generate_experience(prompt)
            all_experiences.extend(experiences)
        
        # Train using GRPO
        loss = grpo_trainer.train_step(all_experiences)
        
        # Calculate and log metrics
        avg_reward = sum(exp['reward'] for exp in all_experiences) / len(all_experiences)
        print(f"Episode {episode}, Loss: {loss:.4f}, Average Reward: {avg_reward:.4f}")
        
        # Sample and print generations periodically
        if episode % 10 == 0:
            print("\nSample generations:")
            for prompt in initial_prompts:
                sequence = rollout(model, prompt, tokenizer, model.max_len, device)
                print(f"Prompt: {' '.join(prompt)}")
                print(f"Generated: {' '.join(sequence)}")
                print(f"Reward: {reward_model.calculate_reward(sequence):.4f}\n")