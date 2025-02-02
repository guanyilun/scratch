import numpy as np
import sympy as sp
from scipy.optimize import curve_fit

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def optimize_params(formula_str, x_obs, y_obs):
    try:
        # Parse formula and extract parameters (C0, C1, ...)
        x_sym = sp.symbols('x')
        params = list(sp.ordered(formula_str.free_symbols - {x_sym}))
        params = [str(p) for p in params if str(p).startswith('C')]
        
        # Convert formula to a numerical function
        func = sp.lambdify([x_sym] + params, formula_str, 'numpy')
        
        # Initial guess for parameters (e.g., all 1s)
        p0 = np.ones(len(params))
        
        # Optimize using nonlinear least squares
        popt, _ = curve_fit(lambda x, *p: func(x, *p), x_obs, y_obs, p0=p0)
        
        # Compute MSE with optimized parameters
        y_pred = func(x_obs, *popt)
        mse = np.mean((y_pred - y_obs)**2)
        return mse, popt
    except:
        return float('inf'), None  # Invalid formula


def compute_reward(formula_str, x_obs, y_obs):
    mse, _ = optimize_params(formula_str, x_obs, y_obs)
    return 1 / (1 + mse)  # Reward inversely proportional to MSE


class PPOTrainer:
    def __init__(self, generator, value_model, tokenizer, lr=1e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, entropy_coef=0.01):
        self.generator = generator
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        
        # Optimizers
        self.generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(value_model.parameters(), lr=lr)
        
    def compute_advantages(self, rewards, values, dones):
        """Compute advantages using Generalized Advantage Estimation (GAE)."""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
    
    def train_step(self, x_obs, y_obs, batch_size=32):
        # 1. Generate rollouts
        rollouts = self.generate_rollouts(x_obs, y_obs, num_rollouts=batch_size)
        formulas, rewards = zip(*rollouts)
        
        # 2. Tokenize formulas and compute values
        tokenized_formulas = [torch.tensor(self.tokenizer.tokenize(f)) for f in formulas]
        padded_formulas = torch.nn.utils.rnn.pad_sequence(tokenized_formulas, batch_first=True, padding_value=0)
        with torch.no_grad():
            values = self.value_model(padded_formulas).squeeze()
        
        # 3. Compute advantages
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.zeros_like(rewards)  # Assume no episode termination
        advantages = self.compute_advantages(rewards, values, dones)
        
        # 4. Prepare data for training
        dataset = TensorDataset(padded_formulas, rewards, advantages, values)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 5. Update policy and value function
        for batch in dataloader:
            formula_tokens, batch_rewards, batch_advantages, batch_values = batch
            
            # Forward pass
            logits = self.generator(formula_tokens)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(formula_tokens)
            entropy = dist.entropy().mean()
            
            # Policy loss (clipped surrogate objective)
            ratios = torch.exp(log_probs - log_probs.detach())
            surr1 = ratios * batch_advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
            # Value loss (MSE between predicted and actual returns)
            returns = batch_advantages + batch_values
            value_loss = nn.functional.mse_loss(self.value_model(formula_tokens).squeeze(), returns)
            
            # Backpropagation
            self.generator_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            self.generator_optimizer.step()
            self.value_optimizer.step()
        
        return policy_loss.item(), value_loss.item()