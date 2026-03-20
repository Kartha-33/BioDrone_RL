import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPO:
    """
    Minimalist PPO implementation for continuous control.
    """
    def __init__(self, policy, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=10, device='cpu'):
        self.policy = policy.to(device)
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        
        self.mse_loss = nn.MSELoss()

    def update(self, memory):
        # Convert list of transitions to tensors
        
        # Stack creates [Batch, 1, Dim]. Squeeze(1) makes it [Batch, Dim]
        states = torch.stack(memory.states).to(self.device).squeeze(1).detach()
        actions = torch.stack(memory.actions).to(self.device).squeeze(1).detach()
        old_log_probs = torch.stack(memory.log_probs).to(self.device).squeeze(1).detach()
        
        rewards = memory.rewards
        is_terminals = memory.is_terminals
        
        # 1. Monte Carlo Estimate of Returns & Advantages (GAE could be added here)
        # Simple Discounted Reward Calculation
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        # Normalize returns
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            log_probs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            
            # Match tensor dimensions
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(log_probs - old_log_probs)

            # Finding Surrogate Loss
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # Final loss = -min(surr1, surr2) + 0.5*MSE(val, ret) - 0.01*entropy
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, returns) - 0.01 * dist_entropy

            # Gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        return loss.mean().item()

# Simple container for trajectory data
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]
    
    def add(self, state, action, log_prob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.is_terminals.append(done)