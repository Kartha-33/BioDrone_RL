"""
train_simple.py — Train on simple tunnel first
"""
import os
import numpy as np
import torch
import torch.optim as optim

from env_simple import SimpleTunnelEnv
from model import FlyPolicyNetwork, get_device

EPISODES = 3000
GAMMA = 0.99
LR = 1e-3
SPARSITY = 0.8
PRINT_EVERY = 50
SAVE_PATH = "weights/fly_policy.pth"

def compute_returns(rewards, gamma):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def train():
    device = get_device()
    print("=" * 55)
    print("  BioDrone-RL  Simple Tunnel Training")
    print("=" * 55)
    
    env = SimpleTunnelEnv()
    model = FlyPolicyNetwork(
        input_dim=7, hidden_dim=32, output_dim=5, sparsity=SPARSITY
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    os.makedirs("weights", exist_ok=True)
    
    best_avg = -float('inf')
    history = []
    
    for ep in range(1, EPISODES + 1):
        obs, _ = env.reset()
        log_probs = []
        rewards = []
        done = False
        
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
            action, log_prob = model.get_action(obs_t)
            obs, reward, term, trunc, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            done = term or trunc
        
        returns = compute_returns(rewards, GAMMA).to(device)
        log_probs = torch.stack(log_probs)
        loss = -(log_probs * returns).sum()
        
        optimizer.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            model.fc1.weight.grad *= model.fc1.mask
            model.fc2.weight.grad *= model.fc2.mask
        
        optimizer.step()
        
        total_reward = sum(rewards)
        history.append(total_reward)
        
        if ep % PRINT_EVERY == 0:
            avg = np.mean(history[-100:])
            print(f"  Ep {ep:4d} | Reward: {total_reward:7.1f} | "
                  f"Avg100: {avg:7.1f} | Steps: {len(rewards)}")
            
            if avg > best_avg:
                best_avg = avg
                torch.save({
                    'model_state': model.state_dict(),
                    'sparsity': SPARSITY,
                    'best_avg_reward': best_avg,
                    'episode': ep,
                    'hyperparams': {
                        'input_dim': 7,
                        'hidden_dim': 32,
                        'output_dim': 5,
                    }
                }, SAVE_PATH)
                print(f"  ✅ Saved best (avg={best_avg:.1f})")
    
    print("\nTraining complete!")
    env.close()

if __name__ == "__main__":
    train()