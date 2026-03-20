import argparse
import os
import torch
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from envs.tunnel_env import TunnelEnv
from models.bio_policy import VisionPolicy  # Update import

def train(args):
    # Device setup (MPS friendly)
    if args.device == 'auto':
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Init
    env = TunnelEnv()
    
    # We now have 32 inputs (vision) instead of 2 (coordinates)
    input_dim = env.observation_space.shape[0]  # Should be 32
    output_dim = env.action_space.n
    
    # Switch to VisionPolicy
    policy = VisionPolicy(input_dim, output_dim, hidden_dim=64).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # Metrics
    episode_rewards = []

    for ep in range(args.episodes):
        state, _ = env.reset(seed=args.seed + ep)
        log_probs = []
        rewards = []
        
        while True:
            # NEW: Explicit float32
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            probs = policy(state_t)
            m = Categorical(probs)
            action = m.sample()
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            
            log_probs.append(m.log_prob(action))
            rewards.append(reward)
            state = next_state
            
            if terminated or truncated:
                break

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + args.gamma * G
            returns.insert(0, G)
        
        # FIX: Explicitly cast to float32 for MPS compatibility
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        
        # Whitening (baseline subtraction)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Policy Update
        loss = []
        for log_prob, G in zip(log_probs, returns):
            loss.append(-log_prob * G)
        
        optimizer.zero_grad()
        policy_loss = torch.stack(loss).sum()
        policy_loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        if (ep + 1) % 10 == 0:
            avg_rew = np.mean(episode_rewards[-10:])
            print(f"Episode {ep+1}/{args.episodes} | Avg Reward: {avg_rew:.2f} | Last Loss: {policy_loss.item():.4f}")

    # Save
    os.makedirs('weights', exist_ok=True)
    torch.save(policy.state_dict(), f"weights/{args.exp_name}.pt")
    print(f"Saved weights to weights/{args.exp_name}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--exp_name', type=str, default='baseline_v0')
    args = parser.parse_args()
    train(args)