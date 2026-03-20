import argparse
import os
import torch
import numpy as np
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from envs.drone_3d_env import Drone3DEnv
from models.bio_policy import ActorCritic
from utils.ppo import PPO, RolloutBuffer

def train_long():
    print("Starting Phase F: Obstacle Course (10k Episodes)...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using Device: {device}")

    env = Drone3DEnv()
    policy = ActorCritic(70, 4, hidden_dim=128).to(device)
    buffer = RolloutBuffer()
    ppo_agent = PPO(policy, lr=3e-4, gamma=0.99, device=device)

    global_step = 0
    
    # Run for 10,000 Episodes
    for ep in range(1, 10001):
        state, _ = env.reset()
        ep_reward = 0
        
        for t in range(env.max_steps):
            global_step += 1
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action, log_prob, _ = policy.get_action(state_t)
            action_np = action.squeeze(0).cpu().detach().numpy()
            
            next_state, reward, done, trunc, _ = env.step(action_np)
            
            buffer.add(state_t, action, log_prob, reward, (done or trunc))
            state = next_state
            ep_reward += reward
            
            if global_step % 2000 == 0:
                loss = ppo_agent.update(buffer)
                buffer.clear()
                print(f"  [Step {global_step}] Loss: {loss:.4f}")

            if done or trunc: break
        
        if ep % 20 == 0:
            print(f"Episode {ep} | Reward: {ep_reward:.2f}")
            
        if ep % 500 == 0:
            torch.save(policy.state_dict(), f"weights/champion_obstacles_{ep}.pt")

    torch.save(policy.state_dict(), "weights/champion_obstacles_final.pt")
    print("Training Complete!")

if __name__ == "__main__":
    train_long()