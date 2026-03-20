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

def train_long(args):
    print("Starting Long Training (Phase E+)...")
    device = torch.device('cpu') # Use CPU for stability, or MPS if fast
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Device: {device}")

    env = Drone3DEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = ActorCritic(state_dim, action_dim, hidden_dim=128).to(device)
    buffer = RolloutBuffer()
    ppo_agent = PPO(policy, lr=3e-4, gamma=0.99, device=device)

    max_episodes = 10000
    update_interval = 2000
    global_step = 0
    
    # Load previous champion if exists to speed up?
    # policy.load_state_dict(torch.load('weights/champion_fixed_last.pt'))
    # Better to start fresh for a new task (obstacles).

    for ep in range(1, max_episodes + 1):
        state, _ = env.reset()
        ep_reward = 0
        
        for t in range(env.max_steps):
            global_step += 1
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action, log_prob, _ = policy.get_action(state_t)
            action_np = action.squeeze(0).cpu().detach().numpy()
            
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            
            buffer.add(state_t, action, log_prob, reward, done)
            state = next_state
            ep_reward += reward
            
            if global_step % update_interval == 0:
                loss = ppo_agent.update(buffer)
                buffer.clear()
                print(f"  [Update] Step {global_step} | Loss: {loss:.4f}")

            if done: break
        
        if ep % 10 == 0:
            print(f"Ep {ep} | Reward: {ep_reward:.2f} | Steps: {t}")
            
        if ep % 500 == 0:
            torch.save(policy.state_dict(), f"weights/champion_obstacles_{ep}.pt")

    torch.save(policy.state_dict(), "weights/champion_obstacles_final.pt")
    print("Done!")

if __name__ == "__main__":
    train_long(None)