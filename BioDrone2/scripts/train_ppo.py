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

def train_ppo(args):
    # Device setup
    if args.device == 'auto':
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Init Env
    env = Drone3DEnv()
    state_dim = env.observation_space.shape[0]  # e.g., 70
    action_dim = env.action_space.shape[0]      # e.g., 4 (roll, pitch, yaw, thrust)

    # Init PPO
    policy = ActorCritic(state_dim, action_dim, hidden_dim=128).to(device)
    buffer = RolloutBuffer()
    ppo_agent = PPO(policy, lr=args.lr, gamma=args.gamma, device=device)

    # Training Loop
    time_step = 0
    max_ep_steps = env.max_steps
    update_timestep = args.update_timesteps # Update every 2000 steps

    global_step = 0

    for ep in range(1, args.max_episodes + 1):
        state, _ = env.reset()
        current_ep_reward = 0
        
        for t in range(1, max_ep_steps+1):
            global_step += 1
            
            # 1. State to Tensor [1, StateDim]
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 2. Action
            action, log_prob, _ = policy.get_action(state_tensor)
            action_np = action.squeeze(0).cpu().detach().numpy()
            
            # 3. Step
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            # 4. Store [1, StateDim], [1, ActionDim]
            buffer.add(state_tensor, action, log_prob, reward, done)
            
            state = next_state
            current_ep_reward += reward

            # 5. UPDATE PPO
            # Update every 'update_timesteps' (e.g. 2000 steps)
            if global_step % args.update_timesteps == 0:
                loss = ppo_agent.update(buffer)
                buffer.clear()
                print(f"  [PPO Update] Step {global_step} | Loss: {loss:.4f}")

            if done:
                break
        
        # Logging
        if ep % 10 == 0:
            print(f"Episode {ep} | Reward: {current_ep_reward:.2f}")
        
        # Save Checkpoint
        if ep % 100 == 0:
            os.makedirs('weights', exist_ok=True)
            torch.save(policy.state_dict(), f"weights/{args.exp_name}_last.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episodes', type=int, default=1000)
    parser.add_argument('--update_timesteps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--exp_name', type=str, default='ppo_v0')
    args = parser.parse_args()
    train_ppo(args)