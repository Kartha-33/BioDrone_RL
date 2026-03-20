import argparse
import os
import torch
import numpy as np
import pandas as pd
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from envs.drone_3d_env import Drone3DEnv
from models.bio_policy import ActorCritic
from utils.ppo import PPO, RolloutBuffer
from utils.connectome import generate_bio_mask

def train_and_log(mask_type, density, seed, args):
    print(f"\n--- Starting Experiment: {mask_type} (Density: {density}, Seed: {seed}) ---")
    
    # Seeding
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Init Env & device
    device = torch.device(args.device) if args.device != 'auto' else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    env = Drone3DEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Init Policy with correct mask
    policy = ActorCritic(state_dim, action_dim, hidden_dim=128).to(device)
    
    # Inject Mask if needed
    if mask_type != 'dense':
        # Apply mask to the first layer of feature extractor
        # Feature[0] is MaskedLinear(state_dim, 128)
        mask = generate_bio_mask(state_dim, 128, connection_type=mask_type, density=density)
        policy.features[0].mask.data = mask.to(device)
        policy.features[0].weight.data *= mask.to(device) # Prune initial weights
        print(f"Applied {mask_type} mask with {mask.sum().item()} connections.")
    else:
        # For 'dense', we make the mask all 1s (effectively disabling sparsity)
        policy.features[0].mask.data.fill_(1.0)
        print("Applied Dense mask (all connections active).")

    # Init PPO
    buffer = RolloutBuffer()
    ppo_agent = PPO(policy, lr=args.lr, gamma=args.gamma, device=device)
    
    # Logs
    history = []
    global_step = 0
    
    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        ep_reward = 0
        
        for t in range(500): # Max steps hardcoded
            global_step += 1
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action, log_prob, _ = policy.get_action(state_tensor)
            action_np = action.squeeze(0).cpu().detach().numpy()
            
            next_state, reward, done, trunc, _ = env.step(action_np)
            buffer.add(state_tensor, action, log_prob, reward, done or trunc)
            
            state = next_state
            ep_reward += reward
            
            if global_step % args.update_timesteps == 0:
                ppo_agent.update(buffer)
                buffer.clear()
            
            if done or trunc:
                break
        
        # Log every 10 episodes
        if ep % 10 == 0:
            history.append({
                'episode': ep,
                'reward': ep_reward,
                'model': mask_type,
                'seed': seed
            })
            print(f"Ep {ep} | Reward: {ep_reward:.2f}")

    return pd.DataFrame(history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000) # Short run for demo
    parser.add_argument('--update_timesteps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seeds', type=int, default=1) # How many runs per model
    args = parser.parse_args()

    # Determine Device
    print(f"Running on {args.device}")

    all_results = []
    
    # 1. Dense Baseline (Control)
    for s in range(args.seeds):
        df = train_and_log('dense', 1.0, 42+s, args)
        all_results.append(df)

    # 2. Random Sparse (Control 2)
    for s in range(args.seeds):
        df = train_and_log('random', 0.2, 42+s, args)
        all_results.append(df)

    # 3. Bio-Constrained (Retinotopic)
    for s in range(args.seeds):
        df = train_and_log('local', 0.2, 42+s, args)
        all_results.append(df)

    # 4. Small-World Bio (The Upgrade)
    for s in range(args.seeds):
        df = train_and_log('small_world', 0.2, 42+s, args)
        all_results.append(df)

    # Save
    final_df = pd.concat(all_results)
    os.makedirs('results', exist_ok=True)
    final_df.to_csv('results/benchmark_phase_E.csv', index=False)
    print("Benchmark Complete! Saved to results/benchmark_phase_E.csv")