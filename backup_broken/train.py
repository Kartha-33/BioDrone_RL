"""
train.py — REINFORCE Training Loop for BioDrone-RL v2
Trains on env_v2: 7 sensors, 5 actions, gates, vertical movement
"""

import os
import time
import numpy as np
import torch
import torch.optim as optim

from env_v2 import BioDroneEnv2
from model import FlyPolicyNetwork, get_device

# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------
EPISODES        = 5000
GAMMA           = 0.99
LEARNING_RATE   = 1e-3
SPARSITY        = 0.8
PRINT_EVERY     = 50
SAVE_PATH       = "weights/fly_policy.pth"
SAVE_EVERY      = 200       # Save checkpoint every N episodes

INPUT_DIM       = 7         # 7 sensors (FL, NL, FW, NR, FR, UP, DN)
HIDDEN_DIM      = 32
OUTPUT_DIM      = 5         # 5 actions (LEFT, FWD, RIGHT, UP, DOWN)

# ------------------------------------------------------------------
# Helper: Compute Discounted Returns
# ------------------------------------------------------------------
def compute_discounted_returns(rewards, gamma):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    # Normalise for stability
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

# ------------------------------------------------------------------
# Main Training Loop
# ------------------------------------------------------------------
def train():
    device = get_device()
    print("=" * 55)
    print("  BioDrone-RL  Training v2  (env_v2 + 7 sensors)")
    print("=" * 55)
    print(f"  Device     : {device}")
    print(f"  Episodes   : {EPISODES}")
    print(f"  Input dim  : {INPUT_DIM}")
    print(f"  Actions    : {OUTPUT_DIM}")
    print(f"  Sparsity   : {SPARSITY}")
    print("=" * 55)

    env   = BioDroneEnv2()
    model = FlyPolicyNetwork(
        input_dim  = INPUT_DIM,
        hidden_dim = HIDDEN_DIM,
        output_dim = OUTPUT_DIM,
        sparsity   = SPARSITY
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs("weights", exist_ok=True)

    best_avg_reward = -float('inf')
    reward_history  = []

    for ep in range(1, EPISODES + 1):
        obs, _ = env.reset()
        log_probs = []
        rewards   = []
        done      = False

        # --- Run one full episode ---
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
            action, log_prob = model.get_action(obs_t)
            obs, reward, terminated, truncated, info = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            done = terminated or truncated

        # --- Compute returns and loss ---
        returns   = compute_discounted_returns(rewards, GAMMA).to(device)
        log_probs = torch.stack(log_probs)
        loss      = -(log_probs * returns).sum()

        # --- Update only active synapses ---
        optimizer.zero_grad()
        loss.backward()

        # Zero out gradients of pruned synapses
        with torch.no_grad():
            model.fc1.weight.grad *= model.fc1.mask
            model.fc2.weight.grad *= model.fc2.mask

        optimizer.step()

        # --- Track progress ---
        total_reward   = sum(rewards)
        gates_passed   = info.get("gates_passed", 0)
        reward_history.append(total_reward)

        if ep % PRINT_EVERY == 0:
            avg = np.mean(reward_history[-100:])
            print(f"  Ep {ep:5d} | "
                  f"Reward: {total_reward:8.1f} | "
                  f"Avg100: {avg:8.1f} | "
                  f"Gates: {gates_passed} | "
                  f"Steps: {len(rewards)}")

            # Save best model
            if avg > best_avg_reward:
                best_avg_reward = avg
                torch.save({
                    'model_state'     : model.state_dict(),
                    'sparsity'        : SPARSITY,
                    'best_avg_reward' : best_avg_reward,
                    'episode'         : ep,
                    'hyperparams'     : {
                        'input_dim'  : INPUT_DIM,
                        'hidden_dim' : HIDDEN_DIM,
                        'output_dim' : OUTPUT_DIM,
                    }
                }, SAVE_PATH)
                print(f"  ✅ New best! Saved → {SAVE_PATH}  "
                      f"(avg={best_avg_reward:.1f})")

        # Periodic save regardless of best
        if ep % SAVE_EVERY == 0:
            torch.save({
                'model_state'     : model.state_dict(),
                'sparsity'        : SPARSITY,
                'best_avg_reward' : best_avg_reward,
                'episode'         : ep,
                'hyperparams'     : {
                    'input_dim'  : INPUT_DIM,
                    'hidden_dim' : HIDDEN_DIM,
                    'output_dim' : OUTPUT_DIM,
                }
            }, SAVE_PATH + f".ep{ep}.bak")

    print("\n" + "=" * 55)
    print(f"  Training complete!")
    print(f"  Best avg reward : {best_avg_reward:.1f}")
    print(f"  Weights saved   : {SAVE_PATH}")
    print("=" * 55)
    env.close()

if __name__ == "__main__":
    train()