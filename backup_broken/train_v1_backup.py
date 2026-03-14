"""
train.py — REINFORCE Policy Gradient Training Loop for BioDrone-RL

Trains the sparse FlyPolicyNetwork to navigate the BioDroneEnv tunnel
using the REINFORCE (Monte Carlo Policy Gradient) algorithm.

Algorithm:
    1. Run a full episode using the current policy
    2. Compute discounted returns G_t for each timestep
    3. Compute loss: -sum(log_prob(a_t) * G_t)
    4. Backpropagate and update only the active synapses
    5. Repeat for N episodes
"""

import os
import time
import numpy as np
import torch
import torch.optim as optim

from env import BioDroneEnv
from model import FlyPolicyNetwork, get_device


# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------

EPISODES        = 1000      # Total training episodes
GAMMA           = 0.99      # Discount factor for future rewards
LEARNING_RATE   = 1e-3      # Adam optimizer learning rate
SPARSITY        = 0.8       # Fraction of severed synapses
PRINT_EVERY     = 50        # Print progress every N episodes
SAVE_PATH       = "weights/fly_policy.pth"   # Where to save the brain


# ------------------------------------------------------------------
# Helper: Compute Discounted Returns
# ------------------------------------------------------------------

def compute_discounted_returns(rewards, gamma):
    """
    Convert a list of step rewards into discounted returns G_t.

    G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^{T-t}·r_T

    This tells the agent: 'the reward you got now is worth more
    than the same reward far in the future.'

    Args:
        rewards (list of float) : Raw rewards collected during episode
        gamma   (float)         : Discount factor (0 < gamma <= 1)

    Returns:
        returns (torch.Tensor)  : Normalized discounted returns
    """
    returns = []
    G = 0.0

    # Traverse rewards in reverse to accumulate future returns
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)

    # Normalize returns: zero mean, unit variance
    # This reduces variance in gradient estimates and stabilizes training
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


# ------------------------------------------------------------------
# Helper: Run One Episode
# ------------------------------------------------------------------

def run_episode(env, model, device):
    """
    Run a single episode using the current policy.
    Collect log probabilities and rewards at each step.

    Args:
        env    (BioDroneEnv)      : The tunnel environment
        model  (FlyPolicyNetwork) : The sparse brain
        device (torch.device)     : MPS or CPU

    Returns:
        log_probs    (list of Tensor) : Log prob of each action taken
        rewards      (list of float)  : Raw reward at each step
        total_reward (float)          : Sum of all rewards this episode
    """
    obs, _ = env.reset()
    log_probs    = []
    rewards      = []
    total_reward = 0.0

    done = False
    while not done:
        # Convert numpy observation to tensor and move to device
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

        # Get action from sparse brain
        action, log_prob = model.get_action(obs_tensor)

        # Step the environment
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store for REINFORCE update
        log_probs.append(log_prob)
        rewards.append(reward)
        total_reward += reward

    return log_probs, rewards, total_reward


# ------------------------------------------------------------------
# Helper: REINFORCE Update Step
# ------------------------------------------------------------------

def update_policy(optimizer, log_probs, returns, model):   # ← add model parameter
    """
    Perform one REINFORCE gradient update.

    REINFORCE Loss = -sum( log_prob(a_t) * G_t )

    Intuition:
      - If G_t is large (good outcome) → increase log_prob(a_t)
        (make that action more likely in similar states)
      - If G_t is small/negative (bad outcome) → decrease log_prob(a_t)
        (make that action less likely in similar states)

    The negative sign converts maximizing reward into minimizing loss
    (since PyTorch optimizers minimize by convention).

    Args:
        optimizer (torch.optim) : Adam optimizer
        log_probs (list)        : Log probs collected during episode
        returns   (Tensor)      : Normalized discounted returns
        model     (nn.Module)   : The policy network
    """
    policy_loss = []

    for log_prob, G_t in zip(log_probs, returns):
        policy_loss.append(-log_prob * G_t)

    total_loss = torch.stack(policy_loss).sum()

    optimizer.zero_grad()
    total_loss.backward()

    # Gradient clipping — now correctly references the passed model
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        max_norm=1.0
    )

    optimizer.step()

    return total_loss.item()


# ------------------------------------------------------------------
# Main Training Loop
# ------------------------------------------------------------------

def train():
    print("=" * 60)
    print("  BioDrone-RL | REINFORCE Training")
    print("=" * 60)

    # --- Setup ---
    device = get_device()
    env    = BioDroneEnv()
    model  = FlyPolicyNetwork(
        input_dim=5,
        hidden_dim=32,
        output_dim=3,
        sparsity=SPARSITY
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create weights directory if it doesn't exist
    os.makedirs("weights", exist_ok=True)

    # --- Print training config ---
    print(f"\n⚙️  Training Configuration:")
    print(f"   Episodes      : {EPISODES}")
    print(f"   Gamma         : {GAMMA}")
    print(f"   Learning Rate : {LEARNING_RATE}")
    print(f"   Sparsity      : {SPARSITY:.0%} synapses severed")
    print(f"   Device        : {device}")
    print(f"   Save Path     : {SAVE_PATH}")

    active_1 = int(model.fc1.mask.sum().item())
    active_2 = int(model.fc2.mask.sum().item())
    print(f"\n🧠 Connectome:")
    print(f"   Layer 1: {active_1}/160 active synapses")
    print(f"   Layer 2: {active_2}/96  active synapses")
    print()
    print(f"{'Episode':>10} | {'Reward':>10} | {'Avg(50)':>10} | {'Loss':>10} | {'Steps':>7}")
    print("-" * 60)

    # --- Tracking ---
    all_rewards  = []
    best_avg     = -float('inf')
    start_time   = time.time()

    # --- Training Episodes ---
    for episode in range(1, EPISODES + 1):

        # 1. Collect episode experience
        log_probs, rewards, total_reward = run_episode(env, model, device)

        # 2. Compute discounted returns
        returns = compute_discounted_returns(rewards, GAMMA)

        # Move returns to device to match log_prob tensors
        returns = returns.to(device)

        # 3. Update policy via REINFORCE
        loss = update_policy(optimizer, log_probs, returns, model)

        # 4. Track progress
        all_rewards.append(total_reward)
        steps = len(rewards)

        # 5. Print every PRINT_EVERY episodes
        if episode % PRINT_EVERY == 0:
            avg_reward = np.mean(all_rewards[-PRINT_EVERY:])
            elapsed    = time.time() - start_time

            print(
                f"{episode:>10} | "
                f"{total_reward:>10.1f} | "
                f"{avg_reward:>10.2f} | "
                f"{loss:>10.4f} | "
                f"{steps:>7}"
            )

            # Save best model
            if avg_reward > best_avg:
                best_avg = avg_reward
                torch.save(
                    {
                        'episode'        : episode,
                        'model_state'    : model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'best_avg_reward': best_avg,
                        'sparsity'       : SPARSITY,
                        'hyperparams'    : {
                            'gamma'        : GAMMA,
                            'lr'           : LEARNING_RATE,
                            'hidden_dim'   : 32,
                            'input_dim'    : 5,
                            'output_dim'   : 3,
                        }
                    },
                    SAVE_PATH
                )
                print(f"           💾 New best! Avg={best_avg:.2f} — weights saved")

    # --- Training Complete ---
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"\n✅ Training complete in {total_time:.1f}s")
    print(f"   Best average reward : {best_avg:.2f}")
    print(f"   Weights saved to    : {SAVE_PATH}")

    # --- Final Connectome Check ---
    # Verify sparsity mask was never corrupted during training
    active_1_post = int(model.fc1.mask.sum().item())
    active_2_post = int(model.fc2.mask.sum().item())
    print(f"\n🧠 Post-Training Connectome Integrity Check:")
    print(f"   Layer 1: {active_1_post}/160 active synapses "
          f"({'✅ intact' if active_1_post == active_1 else '❌ corrupted!'})")
    print(f"   Layer 2: {active_2_post}/96  active synapses "
          f"({'✅ intact' if active_2_post == active_2 else '❌ corrupted!'})")
    print()
    print("=" * 60)
    print("  Brain trained. Ready for Phase 4 — Visualization!")
    print("=" * 60)

    env.close()
    return model


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

if __name__ == "__main__":
    model = train()