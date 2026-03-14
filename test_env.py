"""
Quick sanity-check script for BioDroneEnv.
Runs the environment for a few episodes with random actions
and prints observations, rewards, and episode summaries.
"""
import numpy as np
from env import BioDroneEnv


def test_random_agent(num_episodes=3, max_steps=200):
    env = BioDroneEnv()
    print("=" * 55)
    print("  BioDrone-RL | Environment Sanity Check")
    print("=" * 55)
    print(f"Observation Space : {env.observation_space}")
    print(f"Action Space      : {env.action_space}")
    print(f"Observation Shape : {env.observation_space.shape}")
    print("=" * 55)

    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        total_reward = 0.0
        steps = 0
        reason = "MAX STEPS REACHED"   # ← default value added here

        print(f"\n--- Episode {episode + 1} ---")
        print(f"Initial Observation: {obs}")

        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # Print first 5 steps in detail
            if step < 5:
                action_names = ["Left", "Straight", "Right"]
                print(
                    f"  Step {step + 1:>3} | "
                    f"Action: {action_names[action]:<8} | "
                    f"Reward: {reward:>7.1f} | "
                    f"Obs: {np.round(obs, 3)}"
                )

            if terminated or truncated:
                reason = "CRASHED" if terminated else "TRUNCATED"
                break

        print(f"  ...")
        print(f"  Episode ended: {reason} after {steps} steps")
        print(f"  Total Reward : {total_reward:.1f}")

    env.close()
    print("\n" + "=" * 55)
    print("  Environment check complete. Ready for Phase 2!")
    print("=" * 55)


if __name__ == "__main__":
    test_random_agent()