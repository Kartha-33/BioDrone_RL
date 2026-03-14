"""
test_policy.py — Watch the trained policy in action (text only)
"""
import torch
from env_gates import GateTunnelEnv
from model import FlyPolicyNetwork, get_device

device = get_device()
env = GateTunnelEnv()

# Load model
ck = torch.load("weights/fly_policy.pth", map_location=device, weights_only=False)
model = FlyPolicyNetwork(7, 32, 5, ck['sparsity']).to(device)
model.load_state_dict(ck['model_state'])
model.eval()

print("=" * 60)
print("  Testing trained policy (5 episodes)")
print("=" * 60)

for ep in range(5):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step = 0
    
    while not done and step < 500:
        obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            action, _ = model.get_action(obs_t)
        
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        step += 1
        done = term or trunc
        
        if step % 20 == 0:
            gates_passed = info.get('gates_passed', 0)
            print(f"  Ep {ep+1} Step {step:3d}: reward={total_reward:7.1f} "
                  f"gates={gates_passed} drone=({env.drone_x:.1f},{env.drone_y:.1f})")
    
    gates = info.get('gates_passed', 0)
    print(f"  Ep {ep+1} DONE: reward={total_reward:.1f} gates={gates} steps={step}\n")

env.close()