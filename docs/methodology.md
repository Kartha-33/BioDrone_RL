# 📐 Methodology

## 1. Environment: Pseudo-3D Tunnel

- **Physics**: 6-DOF drone dynamics simplified to 4 control inputs (Thrust, Roll, Pitch, Yaw).
- **Vision**: Simulated "Fly Eye" (Compound Vision).
  - 2 Eyes x 32 Photoreceptors (Total 64 pixels).
  - Raycasting generates depth-based intensity maps.
- **Parameters**:
  - Tunnel Width: 6.0m.
  - Gravity: -2.0 m/s² (Low gravity sim).
  - Damping: Linear velocity drag.

## 2. Neural Architecture: Bio-Constrained Policy

We utilize a masked PPO Actor-Critic network.

- **Input**: 70-dim (64 Vision + 6 IMU).
- **Hidden**: 128 units.
- **Constraint Mechanism**: `MaskedLinear` layer injects fixed adjacency matrices.
  - **Random**: Erdos-Renyi graph (p=0.2).
  - **Local**: Retinotopic connectivity (only neighbors).
  - **Small-World**: Local + 5% random long-range shortcuts (Watts-Strogatz).

## 3. Training Algorithm

- **PPO (Proximal Policy Optimization)**:
  - Clip Range: 0.2
  - Update Frequency: 1000 steps.
  - Learning Rate: 1e-4.
  - Entropy Coefficient: 0.01.
