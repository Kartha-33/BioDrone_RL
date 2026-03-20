# 🪰 BioDrone-RL: Bio-Constrained Drone Navigation

> **Does wiring a drone's neural network like a fruit fly's visual system make it fly better, safer, and more robustly than standard AI?**

BioDrone-RL is a Reinforcement Learning research project investigating the impact of biological connectome constraints on artificial agents. We replace standard "Dense" neural networks with sparse, structured architectures mimicking the motion-detection circuits of *Drosophila* (fruit flies) to navigate complex 3D environments.

---

## 📺 Flight Demonstration

The following video showcases the agent navigating the "Hard Mode" obstacle course (Phase F) using only simulated low-res visual input (64 pixels).

<video src="phase_f_final.mp4" controls="controls" style="max-width: 100%;">
  Your browser does not support the video tag.
</video>

*(Note: If the video does not load, please ensure `phase_f_final.mp4` is in the root directory.)*

---

## 🏗️ System Architecture

The system mimics a biological sensorimotor loop. The drone "sees" the world through a simulated compound eye, processes the visual data using a bio-constrained neural network, and outputs motor commands.

**Mermaid Architecture Diagram:**

graph TD
    subgraph Environment ["Drone3DEnv (Physics)"]
        World[3D Obstacle Course]
        Drone[Drone Dynamics 6-DOF]
    end

    subgraph Sensors ["Sensory System"]
        Eye[FlyRetina (Raycasting)]
        IMU[IMU Sensor]
    end

    subgraph Agent ["Bio-Constrained Brain"]
        VisualProcessing[Masked Visual Layers]
        Integration[Small-World Integration]
        Motor[Motor Command Output]
    end
    
    World -->|Geometry| Eye
    Drone -->|State| IMU
    
    Eye -->|Depth/Intensity (64px)| VisualProcessing
    IMU -->|Orientation/Vel| Integration
    
    VisualProcessing --> Integration
    Integration -->|PPO Policy| Motor
    
    Motor -->|Thrust, Roll, Pitch, Yaw| Drone

### Key Components

1. **Environment (Drone3DEnv):** A custom Gymnasium environment simulating a 3D tunnel with procedural obstacles (Pillars, Bars, Walls). It uses a simplified 4-channel control scheme.

2. **Vision (FlyRetina):** Simulates a 1D compound eye with 2 eyes x 32 photoreceptors. It uses optimized raycasting to generate depth-based intensity maps, mimicking the "Optic Flow" integration of flies.

3. **Policy (BioPolicy):** An Actor-Critic network using MaskedLinear layers. Instead of fully connected layers, it enforces specific sparsity patterns:
   - Local: Retinotopic connectivity (neurons only talk to neighbors).
   - Small-World: Local + ~5% random long-range shortcuts (Watts-Strogatz), balancing specialization and integration.

---

## 🗺️ Project Roadmap & Status

We are currently in **Phase F**, focusing on complex obstacle avoidance and adversarial geometric challenges.

**Project Timeline:**

Phase A: Scale Test (2D Tunnel) - COMPLETE (2026-03-14)
Phase B: Vision System (Retina) - COMPLETE (2026-03-14)
Phase C: 3D Physics (PPO Pilot) - COMPLETE (2026-03-15)
Phase E: The Benchmark (Architectures) - COMPLETE (2026-03-15)
Phase F: Obstacle Course (Hard Mode) - IN PROGRESS (2026-03-16)
Phase G: Real Connectome Data - PLANNED
Phase H: Sim-to-Real Transfer - PLANNED

| Phase | Description | Status |
| :--- | :--- | :--- |
| **A** | **Skeleton:** Basic RL loop validation. | ✅ Complete |
| **B** | **Vision:** Raycasting based "Fly Eye" sensor. | ✅ Complete |
| **C** | **Physics:** 3D continuous control with PPO. | ✅ Complete |
| **E** | **Benchmark:** Comparing Dense vs. Bio vs. Random sparse nets. | ✅ Complete |
| **F** | **Obstacles:** Procedural generation of difficult geometry. | 🔄 **Active** |

---

## 📂 Repository Structure

BioDrone2/
├── BioDrone2/
│   ├── scripts/            # Training and evaluation entry points
│   │   ├── train_obstacles.py  # Main training loop for Phase F
│   │   ├── evaluate.py         # Visualizer for trained models
│   │   └── benchmark_phases.py # Script for running architectural comparisons
│   ├── src/                # Core library code
│   │   ├── envs/               # Gymnasium Environment logic
│   │   │   └── drone_3d_env.py # The 3D world and physics
│   │   ├── models/             # Neural Network definitions
│   │   │   ├── vision.py       # Raycasting engine
│   │   │   └── bio_policy.py   # Masked PPO Actor-Critic
│   ├── weights/            # Saved model checkpoints (*.pt)
│   └── results/            # CSV logs and performance plots
└── docs/                   # Research logs and methodology
    ├── methodology.md
    └── research_log.md

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Gymnasium
- NumPy, Matplotlib

### Installation

git clone https://github.com/yourusername/BioDrone2.git
cd BioDrone2
pip install torch gymnasium numpy matplotlib

### Usage

**1. Train a new agent in the Obstacle Environment:**

python BioDrone2/scripts/train_obstacles.py

This will train a PPO agent for 10,000 episodes and save checkpoints to BioDrone2/weights/.

**2. Evaluate a trained model:**

python BioDrone2/scripts/evaluate.py --model BioDrone2/weights/champion_obstacles_final.pt

**3. Run Benchmarks:**

python BioDrone2/scripts/benchmark_phases.py

---

## 📊 Results Summary

Our Phase E benchmarks provided surprising results regarding network sparsity:

- **Dense Networks:** High variance, prone to overfitting specific trajectories.
- **Random Sparse (20%):** Best overall performance, suggesting sparsity acts as a powerful regularizer for simple flight tasks.
- **Small-World (Bio-Mimetic):** Extremely competitive with Random Sparse but with potentially better generalization for complex maneuvers (currently being tested in Phase F).

*(See BioDrone2/results/benchmark_phase_E.csv for raw data)*

---

## 🔄 Current Work (Phase F)

### The Challenge

The drone learned to navigate a forest of pillars with high rewards (~6000), but discovered a behavior exploit: it learned to "cheat" by hugging the left wall. The current obstacle generation leaves a safe gap near the edges, allowing the drone to avoid active maneuvering.

### Next Steps

1. **Refine Environment:** Update src/envs/drone_3d_env.py to include "Hard Mode" logic (Barriers, Wall-to-Wall pillars) to prevent wall-hugging.
2. **Retrain:** Run the training script again on the harder environment.
3. **Verify:** Confirm the drone is actually dodging (going around/under obstacles) rather than finding loopholes.

---

## 📚 Technical Details

### Environment Dynamics

- **Tunnel Width:** 6.0m
- **Gravity:** -2.0 m/s² (Low gravity simulation)
- **Damping:** Linear velocity drag for stability
- **Max Steps:** 1000 per episode
- **Control Space:** 4 channels (Thrust, Roll, Pitch, Yaw)

### Vision System

- **Input:** 2 compound eyes, 32 photoreceptors each (64 pixels total)
- **FOV:** 100 degrees per eye
- **Max Vision Range:** 10.0 meters
- **Raycasting:** Against tunnel walls and procedural obstacles

### Training Algorithm

- **Algorithm:** Proximal Policy Optimization (PPO)
- **Update Frequency:** 1000 steps
- **Learning Rate:** 3e-4
- **Gamma (Discount Factor):** 0.99
- **Entropy Coefficient:** 0.01
- **Clip Range:** 0.2

---

## 🧪 Experimental Design

### Architecture Comparison (Phase E)

We trained four network architectures to compare their effectiveness:

1. **Dense:** Fully connected layers with no sparsity constraints.
2. **Random Sparse:** Erdos-Renyi random graph with 20% connection density.
3. **Local (Bio-Mimetic):** Retinotopic connectivity, only local neighbors.
4. **Small-World (Bio-Inspired):** Local connectivity + 5% random long-range shortcuts (Watts-Strogatz topology).

### Key Finding

Random sparsity outperformed dense networks, suggesting that sparsity acts as a powerful regularizer. Small-World topology showed competitive performance with excellent generalization potential.

---

## 🎯 Research Questions

1. Does biological constraint improve robustness in noisy environments?
2. Can sparse networks generalize better to unseen obstacle configurations?
3. How does connectome-inspired architecture compare to evolved dense networks?
4. Can we transfer learned policies from simulation to real hardware?

---

## 📖 References

- **Environment:** Gymnasium (formerly OpenAI Gym)
- **RL Algorithm:** Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **Vision Model:** Inspired by Drosophila motion detection circuits (T4/T5 neurons)
- **Connectome Data:** FlyWire and Hemibrain datasets
- **Network Topology:** Watts-Strogatz small-world networks

---

## ⚠️ Known Issues

1. **Wall-Hugging Exploit:** The drone learns to exploit edge gaps in obstacle generation.
2. **Phase F In Progress:** Hard Mode implementation needed to force genuine obstacle avoidance.
3. **Generalization:** Current models may overfit to specific obstacle patterns.

---

## 🤝 Contributing

This is an active research project. For questions or contributions, please open an issue or pull request.

---

## 📄 License

MIT License - See LICENSE file for details.

---

**Author:** Achuth Kartha
**Last Updated:** 2026-03-20
**Project Status:** Active Research