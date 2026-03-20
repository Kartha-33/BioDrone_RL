# 📝 Paper Title: Wiring Matters: Topological Constraints in Drone Navigation

## Abstract

We demonstrate that 80% sparse neural networks outperform fully connected networks in a visual navigation task. Specifically, random small-world topologies achieve 20% higher reward than dense baselines.

## 1. Introduction

- Drones require efficient onboard compute.
- Insects navigate with sparse, structured brains.
- We test if bio-mimetic sparsity improves RL.

## 2. Methods

- **The Pseudo-3D Eye**: Simulating compound vision.
- **Masked PPO**: Injecting adjacency matrices into Actor-Critic.

## 3. Results (Key Findings)

- **Finding 1**: Sparse networks learn faster and more stably than dense ones.
- **Finding 2**: "Small-World" randomness beats strict local connectivity (Retinotopy). Global shortcuts are essential for sensor fusion.

## 4. Discussion

- The "Lottery Ticket Hypothesis" explanation.
- Future work: Using the full _Drosophila_ connectome data.
