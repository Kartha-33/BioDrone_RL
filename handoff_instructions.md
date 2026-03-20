# 🔄 BioDrone-RL Session Handoff

## 📅 Status as of [2026-03-16]

**Project:** Bio-Constrained Drone Navigation
**Goal:** Prove that Fly-Brain topology beats Dense networks in 3D flight.

## ✅ Completed Modules

1.  **Env**: `src/envs/drone_3d_env.py` (Custom 3D corridor).
    - _Status:_ **Phase F Upgraded:** Supports procedural obstacles (Pillars/Floating Blocks) and Raycasting.
2.  **Vision**: `src/models/vision.py` (1D Compound Eye simulation).
    - _Status:_ **Upgraded:** Now performs raycasting against bounding boxes for true Depth Perception.
3.  **Policy**: `src/models/bio_policy.py` (`ActorCritic` with `MaskedLinear`).
    - _Status:_ Stable.
4.  **Training Script**: `scripts/train_obstacles.py`.
    - _Status:_ Validated with a 10,000 episode run.

## 🚧 Work In Progress (Phase F: The Obstacle Course)

We successfully transition to Phase F. The drone learned to navigate a forest of pillars with high rewards (~6000), but we discovered a behavior exploit.

- **The Issue:** The drone learned to "cheat" by hugging the left wall, as the current obstacle generation leaves a safe gap near the edges.
- **Current Task:** We need to implement "Hard Mode" in the environment to force active maneuvering (Barriers, full-width spawning).
- **Latest Checkpoint:** `weights/champion_obstacles_final.pt` (The "Left-Hugger").

## 📂 Key Files

- `scripts/train_obstacles.py`: The marathon training script (10k eps).
- `scripts/evaluate.py`: Updated visualizer that renders obstacles as red boxes.
- `src/models/vision.py`: The new Raycasting engine.

## ⏭️ Next Actions for the AI Agent

1.  **Refine Environment:** Update `src/envs/drone_3d_env.py` to include "Hard Mode" logic (Barriers, Wall-to-Wall pillars) to prevent wall-hugging.
2.  **Retrain:** Run `python scripts/train_obstacles.py` again on the harder environment.
3.  **Verify:** Run `python scripts/evaluate.py` to confirm the drone is actually dodging (going around/under) rather than finding loopholes.
