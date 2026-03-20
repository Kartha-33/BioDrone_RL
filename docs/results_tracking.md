# 📊 Experiment Results

| ID            | Phase | Architecture | Sparsity | Max Reward | Notes                                     |
| ------------- | ----- | ------------ | -------- | ---------- | ----------------------------------------- |
| `smoke_test`  | A     | Dense (2D)   | 0%       | 14.0       | Initial sanity check.                     |
| `vision_v0`   | B     | Vision (2D)  | 0%       | 305.1      | Proved vision-only flight possible.       |
| `ppo_tuned`   | C     | Dense (3D)   | 0%       | 855.8      | Optimized PPO for 3D flight.              |
| **BM_Dense**  | E     | Dense        | 0%       | 705.3      | High variance baseline.                   |
| **BM_Local**  | E     | Local-Bio    | 80%      | 698.9      | Too constrained? Lacks integration.       |
| **BM_Random** | E     | Random       | 80%      | 839.9      | **Current Champion.** Efficient & Robust. |
