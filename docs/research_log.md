# 🧪 BioDrone-RL Research Log

## [2026-03-14] Phase A: Baseline & Smoke Tests

**Goal**: Verify RL loop on simple 2D task.

- **Experiment**: `smoke_test`
- **Result**: Agent learned to center in tunnel (Reward -2 -> +14). Pipeline verified.

## [2026-03-14] Phase B: Vision System ("The Fly Eye")

**Goal**: Replace coordinates with 1D visual inputs.

- **Hypothesis**: Agent will fail without velocity input.
- **Result**: Agent SUCCEEDED (Avg Reward ~300). Damping in env allowed reactive control. Hypothesis rejected (positive surprise).

## [2026-03-15] Phase C: 3D Physics & PPO

**Goal**: Continuous control in 3D corridor.

- **Challenge**: Initial PPO run failed (Reward -5000) due to update bug and high difficulty.
- **Fix**: Fixed PPO buffer logic. Tuned env (Low Gravity, Wider Tunnel).
- **Result**: PPO converged. Reward reached ~800/800. Robust flight achieved.

## [2026-03-15] Phase E: The Grand Benchmark

**Goal**: Compare Dense vs. Bio-Constrained Architectures.

- **Experiment**: Trained Dense, Random-Sparse (20%), Local-Bio (20%), Small-World (20%).
- **Outcome**:
  - Random-Sparse: Best Performance (Mean ~840).
  - Small-World: Competitive (Mean ~800).
  - Dense: High variance, lower mean (~705).
- **Conclusion**: Sparsity acts as a regularizer. Topological constraints (Small-World) work better than rigid Local constraints, mimicking real brain integration.
