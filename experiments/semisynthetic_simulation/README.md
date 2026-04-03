# Semi-Synthetic Simulation

Semi-synthetic causal simulation where **treatment is kept from real data** and only the outcome is simulated from hand-crafted oracle features.

## Quick Links

| What | Where |
|------|-------|
| **How to run locally** | [docs/local.md](docs/local.md) |
| **How to run on Azure** | [docs/azure.md](docs/azure.md) |
| **How to run multiple replicates** | [docs/multiple_runs.md](docs/multiple_runs.md) |
| **Feature definitions** | [docs/features.md](docs/features.md) |
| **Calibrating parameters** | [docs/calibration.md](docs/calibration.md) |
| Config file | `corebehrt/configs/causal/simulate_semisynthetic.yaml` |
| Simulator code | `corebehrt/modules/simulation/semisynthetic_simulator.py` |
| Feature extraction | `corebehrt/modules/simulation/oracle_features.py` |
| Config dataclasses | `corebehrt/modules/simulation/config_semisynthetic.py` |
| Azure component | `corebehrt/azure/components/simulate_semisynthetic.py` |
| Tests | `tests/test_modules/test_simulation/` |

## How It Works (30-Second Summary)

1. Load real EHR data (MEDS format) with real treatment assignments
2. Extract oracle features from each patient's pre-index history (disease burden, age, recency, etc.)
3. Simulate outcome: `P(Y(0)=1) = sigmoid(beta_0 + f(features))`, `P(Y(1)=1) = sigmoid(beta_0 + f(features) + tau)`
4. Observed outcome: `Y = A * Y(1) + (1-A) * Y(0)` where A is the real treatment
5. True ATE is known by construction

## Key Difference from Old Simulation

| | Old (`realistic_simulator`) | New (`semisynthetic_simulator`) |
|---|---|---|
| Treatment | Simulated from latent factors | **Real** (from data) |
| Outcome model | Random latent factor weights | Hand-crafted interpretable features |
| Confounding | Synthetic (shared latent factors) | **Real** (from clinical practice) |
| Index dates | Single global date | Per-patient (from cohort) |
| Config complexity | Latent dims, sparsity, influence scales | Feature coefficients, beta_0, delta |
