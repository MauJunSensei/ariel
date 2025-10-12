from __future__ import annotations

from typing import Final

# Core GA / Evolution settings (mirrors A3_v2/config.py)
GENOTYPE_SIZE: Final[int] = 64
POPULATION_SIZE: Final[int] = 80  # Smaller for more diversity pressure
GENERATIONS: Final[int] = 100
MUTATION_RATE: Final[float] = 0.25  # Higher to create more variation
MUTATION_STRENGTH: Final[float] = 0.15  # How much genes change when mutated
CONTROLLER_HIDDEN_SIZE: Final[int] = 8
RUN_DURATION: Final[float] = 30.0  # Shorter for faster iterations

MODE: Final[str] = "simple"

MAX_WORKERS: Final[int | None] = None
PARALLEL_EVAL: Final[bool] = True

# Tournament selection settings
TOURNAMENT_SIZE: Final[int] = 5  # Smaller for less selection pressure initially
TOURNAMENT_ROUNDS: Final[int] = 1
ELITISM_K: Final[int] = 8  # Keep fewer elites to allow more exploration

# Fitness function weights and thresholds (as in A3_v2)
FIT_W_FORWARD: Final[float] = 2.0
FIT_W_LATERAL: Final[float] = 0.2
FIT_W_BOUNCE: Final[float] = 0.05
FIT_W_SMOOTH: Final[float] = 0.02

# Section thresholds (along X) for the Olympic track; add bonuses when reached.
FIT_SECTION_THRESHOLDS: Final[list[float]] = [0.5, 1.5, 3.0, 5.0]
FIT_SECTION_BONUSES: Final[list[float]] = [0.5, 1.0, 2.0, 3.0]

# --- A3_merge additional knobs for clarity --- #

# Body graph size (used by NDE + decoder)
NUM_OF_MODULES: Final[int] = 15  # More modules for richer morphologies

# Guardrails on decoded graphs (drop NONE nodes, require core-reachable subgraph)
ENFORCE_GUARDRAILS: Final[bool] = True
REQUIRE_MIN_HINGES: Final[int] = 2  # Require at least 2 actuators for movement
REQUIRE_MAX_HINGES: Final[int] = 8  # Cap hinges to avoid over-complex bodies

# Phase 2: train multiple candidates
PHASE2_TOP_K: Final[int] = 10  # Train top K candidates in Phase 2 (best from Phase 1)

# GA evaluation baseline controller (for morphology screening)
GA_BASELINE_AMPLITUDE: Final[float] = 0.8  # Stronger actuation
GA_BASELINE_FREQUENCY: Final[float] = 3.0  # Faster oscillation
GA_CTRL_STEP: Final[int] = 10
GA_SAVE_STEP: Final[int] = 50
GA_USE_RANDOM_CONTROLLER: Final[bool] = True  # Add randomness to controller for diversity

# CMA-ES controller type: "nn" (feedforward 3-layer) or "cpg" (sinusoidal)
CMA_CONTROLLER_TYPE: Final[str] = "cpg"

# CMA-ES budgets
CMA_MAX_EVALS: Final[int] = 1200
CMA_POPSIZE: Final[int] | None = 24  # None => auto-compute from dimension
CMA_SIGMA0: Final[float] = 0.5


