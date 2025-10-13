"""Configuration for A3_template_jack.

Move configurable constants, defaults and paths here so they are easier to
customize without editing the main script.
"""
from pathlib import Path
from typing import Literal

import numpy as np

# Random generator seed
SEED: int = 42

# Workspace / data paths
SCRIPT_NAME = "A3_template_jack.py"
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# World / spawn settings
SPAWN_POS: list[float] = [-0.8, 0, 0.1]
TARGET_POSITION: list[float] = [5, 0, 0.5]

# Robot / genotype settings
NUM_OF_MODULES: int = 30
GENOTYPE_SIZE: int = 64

# Controller / network defaults
HIDDEN_SIZE: int = 8

# Simulation defaults
DEFAULT_DURATION: int = 15
EVALUATION_DURATION: int = 30

# Viewer mode type annotation
ViewerTypes = Literal[
    "launcher",
    "video",
    "simple",
    "tracking",
    "no_control",
    "frame",
]

# Numpy RNG instance (constructed here so modules importing config share the same RNG)
RNG = np.random.default_rng(SEED)

# Hinge insertion/probabilities
P_HINGE: float = 0.8  # Probability to add a hinge between eligible blocks
# Allowed rotation labels for inserted hinges (must match ModuleRotationsIdx names)
HINGE_ROTATION_CHOICES: list[str] = [
    "DEG_0",
    "DEG_90",
    "DEG_180",
    "DEG_270",
]

# CMA-ES defaults
CMA_POPSIZE: int = 48  # Increased for better exploration
CMA_SIGMA0: float = 1.5  # Larger initial step size
CMA_MAX_GENERATIONS: int = 3  # Run longer

# Co-evolution controller dimensionality (fixed IO for stable CMA dimension)
COEV_MAX_ACTUATORS: int = 32  # must be >= possible hinge count
COEV_INPUT_SIZE: int = COEV_MAX_ACTUATORS * 2  # e.g., qpos subset + qvel subset (truncated/padded)

# Co-evolution quick controller polish during evaluation (to gauge body viability)
COEV_POLISH_USE_CMA: bool = True        # Use mini-CMA instead of random search
COEV_POLISH_STEPS: int = 5              # Inner CMA generations
COEV_POLISH_POPSIZE: int = 8            # Candidates per generation
COEV_POLISH_SIGMA: float = 0.3          # Controller CMA sigma
COEV_POLISH_EVAL_DURATION: int = 8      # Shorter eval for polish

# Evaluation robustness
COEV_NUM_EVALS_PER_CANDIDATE: int = 3   # Repeated evals, use median

# Diversity bonus (encourage morphological exploration)
COEV_DIVERSITY_WEIGHT: float = 0.1      # Weight for diversity term in fitness

# Logging frequency for CMA/co-evolution
LOG_EVERY_N_GEN: int = 1

# Parallel processing settings
# Set to 1 to disable parallelization, or None to use (CPU_COUNT - 1)
# Recommended: leave as None for automatic detection, or set to specific number
NUM_PARALLEL_WORKERS: int | None = None  # None = auto-detect and use (cpu_count - 1)
