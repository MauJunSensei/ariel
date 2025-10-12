from typing import Final

GENOTYPE_SIZE: Final[int] = 64
POPULATION_SIZE: Final[int] = 256
GENERATIONS: Final[int] = 100
MUTATION_RATE: Final[float] = 0.2
CONTROLLER_HIDDEN_SIZE: Final[int] = 8
RUN_DURATION: Final[float] = 15.0

MODE: Final[str] = "simple"

MAX_WORKERS: Final[int | None] = None
PARALLEL_EVAL: Final[bool] = True

# Tournament selection settings
TOURNAMENT_SIZE: Final[int] = 3
TOURNAMENT_ROUNDS: Final[int] = 1
ELITISM_K: Final[int] = 2

# Section thresholds (along X) for the Olympic track; add bonuses when reached.
# These are approximate waypoints: end of flat, end of rugged, mid-way, near finish.
FIT_SECTION_THRESHOLDS: Final[list[float]] = [0.5, 1.5, 3.0, 5.0]
FIT_SECTION_BONUSES: Final[list[float]] = [0.3, 0.5, 0.7, 1.5]

# Controller inner-loop training
# If True, for each new body evaluation we run a short controller optimization
# (CMA-ES via nevergrad) before measuring fitness. This estimates viability of
# the body given a learned controller rather than random/previous weights.
TRAIN_CONTROLLER: Final[bool] = True

# Training budget: number of candidate weight vectors to evaluate per body.
# Keep this small for speed; increase for better controller adaptation.
TRAIN_BUDGET: Final[int] = 256

# Duration for controller training rollouts (seconds). Shorter than RUN_DURATION
# to speed up the inner loop; the final fitness is still measured with RUN_DURATION.
TRAIN_DURATION: Final[float] = 30.0

# Algorithm choice for nevergrad optimizer: 'cma' or 'tbpsa' etc.
TRAIN_ALGO: Final[str] = "cma"

# Quick viability check for initial random bodies
VIABILITY_CHECK: Final[bool] = True
VIABILITY_DURATION: Final[float] = 2.0  # seconds
VIABILITY_MIN_DISPLACEMENT: Final[float] = 0.25  # meters along +X
VIABILITY_MAX_ATTEMPTS: Final[int] = 15

