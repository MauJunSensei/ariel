"""Configuration for assignment A3: centralise magic numbers and defaults.

Put all tunable constants here so experiments are reproducible and easy to change.
"""
from typing import Final

# Random seed: use the same seed for repeatability across experiments
SEED: Final[int] = 42

# Robot/body encoding
# GENOTYPE_SIZE: length of each per-module probability array (type/conn/rot)
GENOTYPE_SIZE: Final[int] = 64
# NUM_OF_MODULES: maximum modules the NDE decoder will arrange
NUM_OF_MODULES: Final[int] = 30

# Spawn / target positions used by the evaluation environment
# SPAWN_POS: where the robot is placed initially (x,y,z)
SPAWN_POS = [-0.8, 0.0, 0.1]
# TARGET_POSITION: target x,y,z used by the fitness function
TARGET_POSITION = [5.0, 0.0, 0.5]

# Neural network defaults
# DEFAULT_HIDDEN_SIZE: neurons in hidden layers of the small feedforward NN
DEFAULT_HIDDEN_SIZE: Final[int] = 8
# NN initial weight distribution parameters (loc=mean, scale=std)
NN_WEIGHT_LOC: Final[float] = 0.0138
NN_WEIGHT_SCALE: Final[float] = 0.5

# Controller / genotype defaults (evolutionary algorithm hyper-parameters)
# DEFAULT_POP_SIZE: starting population size
DEFAULT_POP_SIZE: Final[int] = 20
# DEFAULT_GENERATIONS: number of generations to run by default
DEFAULT_GENERATIONS: Final[int] = 50
# DEFAULT_TOURNAMENT_K: tournament size for selection
DEFAULT_TOURNAMENT_K: Final[int] = 5
# DEFAULT_NUM_PARENTS: how many parents to sample each generation
DEFAULT_NUM_PARENTS: Final[int] = 10
# Mutation strengths
DEFAULT_BODY_SIGMA: Final[float] = 0.02
DEFAULT_CONTROLLER_SIGMA: Final[float] = 0.1
# DEFAULT_ELITISM: how many top individuals to carry over unchanged
DEFAULT_ELITISM: Final[int] = 1

# Simulation / run defaults
# DEFAULT_RUN_DURATION: default duration (seconds) for quick debug runs
DEFAULT_RUN_DURATION: Final[int] = 15
# EVALUATE_DURATION: duration used during evaluate() (this is a larger evaluation)
EVALUATE_DURATION: Final[int] = 30

# WIP: Runtime / mode flags (non-numeric)
# DEFAULT_PARALLEL: whether to use process-based parallel evaluation by default.
#   - True: use ProcessPoolExecutor to evaluate individuals in parallel (faster)
#   - False: evaluate sequentially (simpler, safer for debugging)
# Note: when using parallel=True, prefer non-interactive sim modes like "simple"
# or "video" with plot_and_record=False to avoid viewer issues in worker processes.
DEFAULT_PARALLEL: Final[bool] = True

# DEFAULT_SIM_MODE: default mode passed to the simulator/runner when evaluating.
# Valid options (string):
#   "launcher"  - open interactive launcher/viewer (not recommended for headless or parallel runs)
#   "video"     - record a video via the video recorder
#   "simple"    - no visualization, fastest option (recommended for parallel/CI)
#   "tracking"  - record with a tracking camera
#   "no_control"- launch with no control callback set
#   "frame"     - render a single debug frame
# Choose the default mode for experiments here.
DEFAULT_SIM_MODE: Final[str] = "simple"

# DEFAULT_MAX_WORKERS: maximum number of worker processes to use when parallel=True.
# If None, the ProcessPoolExecutor will choose a sensible default (usually os.cpu_count()).
DEFAULT_MAX_WORKERS: Final[int | None] = None

# Whether to produce plots and save tracking videos after evaluations.
# When True, evaluate() will save the tracked position plot and recording (if supported
# by the selected sim/renderer mode). Disable by default to keep automated runs quiet.
DEFAULT_PLOT_AND_RECORD: Final[bool] = False
