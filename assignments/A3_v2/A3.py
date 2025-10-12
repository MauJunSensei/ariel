"""Assignment 3 template code."""

from __future__ import annotations

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from concurrent.futures import ProcessPoolExecutor, as_completed
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
import nevergrad as ng

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Ensure the local `A3_config.py` is importable even when this script is
# executed from the repository root or another working directory. This
# inserts the script's directory at the front of sys.path so the local
# module is preferred.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    POPULATION_SIZE,
    GENERATIONS, 
    MUTATION_RATE,
    CONTROLLER_HIDDEN_SIZE,
    RUN_DURATION,
    GENOTYPE_SIZE,
    MODE,
    TOURNAMENT_SIZE,
    PARALLEL_EVAL,
    ELITISM_K,
    TRAIN_CONTROLLER,
    TRAIN_BUDGET,
    TRAIN_DURATION,
    TRAIN_ALGO,
    VIABILITY_CHECK,
    VIABILITY_DURATION,
    VIABILITY_MIN_DISPLACEMENT,
    VIABILITY_MAX_ATTEMPTS,
    
)

# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]

class Individual:
    def __init__(
        self,
        body_genes: list[np.ndarray],
        controller: Controller,
        controller_weights: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.body_genes: list[np.ndarray] = body_genes
        self.graph: DiGraph[Any] | None = None
        self.controller: Controller = controller
        self.fitness: float = -np.inf
        # Optional controller weights included in genotype; initialized lazily when model dims are known
        self.controller_weights: dict[str, np.ndarray] | None = controller_weights

# ------------------------------
# Controller weight utils and trainer
# ------------------------------
def _weight_shapes_for_model(model: mj.MjModel) -> dict[str, tuple[int, int]]:
    """Expected weight shapes for the simple NN controller for a given MuJoCo model."""
    input_size = model.nq  # same as len(data.qpos)
    hidden = CONTROLLER_HIDDEN_SIZE
    output = model.nu
    return {
        "w1": (input_size, hidden),
        "w2": (hidden, hidden),
        "w3": (hidden, output),
    }


def _flatten_weights(weights: dict[str, np.ndarray], shapes: dict[str, tuple[int, int]]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for k in ("w1", "w2", "w3"):
        r, c = shapes[k]
        w = weights.get(k)
        if w is None or w.shape != (r, c):
            parts.append(np.zeros((r, c)).ravel())
        else:
            parts.append(w.ravel())
    return np.concatenate(parts, dtype=float)


def _unflatten_weights(vec: np.ndarray, shapes: dict[str, tuple[int, int]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    idx = 0
    for k in ("w1", "w2", "w3"):
        r, c = shapes[k]
        size = r * c
        part = vec[idx: idx + size]
        if part.size != size:
            part = np.zeros(size)
        out[k] = np.asarray(part, dtype=float).reshape((r, c))
        idx += size
    return out


def _train_controller_for_body(
    robot: Any,
    controller: Controller,
    *,
    duration: float,
    budget: int,
    algo: str = "cma",
    init_weights: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Train controller weights for the given robot using Nevergrad.

    Runs short rollouts of length `duration` inside the OlympicArena to maximize
    displacement along +X. Returns the best-found weight dictionary.
    """
    # Isolate callbacks
    mj.set_mjcb_control(None)

    # Build environment model once
    world = OlympicArena(load_precompiled=False)
    world.spawn(robot.spec, position=SPAWN_POS, correct_collision_with_floor=True)
    model = world.spec.compile()
    shapes = _weight_shapes_for_model(model)

    # Initial vector
    if init_weights is None:
        rng = np.random.default_rng(123)
        init_weights = {
            "w1": rng.normal(0, 0.5, size=shapes["w1"]),
            "w2": rng.normal(0, 0.5, size=shapes["w2"]),
            "w3": rng.normal(0, 0.5, size=shapes["w3"]),
        }
    x0 = _flatten_weights(init_weights, shapes)

    # Parameterization with bounds
    lower = np.full_like(x0, -3.0, dtype=float)
    upper = np.full_like(x0, 3.0, dtype=float)
    parametrization = ng.p.Array(init=x0).set_bounds(lower, upper)

    if algo.lower() == "cma":
        optimizer: ng.optimization.base.Optimizer = ng.optimizers.CMA(parametrization=parametrization, budget=budget)
    else:
        optimizer = ng.optimizers.TBPSA(parametrization=parametrization, budget=budget)

    def objective(xvec: np.ndarray) -> float:
        # Unpack weights
        weights = _unflatten_weights(np.asarray(xvec, dtype=float), shapes)
        # Fresh data and reset
        data = mj.MjData(model)
        mj.mj_resetData(model, data)
        # Reset and setup tracker
        controller.tracker.reset()
        controller.tracker.setup(world.spec, data)
        # Control callback
        def _cb(m: mj.MjModel, d: mj.MjData) -> None:
            controller.set_control(m, d, weights=weights)
        mj.set_mjcb_control(_cb)
        # Short rollout
        simple_runner(model, data, duration=duration)
        # Score
        hist = controller.tracker.history.get("xpos", {}).get(0, [])
        fit = fitness_function(hist)
        return -float(fit if np.isfinite(fit) else -1e9)

    rec = optimizer.minimize(objective)
    best_vec = np.asarray(rec.value, dtype=float)
    return _unflatten_weights(best_vec, shapes)

def _uniform_mix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise uniform crossover between two arrays of the same shape."""
    if a.shape != b.shape:
        return a.copy()
    mask = RNG.random(a.shape) < 0.5
    out = a.copy()
    out[~mask] = b[~mask]
    return out

def crossover(parent_a: Individual, parent_b: Individual) -> Individual:
    """Create a child by uniform crossover of body genes and controller weights.

    The child's body graph is generated; validity is checked by attempting to
    construct the MuJoCo spec. Caller may still apply mutation and validation.
    """
    child = Individual(body_genes=[], controller=generate_controller(), controller_weights=None)
    # Body genes (3 arrays)
    for ga, gb in zip(parent_a.body_genes, parent_b.body_genes):
        child.body_genes.append(_uniform_mix(ga, gb))

    # Controller weights (optional)
    if parent_a.controller_weights is not None or parent_b.controller_weights is not None:
        child.controller_weights = {}
        keys: set[str] = set()
        if parent_a.controller_weights is not None:
            keys.update(parent_a.controller_weights.keys())
        if parent_b.controller_weights is not None:
            keys.update(parent_b.controller_weights.keys())
        for k in keys:
            wa = parent_a.controller_weights.get(k) if parent_a.controller_weights else None
            wb = parent_b.controller_weights.get(k) if parent_b.controller_weights else None
            if wa is None and wb is not None:
                child.controller_weights[k] = wb.copy()
            elif wb is None and wa is not None:
                child.controller_weights[k] = wa.copy()
            elif wa is None and wb is None:
                # nothing to inherit for this key
                continue
            else:
                child.controller_weights[k] = _uniform_mix(wa, wb)

    # Build graph from genes
    generate_body(child, child.body_genes)
    return child

def _is_valid_body(ind: Individual) -> bool:
    """Validate that individual's body can be constructed into a MuJoCo model."""
    try:
        if ind.graph is None:
            return False
        robot = construct_mjspec_from_graph(ind.graph)
        _model = robot.spec.compile()
        return True
    except Exception as e:
        console.log(f"Invalid body during validation: {e}")
        return False

def generate_body_genes():
    # Generate random body genes
    type_p_genes = RNG.uniform(-100, 100, GENOTYPE_SIZE).astype(np.float32)
    conn_p_genes = RNG.uniform(-100, 100, GENOTYPE_SIZE).astype(np.float32)
    rot_p_genes = RNG.uniform(-100, 100, GENOTYPE_SIZE).astype(np.float32)

    return [type_p_genes, conn_p_genes, rot_p_genes]

def generate_controller():
    # Create a controller using the controller weights and return it
    controller = Controller(
        controller_callback_function=nn_controller,
        tracker=Tracker(
            mujoco_obj_to_find=mj.mjtObj.mjOBJ_BODY,
            name_to_bind="core",
        ),
    )
    return controller
    
def generate_body(ind: Individual, genes: list[np.ndarray] | None = None) -> None:
    # Create a body using the body genes
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_matrices = nde.forward(genes if genes is not None else ind.body_genes)

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    ind.graph = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )


def _quick_viability(robot: Any, duration: float, min_dx: float) -> bool:
    """Fast headless rollout with a naive controller to test if body can move.

    Uses a small sine-wave actuation across all DOFs to induce motion and
    checks the forward displacement along +X. Returns True if displacement
    exceeds min_dx.
    """
    try:
        world = OlympicArena(load_precompiled=False)
        world.spawn(robot.spec, position=SPAWN_POS, correct_collision_with_floor=True)
        model = world.spec.compile()
        data = mj.MjData(model)
        mj.mj_resetData(model, data)

        # Simple sine controller inline (avoid using Controller for speed)
        t = 0.0
        dt = model.opt.timestep
        steps_per = 50
        tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_BODY, name_to_bind="core")
        tracker.setup(world.spec, data)

        while data.time < duration:
            if model.nu > 0:
                freq = 1.5
                amps = 0.8
                data.ctrl[:] = amps * np.sin(2 * np.pi * freq * t)
            mj.mj_step(model, data, nstep=steps_per)
            tracker.update(data)
            t += dt * steps_per

        hist = tracker.history.get("xpos", {}).get(0, [])
        fit = fitness_function(hist)
        return bool(np.isfinite(fit) and fit >= min_dx)
    except Exception:
        return False


def fitness_function(history: list[tuple[float, float, float]]) -> float:
    """Fitness: forward displacement along +X (x_last - x_first).

    Mirrors the displacement-based scoring used in `A3_clean.py`'s
    quick viability / controller-training objectives.
    """
    if not history:
        return -1e6

    pos = np.array(history)
    if pos.size == 0 or pos.ndim != 2 or pos.shape[1] < 1:
        return -1e6

    x = pos[:, 0]
    # Displacement from start to finish
    try:
        displacement = float(x[-1] - x[0])
    except Exception:
        return -1e6

    if not np.isfinite(displacement):
        return -1e6

    return displacement


def show_xpos_history(history: list[float], save_path: Path | str | None = None) -> None:
    # Create a tracking camera
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    # Initialize world to get the background
    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(
        model,
        data,
        save_path=save_path,
        save=True,
    )

    # Setup background image
    img = plt.imread(save_path)
    _, ax = plt.subplots()
    ax.imshow(img)
    w, h, _ = img.shape

    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Calculate initial position
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, SPAWN_POS[0]

    # Convert position data to pixel coordinates
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
    pos_data_pixel = [[xc, yc]]
    for i in range(len(pos_data) - 1):
        xi, yi, _ = pos_data[i]
        xj, yj, _ = pos_data[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pos_data_pixel[i]
        pos_data_pixel.append([xn + int(xd), yn + int(yd)])
    pos_data_pixel = np.array(pos_data_pixel)

    # Plot x,y trajectory
    ax.plot(x0, y0, "kx", label="[0, 0, 0]")
    ax.plot(xc, yc, "go", label="Start")
    ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", label="Path")
    ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", label="End")

    # Add labels and title
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()

    # Title
    plt.title("Robot Path in XY Plane")

    # Save or show results
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
    *,
    weights: dict[str, np.ndarray] | None = None,
) -> npt.NDArray[np.float64]:
    # Simple 3-layer neural network with provided weights
    # Expecting weights dict with keys 'w1','w2','w3'
    # Fallback to zeros if not provided to avoid re-seeding every step
    input_size = len(data.qpos)
    hidden_size = CONTROLLER_HIDDEN_SIZE
    output_size = model.nu

    if weights is None:
        w1 = np.zeros((input_size, hidden_size))
        w2 = np.zeros((hidden_size, hidden_size))
        w3 = np.zeros((hidden_size, output_size))
    else:
        w1 = weights.get("w1")
        w2 = weights.get("w2")
        w3 = weights.get("w3")
        # Safety: if any are missing/mismatched, fall back to zeros of correct shape
        if w1 is None or w1.shape != (input_size, hidden_size):
            w1 = np.zeros((input_size, hidden_size))
        if w2 is None or w2.shape != (hidden_size, hidden_size):
            w2 = np.zeros((hidden_size, hidden_size))
        if w3 is None or w3.shape != (hidden_size, output_size):
            w3 = np.zeros((hidden_size, output_size))

    # Get inputs (joint positions)
    inputs = data.qpos

    # Feedforward
    layer1 = np.tanh(inputs @ w1)
    layer2 = np.tanh(layer1 @ w2)
    outputs = np.tanh(layer2 @ w3)

    # Scale the outputs to [-pi, pi]
    return outputs * np.pi


def experiment(
    robot: Any,
    controller: Controller,
    duration: int = 15,
    mode: ViewerTypes = "viewer",
    controller_weights: dict[str, np.ndarray] | None = None,
) -> None:
    """Run the simulation with random movements."""
    # ==================================================================== #
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = OlympicArena(
        load_precompiled=False,
    )

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    # Pass the model and data to the tracker
    controller.tracker.setup(world.spec, data)

    # Set the control callback function
    # This is called every time step to get the next action.
    args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    # Initialize controller weights if not provided; attach to controller for later retrieval
    if controller_weights is None:
        input_size = len(data.qpos)
        hidden_size = CONTROLLER_HIDDEN_SIZE
        output_size = model.nu
        controller_weights = {
            "w1": RNG.normal(loc=0.0, scale=0.5, size=(input_size, hidden_size)),
            "w2": RNG.normal(loc=0.0, scale=0.5, size=(hidden_size, hidden_size)),
            "w3": RNG.normal(loc=0.0, scale=0.5, size=(hidden_size, output_size)),
        }
    # Attach to controller instance for optional external access
    setattr(controller, "weights", controller_weights)
    kwargs: dict[str, Any] = {"weights": controller_weights}

    mj.set_mjcb_control(
        lambda m, d: controller.set_control(m, d, *args, **kwargs),  # pyright: ignore[reportUnknownLambdaType]
    )

    # ------------------------------------------------------------------ #
    match mode:
        case "simple":
            # This disables visualisation (fastest option)
            simple_runner(
                model,
                data,
                duration=duration,
            )
        case "frame":
            # Render a single frame (for debugging)
            save_path = str(DATA / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
        case "video":
            # This records a video of the simulation
            path_to_video_folder = str(DATA / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)

            # Render with video recorder
            video_renderer(
                model,
                data,
                duration=duration,
                video_recorder=video_recorder,
            )
        case "launcher":
            # This opens a liver viewer of the simulation
            viewer.launch(
                model=model,
                data=data,
            )
        case "no_control":
            # If mj.set_mjcb_control(None), you can control the limbs manually.
            mj.set_mjcb_control(None)
            viewer.launch(
                model=model,
                data=data,
            )
    # ==================================================================== #

def evaluate(individual: Individual) -> float:
    """Evaluate an individual and return its fitness."""
    # Construct the robot from the graph
    # Ensure the graph exists
    if individual.graph is None:
        generate_body(individual)
    robot = construct_mjspec_from_graph(individual.graph)

    # Optionally train controller for this body before full evaluation
    if TRAIN_CONTROLLER:
        try:
            trained = _train_controller_for_body(
                robot,
                individual.controller,
                duration=TRAIN_DURATION,
                budget=TRAIN_BUDGET,
                algo=TRAIN_ALGO,
                init_weights=individual.controller_weights,
            )
            individual.controller_weights = trained
        except Exception as e:
            console.log(f"Controller training failed: {e}")

    # Run the experiment
    experiment(
        robot=robot,
        controller=individual.controller,
        duration=RUN_DURATION,
        mode=MODE,
        controller_weights=individual.controller_weights,
    )

    # If controller weights were lazily initialized in experiment, store them in the individual
    if getattr(individual.controller, "weights", None) is not None and individual.controller_weights is None:
        individual.controller_weights = {
            k: v.copy() for k, v in getattr(individual.controller, "weights").items()
        }

    # Calculate fitness
    fitness = fitness_function(individual.controller.tracker.history["xpos"][0])
    return fitness

def select_survivors(population: list[Individual], num_survivors: int) -> list[Individual]:
    """Tournament selection with elitism and unique winners in a generation."""
    valid = [ind for ind in population]
    if not valid:
        raise ValueError("No individuals with valid fitness to select from.")

    # Typed helper for fitness access to satisfy type checkers
    def _fitness_of(ind: "Individual") -> float:
        return float(ind.fitness)

    ranked = sorted(valid, key=_fitness_of, reverse=True)
    elites = ranked[:min(ELITISM_K, num_survivors)]

    survivors: list[Individual] = elites.copy()
    pool: list[Individual] = [ind for ind in ranked if ind not in set(survivors)]
    while len(survivors) < num_survivors and pool:
        k = min(len(pool), TOURNAMENT_SIZE)
        idxs = RNG.choice(len(pool), size=k, replace=False)
        tourney: list[Individual] = [pool[i] for i in idxs]
        winner = max(tourney, key=_fitness_of)
        survivors.append(winner)
        pool.remove(winner)

    return survivors

def _evaluate_worker(payload: tuple[list[np.ndarray], dict[str, np.ndarray] | None]) -> tuple[float, dict[str, np.ndarray] | None]:
    """Worker function executed in a separate process.

    Accepts a picklable payload (body_genes, controller_weights). Reconstructs
    an Individual, builds its body, runs a headless evaluation and returns
    the fitness and any resulting controller weights. This avoids pickling
    non-serializable objects like Controller/Tracker/MuJoCo objects.
    """
    body_genes, controller_weights = payload

    # Reconstruct a minimal Individual inside the worker
    try:
        ind = Individual(
            body_genes=[g.copy() for g in body_genes],
            controller=generate_controller(),
            controller_weights=None,
        )
        # Build body graph and compile robot spec
        generate_body(ind, genes=ind.body_genes)
        robot = construct_mjspec_from_graph(ind.graph)
    except Exception:
        # Invalid body or build failure -> very low fitness
        return (-1e9, None)

    # Optionally train controller in the worker
    try:
        if TRAIN_CONTROLLER:
            trained = _train_controller_for_body(
                robot,
                ind.controller,
                duration=TRAIN_DURATION,
                budget=TRAIN_BUDGET,
                algo=TRAIN_ALGO,
                init_weights=controller_weights,
            )
            controller_weights = trained
    except Exception:
        pass

    # Run the experiment headless (use "simple" mode for speed)
    try:
        experiment(
            robot=robot,
            controller=ind.controller,
            duration=RUN_DURATION,
            mode="simple",
            controller_weights=controller_weights,
        )

        resulting_weights = getattr(ind.controller, "weights", None)
        fitness = fitness_function(ind.controller.tracker.history["xpos"][0])
        return (float(fitness), {k: v.copy() for k, v in resulting_weights.items()} if resulting_weights is not None else None)
    except Exception:
        return (-1e9, None)


def evaluate_all(population: list[Individual]) -> None:
    """Evaluate all individuals, optionally in parallel when PARALLEL_EVAL is True.

    Uses a top-level worker that accepts only picklable data (body_genes,
    controller_weights) so it is safe with ProcessPoolExecutor.

    IMPORTANT: skip individuals that already have a finite fitness so that
    elitism (carrying over best individuals without re-evaluation) works.
    """
    # Determine which indices actually need evaluation
    to_eval_idxs = [i for i, ind in enumerate(population) if not np.isfinite(ind.fitness)]
    if not to_eval_idxs:
        return

    if not PARALLEL_EVAL:
        for i in to_eval_idxs:
            population[i].fitness = evaluate(population[i])
        return

    # Prepare serializable tasks only for individuals that need evaluation
    tasks: list[tuple[int, list[np.ndarray], dict[str, np.ndarray] | None]] = [
        (i, [g.copy() for g in population[i].body_genes], population[i].controller_weights)
        for i in to_eval_idxs
    ]

    # Map futures -> original population index
    with ProcessPoolExecutor() as ex:
        futures = {ex.submit(_evaluate_worker, (body_genes, controller_weights)): idx for (idx, body_genes, controller_weights) in tasks}
        for future in as_completed(futures):
            orig_idx = futures[future]
            ind = population[orig_idx]
            try:
                fit, weights = future.result()
                ind.fitness = fit
                # If worker returned trained weights and main process had none, store them
                if ind.controller_weights is None and weights is not None:
                    ind.controller_weights = weights
            except Exception as e:
                console.log(f"Evaluation failed for individual {orig_idx}: {e}")
                ind.fitness = -1e9

def mutate(individual: Individual) -> None:
    """Mutate an individual's genes with a certain probability. Currently does nothing."""
    noise_scale = 5.0  # tweak this to increase/decrease mutation magnitude
    gene_min, gene_max = -100.0, 100.0

    for i, arr in enumerate(individual.body_genes):
        arr = arr.copy()
        mask = RNG.random(arr.shape) < MUTATION_RATE
        if mask.any():
            noise = RNG.normal(loc=0.0, scale=noise_scale, size=arr.shape).astype(arr.dtype)
            arr[mask] = np.clip(arr[mask] + noise[mask], gene_min, gene_max)
        individual.body_genes[i] = arr

    # Rebuild the graph from mutated genes and invalidate fitness
    generate_body(individual, genes=individual.body_genes)
    # Mutate controller weights if present
    if individual.controller_weights is not None:
        for k, w in individual.controller_weights.items():
            w2 = w.copy()
            mask_w = RNG.random(w2.shape) < MUTATION_RATE
            if mask_w.any():
                noise_w = RNG.normal(loc=0.0, scale=0.1, size=w2.shape).astype(w2.dtype)
                w2[mask_w] = w2[mask_w] + noise_w[mask_w]
            individual.controller_weights[k] = w2
    individual.fitness = -np.inf
    

def main() -> None:
    """Entry point."""
    # ? ------------------------------------------------------------------ #
    # Initialize the population
    
    # Store population as list of (graph, controller) tuples with precise typing
    population: list[Individual] = []
    attempts = 0
    while len(population) < POPULATION_SIZE and attempts < (POPULATION_SIZE * max(1, VIABILITY_MAX_ATTEMPTS)):
        attempts += 1
        ind = Individual(
            body_genes=generate_body_genes(),
            controller=generate_controller(),
            controller_weights=None,
        )
        generate_body(ind)
        if not _is_valid_body(ind):
            continue
        robot = construct_mjspec_from_graph(ind.graph)
        if not VIABILITY_CHECK or _quick_viability(robot, duration=VIABILITY_DURATION, min_dx=VIABILITY_MIN_DISPLACEMENT):
            population.append(ind)

    if len(population) == 0:
        raise ValueError("Population is empty.")
    else:
        console.log(f"Initialized population with {len(population)} individuals.")
    
    # ? ------------------------------------------------------------------ #
    # Evolutionary loop

    fitness_history: list[list[float]] = []

    for gen in range(GENERATIONS):
        console.log(f"Generation {gen + 1}/{GENERATIONS}, Population size: {len(population)}")

        evaluate_all(population)

        fitness_vals = [ind.fitness for ind in population]
        if fitness_vals:
            console.log(f"Current best fitness: {max(fitness_vals)}")
            console.log(f"Current average fitness: {sum(fitness_vals) / len(fitness_vals)}")

        survivors = select_survivors(population, num_survivors=POPULATION_SIZE // 2)
        offspring: list[Individual] = []

        # Generate offspring using crossover + mutation with validity check
        while len(offspring) < POPULATION_SIZE - len(survivors):
            # Randomly choose two parents from survivors (sample indices for type safety)
            idxs = RNG.choice(len(survivors), size=2, replace=True)
            pa, pb = survivors[int(idxs[0])], survivors[int(idxs[1])]
            child = crossover(pa, pb)
            # Mutate child
            mutate(child)
            # Rebuild and validate body
            generate_body(child, child.body_genes)
            if _is_valid_body(child):
                offspring.append(child)
            else:
                # Fallback: choose fitter parent clone and slight mutate
                base = pa if pa.fitness >= pb.fitness else pb
                clone = Individual(
                    body_genes=[arr.copy() for arr in base.body_genes],
                    controller=generate_controller(),
                    controller_weights=(
                        {k: v.copy() for k, v in base.controller_weights.items()} if base.controller_weights else None
                    ),
                )
                generate_body(clone, clone.body_genes)
                mutate(clone)
                if _is_valid_body(clone):
                    offspring.append(clone)
        
        console.log(f"Selected {len(survivors)} survivors, generated {len(offspring)} offspring.")
        fitness_history.append([ind.fitness for ind in population])

        population = survivors + offspring

    # Plot average and best fitness over generations
    if fitness_history:
        arr = np.array(fitness_history, dtype=np.float64)
        # Replace non-finite values with NaN so nan-aware reductions work
        arr = np.where(np.isfinite(arr), arr, np.nan)

        avg_fitness = np.nanmean(arr, axis=1)
        best_fitness = np.nanmax(arr, axis=1)

        generations = np.arange(1, len(avg_fitness) + 1)

        plt.figure(figsize=(8, 4.5))
        plt.plot(generations, avg_fitness, label="Average fitness", linewidth=2)
        plt.plot(generations, best_fitness, label="Best fitness", linewidth=2)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Average and Best Fitness Over Generations")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        out_path = DATA / "fitness_over_time.png"
        plt.savefig(str(out_path), dpi=200)
        console.log(f"Saved fitness plot to: {out_path}")
    else:
        console.log("No fitness history available to plot.")

    # ------------------------------------------------------------------ #
    # After evolution: find best individual and save its graph + visualization
    # best: Individual = population[0]
    # for ind in population:
    #     if ind.fitness > best.fitness:
    #         best = ind

    # # ? ------------------------------------------------------------------ #
    # # Print all nodes
    # core = construct_mjspec_from_graph(best.graph) # type: ignore

    # # ? ------------------------------------------------------------------ #
    # mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    # name_to_bind = "core"
    # tracker = Tracker(
    #     mujoco_obj_to_find=mujoco_type_to_find,
    #     name_to_bind=name_to_bind,
    # )

    # # ? ------------------------------------------------------------------ #
    # # Simulate the robot
    # ctrl = Controller(
    #     controller_callback_function=nn_controller,
    #     # controller_callback_function=random_move,
    #     tracker=tracker,
    # )

    # experiment(robot=core, controller=ctrl, mode=MODE)

    # show_xpos_history(tracker.history["xpos"][0])

    # fitness = fitness_function(tracker.history["xpos"][0])
    # msg = f"Fitness of generated robot: {fitness}"
    # console.log(msg)

    # save_graph_as_json(
    #     best.graph,
    #     DATA / "robot_graph.json",
    # )


if __name__ == "__main__":
    main()