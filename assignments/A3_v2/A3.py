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
    save_graph_as_json,
    draw_graph,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from concurrent.futures import ProcessPoolExecutor, as_completed
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

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
    FIT_W_FORWARD,
    FIT_W_LATERAL,
    FIT_W_BOUNCE,
    FIT_W_SMOOTH,
    FIT_SECTION_THRESHOLDS,
    FIT_SECTION_BONUSES,
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
        self.fitness: float | None = None
        # Optional controller weights included in genotype; initialized lazily when model dims are known
        self.controller_weights: dict[str, np.ndarray] | None = controller_weights

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


def fitness_function(history: list[tuple[float, float, float]]) -> float:
    """Richer fitness for Olympic track: reward forward progress and robustness.

    Components:
    - Forward progress: max x achieved (normalized by expected track length ~5m)
    - Lateral penalty: average absolute y (stay near center line)
    - Bounce penalty: average absolute dz per step (discourage hopping)
    - Smoothness: average absolute second derivative of x (jerk)
    - Section bonuses: add bonuses when passing X thresholds
    """
    if not history:
        return -1e6

    pos = np.array(history)  # shape (T, 3)
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    # Forward progress (clamped 0..1 by dividing by approximate course length)
    course_len = max(FIT_SECTION_THRESHOLDS[-1], 1.0)
    fwd = np.clip(np.max(x) / course_len, 0.0, 1.0)

    # Lateral deviation (lower is better)
    lateral = float(np.mean(np.abs(y))) if y.size > 0 else 0.0

    # Vertical bounce proxy: mean |dz| per time step
    dz = np.diff(z) if z.size > 1 else np.array([0.0])
    bounce = float(np.mean(np.abs(dz)))

    # Smoothness in forward motion: mean |d2x|
    dx = np.diff(x) if x.size > 1 else np.array([0.0])
    ddx = np.diff(dx) if dx.size > 1 else np.array([0.0])
    smooth = float(np.mean(np.abs(ddx)))

    # Section bonuses
    bonus = 0.0
    xmax = float(np.max(x))
    for thr, b in zip(FIT_SECTION_THRESHOLDS, FIT_SECTION_BONUSES):
        if xmax >= thr:
            bonus += b

    # Combine; penalties subtract from reward
    score = (
        FIT_W_FORWARD * fwd
        - FIT_W_LATERAL * lateral
        - FIT_W_BOUNCE * bounce
        - FIT_W_SMOOTH * smooth
        + bonus
    )
    return float(score)


def show_xpos_history(history: list[float]) -> None:
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

    # Show results
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
    valid = [ind for ind in population if ind.fitness is not None]
    if not valid:
        raise ValueError("No individuals with valid fitness to select from.")

    # Typed helper for fitness access to satisfy type checkers
    def _fitness_of(ind: "Individual") -> float:
        return float(ind.fitness) if ind.fitness is not None else -1e9

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

def evaluate_all(population: list[Individual]) -> None:
    """Evaluate all individuals, optionally in parallel when PARALLEL_EVAL is True."""
    if not PARALLEL_EVAL:
        for ind in population:
            ind.fitness = evaluate(ind)
        return

    with ProcessPoolExecutor() as ex:
        futures = {ex.submit(evaluate, ind): ind for ind in population}
        for future in as_completed(futures):
            ind = futures[future]
            try:
                ind.fitness = future.result()
            except Exception as e:
                console.log(f"Evaluation failed: {e}")
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
    individual.fitness = None
    

def main() -> None:
    """Entry point."""
    # ? ------------------------------------------------------------------ #
    # Initialize the population
    
    # Store population as list of (graph, controller) tuples with precise typing
    population: list[Individual] = [
        Individual(
            body_genes=generate_body_genes(),
            controller=generate_controller(),
            controller_weights=None,
        ) for _ in range(POPULATION_SIZE)
    ]

    for ind in population:
        generate_body(ind)

    if len(population) == 0:
        raise ValueError("Population is empty.")
    else:
        console.log(f"Initialized population with {len(population)} individuals.")
    
    # ? ------------------------------------------------------------------ #
    # Evolutionary loop

    for gen in range(GENERATIONS):
        console.log(f"Generation {gen + 1}/{GENERATIONS}, Population size: {len(population)}")
        if gen > 0:
            fitness_vals = [ind.fitness for ind in population if ind.fitness is not None]
            if fitness_vals:
                console.log(f"Current best fitness: {max(fitness_vals)}")

        evaluate_all(population)
        for ind in population:
            console.log(f"Individual fitness: {ind.fitness}")

        survivors = select_survivors(population, num_survivors=POPULATION_SIZE // 2)
        offspring: list[Individual] = []

        for ind in survivors:
            child = Individual(
                body_genes=[],
                controller=generate_controller(),
                controller_weights=None,
            )
            # Deep-copy body genes and build graph in-place
            child.body_genes = [arr.copy() for arr in ind.body_genes]
            generate_body(child, child.body_genes)
            # Inherit controller weights (mutated later) but not the controller instance
            if ind.controller_weights is not None:
                child.controller_weights = {k: v.copy() for k, v in ind.controller_weights.items()}
            mutate(child)
            offspring.append(child)
        
        console.log(f"Selected {len(survivors)} survivors, generated {len(offspring)} offspring.")

        population = survivors + offspring

    # ------------------------------------------------------------------ #
    # After evolution: find best individual and save its graph + visualization
    best = None
    best_fit = -1e12
    for ind in population:
        if ind.fitness is not None and ind.fitness > best_fit:
            best = ind
            best_fit = ind.fitness

    if best is not None and best.graph is not None:
        DATA.mkdir(exist_ok=True)
        json_path = DATA / "best_robot_graph.json"
        png_path = DATA / "best_robot_graph.png"
        try:
            save_graph_as_json(best.graph, save_file=json_path)
            draw_graph(best.graph, title=f"Best graph (fit={best_fit:.3f})", save_file=png_path)
            console.log(f"Saved best graph JSON to {json_path} and image to {png_path}")
        except Exception as e:
            console.log(f"Failed to save best graph: {e}")
    else:
        console.log("No best individual with a graph found to save.")

    # ? ------------------------------------------------------------------ #
    # Print all nodes
    # core = construct_mjspec_from_graph(robot_graph)

    # ? ------------------------------------------------------------------ #
    # mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    # name_to_bind = "core"
    # tracker = Tracker(
    #     mujoco_obj_to_find=mujoco_type_to_find,
    #     name_to_bind=name_to_bind,
    # )

    # ? ------------------------------------------------------------------ #
    # Simulate the robot
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


if __name__ == "__main__":
    main()