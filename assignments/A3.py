"""EC A3 Template Code (Jack)."""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer
# progress helper (not used right now)

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    CoreModule,
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import (
    single_frame_renderer,
    tracking_video_renderer,
    video_renderer,
)
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
from a3_config import (
    SEED,
    GENOTYPE_SIZE,
    NUM_OF_MODULES,
    SPAWN_POS,
    TARGET_POSITION,
    DEFAULT_HIDDEN_SIZE,
    NN_WEIGHT_LOC,
    NN_WEIGHT_SCALE,
    DEFAULT_POP_SIZE,
    DEFAULT_GENERATIONS,
    DEFAULT_TOURNAMENT_K,
    DEFAULT_NUM_PARENTS,
    DEFAULT_BODY_SIGMA,
    DEFAULT_CONTROLLER_SIGMA,
    DEFAULT_ELITISM,
    DEFAULT_PARALLEL,
    DEFAULT_SIM_MODE,
    DEFAULT_MAX_WORKERS,
    DEFAULT_RUN_DURATION,
    EVALUATE_DURATION,
    DEFAULT_PLOT_AND_RECORD,
)

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Type Aliases
type ViewerTypes = Literal[
    "launcher",
    "video",
    "simple",
    "tracking",
    "no_control",
    "frame",
]
type Vector = npt.NDArray[np.float64]

# --- RANDOM GENERATOR SETUP --- #
RNG = np.random.default_rng(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# Global variables from config (imported above)


def fitness_function(history: list[tuple[float, float, float]]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    return -cartesian_distance


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
    plt.savefig(DATA / "robot_path.png")


def create_robot_body(
    genotype: list[np.ndarray] | None = None,
    *,
    save_graph: bool = False,
) -> CoreModule:
    # Create random genotype if None is provided
    if genotype is None:
        type_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
        conn_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
        rot_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
        genotype = [
            type_p_genes,
            conn_p_genes,
            rot_p_genes,
        ]

    # Decode the genotype into probability matrices
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_matrices = nde.forward(genotype)

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    # Save the graph to a file
    if save_graph is True:
        save_graph_as_json(
            robot_graph,
            DATA / "robot_graph.json",
        )

    # Print all nodes
    return construct_mjspec_from_graph(robot_graph)


def controller_genotype_length_for_robot(robot: CoreModule, hidden_size: int = DEFAULT_HIDDEN_SIZE) -> int:
    """Return the flattened number of parameters required to encode the NN weights.

    We use three dense matrices (no biases):
      W1: input_size x hidden_size
      W2: hidden_size x hidden_size
      W3: hidden_size x output_size

    The function instantiates an NN to read input/output sizes from the robot.
    """
    nn = NN(robot, hidden_size=hidden_size)
    in_sz = int(nn.input_size)
    hid = int(nn.hidden_size)
    out_sz = int(nn.output_size)
    return int(in_sz * hid + hid * hid + hid * out_sz)


def decode_controller_genotype(
    controller_genotype: npt.NDArray[np.floating] | np.ndarray,
    robot: CoreModule,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
 ) -> tuple[Vector, Vector, Vector]:
    """Decode a 1-D controller_genotype into weight matrices for NN.

    If the provided genotype is shorter or longer than expected it will be
    padded with zeros or trimmed respectively.
    Returns a tuple (W1, W2, W3) shaped appropriately.
    """
    total = controller_genotype_length_for_robot(robot, hidden_size=hidden_size)
    arr = np.asarray(controller_genotype, dtype=np.float64)
    if arr.size < total:
        pad = np.zeros(total - arr.size, dtype=np.float64)
        arr = np.concatenate([arr, pad])
    elif arr.size > total:
        arr = arr[:total]

    nn = NN(robot, hidden_size=hidden_size)
    in_sz = nn.input_size
    hid = nn.hidden_size
    out_sz = nn.output_size

    idx = 0
    s1 = in_sz * hid
    w1 = arr[idx: idx + s1].reshape((in_sz, hid))
    idx += s1
    s2 = hid * hid
    w2 = arr[idx: idx + s2].reshape((hid, hid))
    idx += s2
    s3 = hid * out_sz
    w3 = arr[idx: idx + s3].reshape((hid, out_sz))
    return (w1, w2, w3)


def create_full_genotype(
    body_genotype: list[np.ndarray] | None = None,
    controller_genotype: npt.NDArray[np.floating] | None = None,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
) -> tuple[list[np.ndarray] | None, npt.NDArray[np.floating], CoreModule]:
    """Create (or complete) a full genotype consisting of body and controller.

    - If body_genotype is None, a random body genotype is created (same as
      create_robot_body does internally).
    - The body genotype is decoded into a robot to infer controller sizes.
    - If controller_genotype is None, a random controller genotype of the
      required flattened length is created using the global RNG.

    Returns (body_genotype, controller_genotype, robot)
    """
    # If no body genotype provided, create one and keep it so we can return it
    if body_genotype is None:
        type_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
        conn_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
        rot_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
        body_genotype = [type_p_genes, conn_p_genes, rot_p_genes]

    # Create robot from the (possibly newly created) body genotype
    robot = create_robot_body(body_genotype)

    # Compute controller genotype length and create / validate genotype
    length = controller_genotype_length_for_robot(robot, hidden_size=hidden_size)
    if controller_genotype is None:
        controller_genotype = RNG.normal(loc=NN_WEIGHT_LOC, scale=NN_WEIGHT_SCALE, size=(length,)).astype(np.float64)
    else:
        controller_genotype = np.asarray(controller_genotype, dtype=np.float64)
        if controller_genotype.size != length:
            # pad or trim to expected length
            if controller_genotype.size < length:
                pad = np.zeros(length - controller_genotype.size, dtype=np.float64)
                controller_genotype = np.concatenate([controller_genotype, pad])
            else:
                controller_genotype = controller_genotype[:length]

    return (body_genotype, controller_genotype, robot)


def initialize_population(pop_size: int = DEFAULT_POP_SIZE, hidden_size: int = DEFAULT_HIDDEN_SIZE) -> list[tuple[list[np.ndarray], npt.NDArray[np.floating]]]:
    """Initialize a small population of individuals.

    Each individual is a tuple (body_genotype, controller_genotype).
    - body_genotype: list of three numpy arrays (type, conn, rot) of length GENOTYPE_SIZE
    - controller_genotype: 1-D array of flattened network weights

    Default population size is small (8) for fast iterations.
    """
    population: list[tuple[list[np.ndarray], npt.NDArray[np.floating]]] = []
    for _ in range(int(pop_size)):
        # create body genotype and robot; create_full_genotype will construct
        # both body and controller if controller_genotype is None
        body_gen, controller_gen, _robot = create_full_genotype(
            body_genotype=None,
            controller_genotype=None,
            hidden_size=hidden_size,
        )
        # body_gen is guaranteed to be a list[np.ndarray]
        population.append((body_gen, controller_gen.copy()))
    return population


def mutate_body_genotype(
    body_genotype: list[np.ndarray],
    sigma: float = 0.02,
    clip: tuple[float, float] = (0.0, 1.0),
) -> list[np.ndarray]:
    """Return a mutated copy of the body_genotype.

    Mutation is performed by adding Gaussian noise (mean=0, std=sigma)
    to each gene array and clipping to the valid range (default [0,1]).
    """
    mutated: list[np.ndarray] = []
    for arr in body_genotype:
        # ensure float64 for stable noise addition
        a = np.asarray(arr, dtype=np.float64)
        noise = RNG.normal(loc=0.0, scale=float(sigma), size=a.shape)
        a_mut = a + noise
        a_mut = np.clip(a_mut, clip[0], clip[1])
        # keep original dtype
        mutated.append(a_mut.astype(arr.dtype))
    return mutated


def mutate_controller_genotype(
    controller_genotype: npt.NDArray[np.floating] | np.ndarray,
    sigma: float = 0.1,
) -> npt.NDArray[np.floating]:
    """Return a mutated copy of the flattened controller genotype.

    Mutation is Gaussian additive noise on every weight.
    """
    arr = np.asarray(controller_genotype, dtype=np.float64)
    noise = RNG.normal(loc=0.0, scale=float(sigma), size=arr.shape)
    mutated = arr + noise
    return mutated.astype(controller_genotype.dtype if hasattr(controller_genotype, "dtype") else np.float64)


def mutate_individual(
    individual: tuple[list[np.ndarray], npt.NDArray[np.floating]],
    body_sigma: float = 0.02,
    controller_sigma: float = 0.1,
) -> tuple[list[np.ndarray], npt.NDArray[np.floating]]:
    """Mutate an individual (body_genotype, controller_genotype) and return the new individual.

    This creates copies and does not mutate the input in-place.
    """
    body_gen, controller_gen = individual
    new_body = mutate_body_genotype(body_gen, sigma=body_sigma)
    new_controller = mutate_controller_genotype(controller_gen, sigma=controller_sigma)
    return (new_body, new_controller)


def quick_spawn(
    robot: CoreModule,
) -> tuple[mj.MjModel, mj.MjData, OlympicArena]:
    mj.set_mjcb_control(None)
    world = OlympicArena()
    temp_robot_xml = robot.spec.to_xml()
    temp_robot = mj.MjSpec.from_string(temp_robot_xml)
    world.spawn(
        temp_robot,
        position=SPAWN_POS,
    )
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    return (cast("mj.MjModel", model), data, world)


class NN:
    def __init__(self, robot: CoreModule, hidden_size: int = DEFAULT_HIDDEN_SIZE) -> None:
        _, data, _ = quick_spawn(robot)

        # Get relevant info
        self.input_size = int(len(data.qpos.copy()))
        self.hidden_size = int(hidden_size)
        self.output_size = int(len(data.ctrl))

        # Clear cache
        del data

    def random_controller(
        self,
    ) -> None:
        # Initialize the networks weights randomly
        # Normally, you would use the genes of an individual as the weights,
        # Here we set them randomly for simplicity.
        w1 = RNG.normal(
            loc=NN_WEIGHT_LOC,
            scale=NN_WEIGHT_SCALE,
            size=(self.input_size, self.hidden_size),
        )
        w2 = RNG.normal(
            loc=NN_WEIGHT_LOC,
            scale=NN_WEIGHT_SCALE,
            size=(self.hidden_size, self.hidden_size),
        )
        w3 = RNG.normal(
            loc=NN_WEIGHT_LOC,
            scale=NN_WEIGHT_SCALE,
            size=(self.hidden_size, self.output_size),
        )
        self.weights = (w1, w2, w3)

    def set_controller_weights(self, weights: tuple[Vector, Vector, Vector]) -> None:
        self.weights = weights

    def forward(
        self,
        model: mj.MjModel,
        data: mj.MjData,
    ) -> npt.NDArray[np.float64]:
        # Get inputs, in this case the positions of the actuator motors (hinges)
        inputs = data.qpos

        # Run the inputs through the lays of the network.
        layer1 = np.tanh(np.dot(inputs, self.weights[0]))
        layer2 = np.tanh(np.dot(layer1, self.weights[1]))
        outputs = np.tanh(np.dot(layer2, self.weights[2]))

        # Scale the outputs
        return outputs * np.pi


def run(
    model: mj.MjModel,
    data: mj.MjData,
    duration: int = DEFAULT_RUN_DURATION,
    mode: ViewerTypes = "viewer",
    *,
    output_folder: str | None = None,
) -> None:
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
            path_to_video_folder = str(DATA / "videos") if output_folder is None else str(output_folder)
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)

            # Render with video recorder
            video_renderer(
                model,
                data,
                duration=duration,
                video_recorder=video_recorder,
            )
        case "tracking":
            # This records a video of the simulation
            path_to_video_folder = str(DATA / "videos") if output_folder is None else str(output_folder)
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)

            # Render with video recorder
            tracking_video_renderer(
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


def evaluate(
    robot: CoreModule,
    nn: NN,
    *,
    plot_and_record: bool = DEFAULT_PLOT_AND_RECORD,
    mode: ViewerTypes = "simple",
    output_folder: str | None = None,
) -> float:
    # Define what to track
    mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )

    # Create the controller
    controller = Controller(
        controller_callback_function=nn.forward,
        tracker=tracker,
    )

    # Create the robot in the world
    model, data, world = quick_spawn(robot)

    # Pass the model and data to the tracker
    controller.tracker.setup(world.spec, data)

    # Set the control callback function
    mj.set_mjcb_control(controller.set_control)

    # Choose the mode
    duration = EVALUATE_DURATION
    # Use provided mode for both recording and non-recording runs. If the
    # caller requests plot_and_record=True they may still pick the same mode.
    run(model, data, duration=duration, mode=mode, output_folder=output_folder)

    if plot_and_record is True:
        # Show the tracked history after the run
        show_xpos_history(tracker.history["xpos"][0])

    # Calculate and print the fitness
    return fitness_function(tracker.history["xpos"][0])


def main() -> None:
    """Entry point."""
    run_evolution(
        generations=DEFAULT_GENERATIONS,
        pop_size=DEFAULT_POP_SIZE,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        tournament_k=DEFAULT_TOURNAMENT_K,
        num_parents=DEFAULT_NUM_PARENTS,
        body_sigma=DEFAULT_BODY_SIGMA,
        controller_sigma=DEFAULT_CONTROLLER_SIGMA,
        elitism=DEFAULT_ELITISM,
        parallel=DEFAULT_PARALLEL,
        max_workers=DEFAULT_MAX_WORKERS,
        sim_mode=DEFAULT_SIM_MODE,
    )


def run_evolution(
    generations: int = DEFAULT_GENERATIONS,
    pop_size: int = DEFAULT_POP_SIZE,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    tournament_k: int = DEFAULT_TOURNAMENT_K,
    num_parents: int = DEFAULT_NUM_PARENTS,
    body_sigma: float = DEFAULT_BODY_SIGMA,
    controller_sigma: float = DEFAULT_CONTROLLER_SIGMA,
    elitism: int = DEFAULT_ELITISM,
    parallel: bool = True,
    max_workers: int | None = None,
    sim_mode: ViewerTypes = "simple",
    plot_and_record: bool = DEFAULT_PLOT_AND_RECORD,
    output_folder: str | None = None,
) -> tuple[tuple[list[np.ndarray], npt.NDArray[np.floating]] | None, float]:
    """Run a simple generational evolutionary algorithm.

    Returns the best individual and its fitness.
    """
    console.log("Initializing population")
    population = initialize_population(pop_size=pop_size, hidden_size=hidden_size)

    best_individual = None
    best_fitness = -np.inf

    for gen in range(int(generations)):
        console.log(f"Generation {gen+1}/{generations}: evaluating population")
        # Create per-generation output folder for videos/plots when requested.
        gen_output: str | None = None
        if output_folder is not None:
            gen_path = Path(output_folder) / f"gen_{gen+1}"
            gen_path.mkdir(parents=True, exist_ok=True)
            gen_output = str(gen_path)
        else:
            # default location under DATA/videos/gen_{n}
            gen_path = DATA / "videos" / f"gen_{gen+1}"
            gen_path.mkdir(parents=True, exist_ok=True)
            gen_output = str(gen_path)

        fitnesses = evaluate_population(
            population,
            hidden_size=hidden_size,
            plot_and_record=plot_and_record,
            parallel=parallel,
            max_workers=max_workers,
            mode=sim_mode,
            output_folder=gen_output,
        )

        # Track best
        for ind, fit in zip(population, fitnesses):
            if fit > best_fitness:
                best_fitness = fit
                best_individual = ind

        console.log(f"Generation {gen+1} best fitness: {max(fitnesses):.5f} mean: {float(np.mean(fitnesses)):.5f}")

        # Selection
        parents = tournament_selection(population, fitnesses, k=tournament_k, num_parents=num_parents)

        # Reproduction: fill new population via mutation of parents
        new_population: list[tuple[list[np.ndarray], npt.NDArray[np.floating]]] = []

        # Elitism: keep top `elitism` individuals (avoid scalar conversion warnings)
        if elitism > 0 and len(fitnesses) > 0:
            # get indices of top elites (highest fitness)
            idxs = np.argsort(fitnesses)[-int(elitism):]
            # ensure we append elites in descending fitness order
            idxs = idxs[::-1]
            for i in idxs:
                new_population.append(population[int(i)])

        # Fill the remainder of new_population
        while len(new_population) < pop_size:
            # Select random parent from parents list
            parent = parents[RNG.integers(len(parents))]
            child = mutate_individual(parent, body_sigma=body_sigma, controller_sigma=controller_sigma)
            new_population.append(child)

        population = new_population

    console.log(f"Evolution finished. Best fitness: {best_fitness}")
    return (best_individual, float(best_fitness))


def evaluate_individual(
    individual: tuple[list[np.ndarray], npt.NDArray[np.floating]],
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    plot_and_record: bool = DEFAULT_PLOT_AND_RECORD,
    mode: ViewerTypes = "simple",
    output_folder: str | None = None,
) -> float:
    """Decode an individual and return its fitness.

    Individual is (body_genotype, controller_genotype).
    This function constructs the robot body, decodes controller weights,
    sets up an NN, and calls the existing evaluate() function.
    """
    body_gen, controller_gen = individual
    # Build robot from body genotype
    robot = create_robot_body(body_gen)

    # Create NN and set weights from controller genotype
    nn = NN(robot, hidden_size=hidden_size)
    try:
        weights = decode_controller_genotype(controller_gen, robot, hidden_size=hidden_size)
        nn.set_controller_weights(weights)
    except Exception:
        nn.random_controller()

    # Evaluate and return fitness
    fitness = evaluate(robot, nn, plot_and_record=plot_and_record, mode=mode, output_folder=output_folder)
    return float(fitness)


def _eval_worker(args: tuple[int, tuple[list[np.ndarray], npt.NDArray[np.floating]], int, bool, ViewerTypes, str | None]) -> tuple[int, float]:
    """Worker wrapper for parallel evaluation. Returns (index, fitness)."""
    idx, individual, hidden_size, plot_and_record, mode, output_folder = args
    fit = evaluate_individual(individual, hidden_size=hidden_size, plot_and_record=plot_and_record, mode=mode, output_folder=output_folder)
    return (idx, float(fit))


def evaluate_population(
    population: list[tuple[list[np.ndarray], npt.NDArray[np.floating]]],
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    plot_and_record: bool = DEFAULT_PLOT_AND_RECORD,
    parallel: bool = False,
    max_workers: int | None = None,
    mode: ViewerTypes = "simple",
    output_folder: str | None = None,
) -> list[float]:
    """Evaluate all individuals and return a list of fitnesses.

    If `parallel` is True, evaluations will be dispatched to a
    ProcessPoolExecutor. We preserve ordering by returning results
    as a list aligned with the input population.
    """
    if not parallel:
        fitnesses: list[float] = []
        for ind in population:
            fitnesses.append(evaluate_individual(ind, hidden_size=hidden_size, plot_and_record=plot_and_record, mode=mode, output_folder=output_folder))
        return fitnesses

    # Parallel evaluation using processes (safer for mujoco heavy work)
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # When running in parallel, avoid plotting/recording in worker processes. Headless
    # worker processes may not have access to display backends or the background
    # image file used by `show_xpos_history`. If the caller requested plotting while
    # parallel=True, we disable it for workers and log a short note.
    if parallel and plot_and_record:
        console.log("plot_and_record requested but parallel=True; disabling plots/recordings in worker processes")
    worker_plot_flag = False if parallel else bool(plot_and_record)

    args_iter = [(i, population[i], int(hidden_size), worker_plot_flag, mode, output_folder) for i in range(len(population))]
    results: list[float | None] = [None] * len(population)
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(_eval_worker, arg) for arg in args_iter]
        for fut in as_completed(futures):
            idx, fit = fut.result()
            results[int(idx)] = float(fit)
    # all entries should be filled; replace any stray None with 0.0
    return [float(r) if r is not None else 0.0 for r in results]


def tournament_selection(
    population: list[tuple[list[np.ndarray], npt.NDArray[np.floating]]],
    fitnesses: list[float],
    k: int = 3,
    num_parents: int = 4,
) -> list[tuple[list[np.ndarray], npt.NDArray[np.floating]]]:
    """Select parents using k-way tournament selection.

    - k: tournament size
    - num_parents: number of parents to return
    Returns a list of selected individuals (may contain duplicates).
    """
    pop_size = len(population)
    selected: list[tuple[list[np.ndarray], npt.NDArray[np.floating]]] = []
    for _ in range(int(num_parents)):
        # sample k unique indices
        ids = RNG.choice(pop_size, size=min(k, pop_size), replace=False)
        best_idx = ids[0]
        best_fit = fitnesses[best_idx]
        for idx in ids:
            if fitnesses[int(idx)] > best_fit:
                best_fit = fitnesses[int(idx)]
                best_idx = int(idx)
        selected.append(population[best_idx])
    return selected


if __name__ == "__main__":
    main()
