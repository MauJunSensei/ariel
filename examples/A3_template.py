"""Assignment 3 template code."""

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
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
from ariel.body_phenotypes.robogen_lite.config import ModuleType

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)
GRAPHS_DIR = DATA / "graphs"
GRAPHS_DIR.mkdir(exist_ok=True)

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]


def fitness_function(history: list[float]) -> float:
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
        camera=camera,
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
) -> npt.NDArray[np.float64]:
    # Simple 3-layer neural network
    input_size = len(data.qpos)
    hidden_size = 8
    output_size = model.nu

    # Initialize the networks weights randomly
    # Normally, you would use the genes of an individual as the weights,
    # Here we set them randomly for simplicity.
    w1 = RNG.normal(loc=0.0138, scale=0.5, size=(input_size, hidden_size))
    w2 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, hidden_size))
    w3 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, output_size))

    # Get inputs, in this case the positions of the actuator motors (hinges)
    inputs = data.qpos

    # Run the inputs through the lays of the network.
    layer1 = np.tanh(np.dot(inputs, w1))
    layer2 = np.tanh(np.dot(layer1, w2))
    outputs = np.tanh(np.dot(layer2, w3))

    # Scale the outputs
    return outputs * np.pi
def _build_body_from_genotype(
    num_modules: int,
    genotype: list[npt.NDArray[np.float32]],
):
    nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
    p_matrices = nde.forward(genotype)

    hpd = HighProbabilityDecoder(num_modules)
    robot_graph = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )
    core = construct_mjspec_from_graph(robot_graph)
    return robot_graph, core


def _random_genotype(gene_size: int) -> list[npt.NDArray[np.float32]]:
    return [
        RNG.random(gene_size).astype(np.float32),
        RNG.random(gene_size).astype(np.float32),
        RNG.random(gene_size).astype(np.float32),
    ]


def _quick_viability_score(
    graph: "DiGraph",
    core: Any,
    *,
    run_seconds: float = 1.2,
) -> float:
    # Graph checks
    num_nodes = graph.number_of_nodes()
    hinge_count = sum(1 for _, d in graph.nodes(data=True) if d["type"] == ModuleType.HINGE.name)
    if num_nodes < 6 or hinge_count < 2 or num_nodes > NUM_OF_MODULES:
        return -np.inf

    # Minimal headless run in OlympicArena with tiny oscillatory control
    try:
        world = OlympicArena()
        # Avoid pre-step bounding-box correction (can cause instability for random bodies)
        world.spawn(
            core.spec,
            spawn_position=SPAWN_POS.copy(),
            small_gap=0.02,
            correct_for_bounding_box=False,
        )
        model = world.spec.compile()
        if model.nu == 0:
            return -np.inf
        data = mj.MjData(model)
    except Exception:
        return -np.inf

    def baseline_ctrl(m: mj.MjModel, d: mj.MjData) -> None:
        d.ctrl[:] = 0.2 * np.sin(2.0 * d.time)

    mj.set_mjcb_control(baseline_ctrl)
    dt = model.opt.timestep
    steps = int(run_seconds / dt)

    # Track body named "core" if present; fallback to root qpos x otherwise
    try:
        core_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "core")
        x0 = float(data.xpos[core_bid, 0])
    except Exception:
        x0 = float(data.qpos[0]) if len(data.qpos) > 0 else 0.0

    for _ in range(steps):
        mj.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            return -np.inf

    try:
        x1 = float(data.xpos[core_bid, 0])
    except Exception:
        x1 = float(data.qpos[0]) if len(data.qpos) > 0 else 0.0
    disp = abs(x1 - x0)
    return disp


def _save_graph(graph: "DiGraph", save_path: Path) -> None:
    save_graph_as_json(graph, save_path)


def sample_and_select_bodies(
    *,
    total_trials: int = 500,
    gene_size: int = 64,
    top_k: int = 50,
    seeds: list[int] | None = None,
) -> list[tuple[float, "DiGraph", Any]]:
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    per_seed = max(1, total_trials // len(seeds))
    candidates: list[tuple[float, "DiGraph", Any]] = []

    for s in seeds:
        local_rng = np.random.default_rng(s)
        for _ in range(per_seed):
            genotype = [
                local_rng.random(gene_size).astype(np.float32),
                local_rng.random(gene_size).astype(np.float32),
                local_rng.random(gene_size).astype(np.float32),
            ]
            graph, core = _build_body_from_genotype(NUM_OF_MODULES, genotype)
            score = _quick_viability_score(graph, core)
            candidates.append((score, graph, core))

    # Rank and take top_k
    candidates.sort(key=lambda x: (np.isfinite(x[0]), x[0]), reverse=True)
    top = [c for c in candidates if np.isfinite(c[0])][:top_k]

    # Persist graphs and manifest
    manifest = []
    for rank, (score, graph, _core) in enumerate(top, start=1):
        graph_path = GRAPHS_DIR / f"top_{rank:02d}.json"
        _save_graph(graph, graph_path)
        manifest.append({"rank": rank, "score": float(score), "graph": str(graph_path.name)})

    # Save worst (for completeness)
    worst = candidates[-1]
    _save_graph(worst[1], GRAPHS_DIR / "worst.json")

    # Freeze best body for later use
    if top:
        best_graph = top[0][1]
        save_graph_as_json(best_graph, DATA / "robot_graph_best.json")

    # Save manifest
    try:
        import json
        with (DATA / "screening_manifest.json").open("w", encoding="utf-8") as f:
            json.dump({"total_trials": total_trials, "seeds": seeds, "top": manifest}, f, indent=2)
    except Exception:
        pass

    return top



def experiment(
    robot: Any,
    controller: Controller,
    duration: int = 15,
    mode: ViewerTypes = "viewer",
) -> None:
    """Run the simulation with random movements."""
    # ==================================================================== #
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(robot.spec, spawn_position=SPAWN_POS)

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    # Pass the model and data to the tracker
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    # Set the control callback function
    # This is called every time step to get the next action.
    args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

    mj.set_mjcb_control(
        lambda m, d: controller.set_control(m, d, *args, **kwargs),
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


def main() -> None:
    """Entry point."""
    # ? ------------------------------------------------------------------ #
    # Multi-seed screening: sample many bodies, keep top-50, freeze best
    console.rule("Sampling bodies and selecting best")
    top_candidates = sample_and_select_bodies(
        total_trials=500,
        gene_size=64,
        top_k=50,
        seeds=[42, 43, 44, 45, 46],
    )
    robot_graph: DiGraph[Any] = top_candidates[0][1]

    # ? ------------------------------------------------------------------ #
    # Save the graph to a file
    save_graph_as_json(
        robot_graph,
        DATA / "robot_graph.json",
    )

    # ? ------------------------------------------------------------------ #
    # Print all nodes
    core = construct_mjspec_from_graph(robot_graph)

    # ? ------------------------------------------------------------------ #
    mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )

    # ? ------------------------------------------------------------------ #
    # Simulate the robot
    ctrl = Controller(
        controller_callback_function=nn_controller,
        # controller_callback_function=random_move,
        tracker=tracker,
    )

    experiment(robot=core, controller=ctrl, mode="launcher")

    show_xpos_history(tracker.history["xpos"][0])

    fitness = fitness_function(tracker.history["xpos"][0])
    msg = f"Fitness of generated robot: {fitness}"
    console.log(msg)


if __name__ == "__main__":
    main()
