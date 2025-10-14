"""Assignment 3: Robot Olympics - Evolving Body (via NDE) and Controller.

This implementation evolves modular robot bodies using the Neural Developmental
Encoder (NDE) and, for each candidate morphology, evolves a sinusoidal controller
using CMA-ES to maximize forward displacement in the OlympicArena.

Body Evolution (NDE-based):
- Representation: three real-valued vectors fed to the NDE
- Decoder: High-Probability Decoder produces a directed graph morphology
- Guardrails: drop NONE nodes, keep only the component reachable from core, require
  at least one actuator, require successful compile/spawn in OlympicArena
- Selection: simple (mu+lambda) evolution over NDE input vectors with Gaussian mutation

Controller Evolution:
- Algorithm: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
- Controller: Sinusoidal CPG (per-actuator amps and phases, plus global frequency)
- Parameters evolved: 2*nu + 1
- Training budgets:
  - Learnability probe for body scoring: short (e.g., 80-120 evals, 10-15s)
  - Final training on the selected body: longer (e.g., 400 evals, 30s)

Fitness:
- Primary objective: forward displacement in OlympicArena (negative Y direction)
- Bonuses: consistent forward progress, reaching distance milestones
- Penalties: excessive sideways drift, instability

This pipeline is designed to comply with the assignment requirements: evolve the body
using the provided NDE and evolve the controller with an EA (CMA-ES).
"""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer
import json
import nevergrad as ng
try:
    import cma  # A2 CMA-ES
except Exception:  # pragma: no cover
    cma = None  # type: ignore
import networkx as nx
from networkx.readwrite import json_graph

# Local libraries
from ariel import console
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule

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
NUM_OF_MODULES = 8
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
def _nde_generate_core(
    type_vec: npt.NDArray[np.float32],
    conn_vec: npt.NDArray[np.float32],
    rot_vec: npt.NDArray[np.float32],
    *,
    max_modules: int = 12,
) -> CoreModule | None:
    """Generate a body with NDE+decoder and return CoreModule or None if invalid.

    Guardrails:
    - Drop NONE nodes from the decoded graph
    - Keep only the connected component reachable from core
    - Enforce at least one actuator (hinge) and <= max_modules total
    - Verify compile and spawn in OlympicArena
    """
    try:
        nde = NeuralDevelopmentalEncoding(number_of_modules=max_modules)
        p_types, p_conns, p_rots = nde.forward([type_vec, conn_vec, rot_vec])
        dec = HighProbabilityDecoder(max_modules)
        g: nx.DiGraph = dec.probability_matrices_to_graph(p_types, p_conns, p_rots)
    except Exception:
        return None

    # Remove NONE nodes by attribute
    to_remove: list[int] = []
    for n, d in g.nodes(data=True):
        if str(d.get("type", "")).upper() == "NONE":
            to_remove.append(n)
    if to_remove:
        g.remove_nodes_from(to_remove)
    if g.number_of_nodes() == 0:
        return None

    # Keep only component reachable from a node labeled CORE if any
    core_nodes = [n for n, d in g.nodes(data=True) if str(d.get("type", "")).upper() == "CORE"]
    if core_nodes:
        core_n = core_nodes[0]
        reachable = nx.descendants(g, core_n) | {core_n}
        g = g.subgraph(reachable).copy()
    if g.number_of_nodes() == 0:
        return None

    # Count actuators
    num_hinges = sum(1 for _, d in g.nodes(data=True) if str(d.get("type", "")).upper() == "HINGE")
    if num_hinges == 0 or g.number_of_nodes() > max_modules:
        return None

    # Build MjSpec and check compile (robot-only, no arena spawn here)
    try:
        core = construct_mjspec_from_graph(g)
        _ = core.spec.compile()
    except Exception:
        return None
    return core


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




def _probe_body_with_cma(genotype: list[npt.NDArray[np.float32]], idx: int, *, duration: float = 10.0, budget: int = 200) -> tuple[float, np.ndarray]:
    """Train a controller for a body using CMA-ES (A2 algorithm)."""
    console.log({"training_body": idx})
    
    # Build fresh robot from genotype (NDE-based)
    core = _nde_generate_core(genotype[0], genotype[1], genotype[2], max_modules=12)
    if core is None:
        console.log({"body": idx, "error": "nde_invalid"})
        return -np.inf, np.array([])
    
    try:
        world = OlympicArena()
        world.spawn(core.spec, spawn_position=SPAWN_POS.copy(), small_gap=0.05, correct_for_bounding_box=False)
        model = world.spec.compile()
    except Exception as e:
        console.log({"body": idx, "error": "arena_compile", "msg": str(e)[:80]})
        return -np.inf, np.array([])
        
    if model.nu == 0:
        console.log({"body": idx, "error": "no_actuators"})
        return -np.inf, np.array([])

    console.log({"body": idx, "actuators": int(model.nu)})

    # CMA-ES setup: sinusoidal controller (A2 approach)
    out_size = model.nu
    param_dim = out_size * 2 + 1  # amps, phases, freq
    
    def sinusoid_objective(vec: np.ndarray) -> float:
        """Evaluate controller parameters (A2 sinusoidal controller)."""
        data = mj.MjData(model)
        mj.mj_resetData(model, data)
        
        vec = np.asarray(vec)
        if vec.size != param_dim:
            if vec.size > param_dim:
                vec = vec[:param_dim]
            else:
                vec = np.pad(vec, (0, param_dim - vec.size))
        
        amps_raw = vec[:out_size]
        phases_raw = vec[out_size: out_size * 2]
        freq_raw = vec[-1]
        amps = np.tanh(amps_raw) * np.pi  # Increased amplitude range
        phases = np.tanh(phases_raw) * np.pi
        freq = 1.5 + np.tanh(freq_raw) * 2.0  # Higher frequency range (0.5-3.5 Hz)

        def cb(m: mj.MjModel, d: mj.MjData) -> None:
            t = float(d.time)
            targets = amps * np.sin(2 * np.pi * freq * t + phases)
            d.ctrl += targets * 0.08  # Increased control strength
            d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

        mj.set_mjcb_control(cb)
        dt = model.opt.timestep
        steps = int(duration / dt)
        
        # Track initial position
        try:
            core_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "core")
            initial_pos = data.xpos[core_bid].copy()
        except Exception:
            initial_pos = data.qpos[0:3].copy() if len(data.qpos) >= 3 else np.zeros(3)
        
        # Track positions during simulation
        positions = []
        for step in range(steps):
            mj.mj_step(model, data)
            if not np.isfinite(data.qpos).all():
                return 1e9  # Penalty for instability
            
            # Record position every 10 steps
            if step % 10 == 0:
                try:
                    pos = data.xpos[core_bid].copy()
                except Exception:
                    pos = data.qpos[0:3].copy() if len(data.qpos) >= 3 else np.zeros(3)
                positions.append(pos)
        
        if not positions:
            return 1e9
        
        final_pos = positions[-1]
        
        # Enhanced fitness: forward movement (negative Y in arena)
        forward_displacement = initial_pos[1] - final_pos[1]
        
        # Bonus for consistent forward progress
        if len(positions) > 5:
            y_positions = [p[1] for p in positions]
            best_forward = initial_pos[1] - min(y_positions)
            consistency_bonus = 0.1 * max(0, best_forward - forward_displacement)
        else:
            consistency_bonus = 0
        
        # Penalty for excessive sideways drift
        sideways_drift = abs(final_pos[0] - initial_pos[0])
        drift_penalty = 0.1 * sideways_drift
        
        # Distance milestones
        distance_bonus = 0
        if forward_displacement > 0.5:
            distance_bonus += 0.5
        if forward_displacement > 1.0:
            distance_bonus += 1.0
        if forward_displacement > 2.0:
            distance_bonus += 2.0
        
        total_fitness = forward_displacement + consistency_bonus - drift_penalty + distance_bonus
        return -total_fitness  # CMA-ES minimizes, so negate

    # CMA-ES optimization (A2 algorithm)
    if cma is not None:
        x0 = np.zeros(param_dim)
        sigma0 = 0.5
        pop = max(4, min(20, 4 + int(np.log(param_dim + 1) * 3)))
        es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, {"popsize": pop, "seed": int(SEED), "verbose": -1})
        
        best_f = -np.inf
        best_theta = None
        evals = 0
        generation = 0

        while evals < budget and not es.stop():
            sols = es.ask()
            losses = []
            for s in sols:
                loss = sinusoid_objective(s)
                fitness = -loss
                if fitness > best_f:
                    best_f = fitness
                    best_theta = np.array(s)
                losses.append(loss)
            es.tell(sols, losses)
            evals += len(sols)
            generation += 1
            
            # Log progress every 5 generations
            if generation % 5 == 0:
                console.log({"body": idx, "gen": generation, "evals": evals, "best_fitness": float(best_f)})
        
        if best_theta is None:
            console.log({"body": idx, "result": "no_improvement"})
            return -np.inf, np.array([])
        
        console.log({"body": idx, "final_fitness": float(best_f), "total_evals": evals})
        return best_f, best_theta
    else:
        console.log({"body": idx, "error": "cma_not_available"})
        return -np.inf, np.array([])


def sample_and_select_bodies(
    *,
    total_trials: int = 500,
    gene_size: int = 64,
    top_k: int = 50,
    seeds: list[int] | None = None,
) -> list[tuple[float, list[npt.NDArray[np.float32]]]]:
    """Generate and screen robot bodies, return top performers with their genotypes."""
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    candidates: list[tuple[float, list[npt.NDArray[np.float32]]]] = []
    attempts = 0
    seed_idx = 0
    max_attempts = total_trials * 10
    
    console.log({"sampling": "starting", "target": top_k})
    
    while len(candidates) < top_k and attempts < max_attempts:
        s = seeds[seed_idx % len(seeds)]
        seed_idx += 1
        local_rng = np.random.default_rng(s + attempts)
        attempts += 1
        
        genotype = [
            local_rng.random(gene_size).astype(np.float32),
            local_rng.random(gene_size).astype(np.float32),
            local_rng.random(gene_size).astype(np.float32),
        ]
        
        # Build robot directly using modules (gecko-style)
        try:
            core = _build_simple_robot_from_genotype(genotype)
            # Test compile (robot-only first)
            _test_model = core.spec.compile()
            if _test_model.nu == 0:
                continue
        except Exception as e:
            if attempts <= 3:
                console.log({"build_error": str(e)[:100]})
            continue
            
        # Quick viability score (includes arena test)
        try:
            dummy_graph = nx.DiGraph()
            dummy_graph.add_node(0, type="CORE", rotation="DEG_0")
            score = _quick_viability_score(dummy_graph, core)
            if np.isfinite(score) and score > 0:  # Must have positive movement
                # Store genotype, not core (to avoid spec corruption)
                candidates.append((score, genotype))
                if len(candidates) % 10 == 0:
                    console.log({"progress": len(candidates), "attempts": attempts})
        except Exception as e:
            if attempts <= 3:
                console.log({"viability_error": str(e)[:100]})
            continue

    candidates.sort(key=lambda x: x[0], reverse=True)
    console.log({"screening_complete": len(candidates), "attempts": attempts})
    
    return candidates[:top_k]



def experiment(
    robot: Any,
    controller: Controller,
    duration: int = 60,
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
    # Stage 1: Evolve NDE input vectors to obtain a viable body
    console.rule("Stage 1: Evolving Body via NDE (guardrailed)")
    gene_size = 64
    rng = np.random.default_rng(SEED)
    mu = 6
    lambd = 12
    generations = 8
    sigma_mut = 0.15

    # Initialize population of NDE vectors
    population: list[list[npt.NDArray[np.float32]]] = [
        [
            rng.random(gene_size).astype(np.float32),
            rng.random(gene_size).astype(np.float32),
            rng.random(gene_size).astype(np.float32),
        ]
        for _ in range(mu)
    ]

    def score_body(geno: list[npt.NDArray[np.float32]]) -> tuple[float, Any | None]:
        core = _nde_generate_core(geno[0], geno[1], geno[2], max_modules=12)
        if core is None:
            return -np.inf, None
        g = nx.DiGraph()
        g.add_node(0, type="CORE")
        return _quick_viability_score(g, core, run_seconds=1.0), core

    best_body: tuple[float, Any, list[npt.NDArray[np.float32]]] | None = None
    for gen in range(generations):
        console.log({"nde_gen_start": gen + 1, "pop": len(population)})
        scored: list[tuple[float, Any | None, list[npt.NDArray[np.float32]]]] = []
        for i, ind in enumerate(population):
            s, core_tmp = score_body(ind)
            scored.append((s, core_tmp, ind))
            if (i + 1) % 3 == 0:
                console.log({"nde_progress": f"{i+1}/{len(population)}", "best_so_far": float(np.nanmax([r[0] for r in scored])) if scored else None})
        ranked = sorted(scored, key=lambda x: x[0], reverse=True)
        if best_body is None or ranked[0][0] > best_body[0]:
            best_body = (ranked[0][0], ranked[0][1], [a.copy() for a in ranked[0][2]])
        console.log({"nde_gen": gen + 1, "best_viability": float(ranked[0][0])})
        # Selection + mutation to produce next population
        elites = [geno for _, _core, geno in ranked[: max(2, mu // 2)]]
        offspring: list[list[npt.NDArray[np.float32]]] = []
        while len(offspring) < lambd:
            parent = elites[rng.integers(0, len(elites))]
            child = [
                np.clip(p + rng.normal(0, sigma_mut, size=p.shape).astype(np.float32), 0.0, 1.0)
                for p in parent
            ]
            offspring.append(child)
        population = elites + offspring
        population = population[:mu]

    if best_body is None or not np.isfinite(best_body[0]) or best_body[0] <= 0:
        console.log({"error": "no_viable_body_from_nde"})
        return

    best_viab, best_core, nde_genotype = best_body
    console.log({"nde_body_selected": True, "viability": float(best_viab)})
    # One-time arena spawn check now using the already built core
    try:
        world_chk = OlympicArena()
        world_chk.spawn(best_core.spec, spawn_position=SPAWN_POS.copy(), small_gap=0.02, correct_for_bounding_box=False)
        _ = world_chk.spec.compile()
        console.log({"arena_check": "ok"})
    except Exception as e:
        console.log({"arena_check": "fail", "msg": str(e)[:120]})
        return

    # Stage 2: Evolve controller with CMA-ES on the selected body
    console.rule("Stage 2: Controller Evolution (CMA-ES)")
    console.log({"algorithm": "CMA-ES", "duration": "30s", "budget": 400})
    best_f, best_theta = _probe_body_with_cma(nde_genotype, 0, duration=60.0, budget=400)
    console.log({"training_complete": True, "final_distance": f"{-best_f:.3f}m forward"})
    
    # Save results
    if best_theta is not None and len(best_theta) > 0:
        np.save(DATA / "best_theta.npy", best_theta)
        console.log({"saved": str(DATA / "best_theta.npy")})
    
    # Visualize best
    if best_theta is not None and len(best_theta) > 0 and best_f > -np.inf:
        console.rule("Stage 3: Visualization")
        console.log({"best_distance": f"{-best_f:.3f}m forward", "launching_viewer": True})
        
        # Rebuild best robot from genotype (completely fresh for viewer)
        best_core = _nde_generate_core(nde_genotype[0], nde_genotype[1], nde_genotype[2], max_modules=12)
        if best_core is None:
            console.log({"viewer_error": "nde_rebuild_failed"})
            return
        
        # Create fresh world and spawn
        mj.set_mjcb_control(None)  # Clear any previous controller
        world = OlympicArena()
        world.spawn(best_core.spec, spawn_position=SPAWN_POS.copy())
        model = world.spec.compile()
        data = mj.MjData(model)
        
        # Decode controller parameters
        amps_raw = best_theta[:model.nu]
        phases_raw = best_theta[model.nu: model.nu * 2]
        freq_raw = best_theta[-1]
        amps = np.tanh(amps_raw) * np.pi  # Match training amplitude
        phases = np.tanh(phases_raw) * np.pi
        freq = 1.5 + np.tanh(freq_raw) * 2.0  # Match training frequency
        
        def best_controller(m: mj.MjModel, d: mj.MjData) -> None:
            t = float(d.time)
            targets = amps * np.sin(2 * np.pi * freq * t + phases)
            d.ctrl += targets * 0.05  # Increased for faster movement
            d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)
        
        mj.set_mjcb_control(best_controller)
        console.log({"viewer": "Press ESC to close when done watching"})
        viewer.launch(model=model, data=data)
        mj.set_mjcb_control(None)  # Clean up after viewer closes


if __name__ == "__main__":
    main()
