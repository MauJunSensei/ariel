"""EC A3 Template Code (Jack)."""

# Standard library
from typing import TYPE_CHECKING, Any, cast
import logging
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer
# from rich.progress import track  # not used in CMA-driven main
import networkx as nx
import math

# Optional third-party optimizer (pycma)
try:
    import cma  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    cma = None

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

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Type Aliases
import importlib.util
from pathlib import Path as _Path

# Load configuration from config.py located next to this file. This keeps the
# import robust whether the script is executed as a module or a standalone
# script.
_config_path = _Path(__file__).resolve().parent / "config.py"
spec = importlib.util.spec_from_file_location("a3_config", str(_config_path))
cfg = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(cfg)  # type: ignore[attr-defined]

DATA = cfg.DATA
EVALUATION_DURATION = cfg.EVALUATION_DURATION
GENOTYPE_SIZE = cfg.GENOTYPE_SIZE
HIDDEN_SIZE = cfg.HIDDEN_SIZE
NUM_OF_MODULES = cfg.NUM_OF_MODULES
RNG = cfg.RNG
SPAWN_POS = cfg.SPAWN_POS
TARGET_POSITION = cfg.TARGET_POSITION
ViewerTypes = cfg.ViewerTypes
P_HINGE = cfg.P_HINGE
HINGE_ROTATION_CHOICES = cfg.HINGE_ROTATION_CHOICES
CMA_POPSIZE = cfg.CMA_POPSIZE
CMA_SIGMA0 = cfg.CMA_SIGMA0
CMA_MAX_GENERATIONS = cfg.CMA_MAX_GENERATIONS
COEV_MAX_ACTUATORS = cfg.COEV_MAX_ACTUATORS
COEV_INPUT_SIZE = cfg.COEV_INPUT_SIZE
COEV_POLISH_USE_CMA = cfg.COEV_POLISH_USE_CMA
COEV_POLISH_STEPS = cfg.COEV_POLISH_STEPS
COEV_POLISH_POPSIZE = cfg.COEV_POLISH_POPSIZE
COEV_POLISH_SIGMA = cfg.COEVI_POLISH_SIGMA if hasattr(cfg, 'COEVI_POLISH_SIGMA') else cfg.COEV_POLISH_SIGMA
COEV_POLISH_EVAL_DURATION = cfg.COEVI_POLISH_EVAL_DURATION if hasattr(cfg, 'COEVI_POLISH_EVAL_DURATION') else cfg.COEV_POLISH_EVAL_DURATION
COEV_NUM_EVALS_PER_CANDIDATE = cfg.COEV_NUM_EVALS_PER_CANDIDATE
COEV_DIVERSITY_WEIGHT = cfg.COEV_DIVERSITY_WEIGHT
LOG_EVERY_N_GEN = cfg.LOG_EVERY_N_GEN

# Parallel processing configuration
_workers_config = getattr(cfg, 'NUM_PARALLEL_WORKERS', None)
NUM_PARALLEL_WORKERS = _workers_config if _workers_config is not None else max(1, cpu_count() - 1)

type Vector = npt.NDArray[np.float64]

# --- LOGGING SETUP --- #
# Configure comprehensive logging to file (overwrite mode)
log_file_path = _config_path.parent / "A3_execution.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'),  # 'w' mode overwrites old logs
        logging.StreamHandler()  # Also output to console
    ]
)
logger = logging.getLogger(__name__)
logger.info("="*80)
logger.info(f"Starting A3 execution at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Log file: {log_file_path}")
logger.info(f"Parallel workers: {NUM_PARALLEL_WORKERS} (CPU count: {cpu_count()})")
logger.info("="*80)

# --- RANDOM GENERATOR SETUP --- #
# --- configuration values are imported from assignments.A3_v3.config --- #


def fitness_function(history: list[tuple[float, float, float]]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    logger.debug(f"Fitness calculation: distance={cartesian_distance:.4f}, final_pos=({xc:.3f}, {yc:.3f}, {zc:.3f})")
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
    logger.info("Creating robot body...")
    # Create random genotype if None is provided
    if genotype is None:
        logger.info("No genotype provided, generating random genotype")
        type_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
        conn_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
        rot_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
        genotype = [
            type_p_genes,
            conn_p_genes,
            rot_p_genes,
        ]

    # Use NUM_OF_MODULES directly to maximize module count
    base_modules = NUM_OF_MODULES
    logger.info(f"Base modules count: {base_modules} (using maximum allowed)")

    # Decode the genotype into probability matrices
    logger.info("Decoding genotype with NDE...")
    nde = NeuralDevelopmentalEncoding(number_of_modules=base_modules)
    p_matrices = nde.forward(genotype)

    # Decode the high-probability graph
    logger.info("Creating high-probability graph...")
    hpd = HighProbabilityDecoder(base_modules)
    robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    # Ensure all leaf nodes are BRICK type before hinge insertion
    leaf_count = 0
    for n in list(robot_graph.nodes()):
        if robot_graph.out_degree(n) == 0 and robot_graph.nodes[n].get("type") != "CORE":
            robot_graph.nodes[n]["type"] = "BRICK"
            leaf_count += 1
    logger.info(f"Set {leaf_count} leaf nodes to BRICK type")

    # Insert hinges (joints) deterministically using connection probabilities
    logger.info("Inserting hinges between blocks...")
    robot_graph = insert_hinges_between_blocks(robot_graph, conn_p_matrices=p_matrices[1])
    
    final_nodes = robot_graph.number_of_nodes()
    final_edges = robot_graph.number_of_edges()
    logger.info(f"Robot graph created: {final_nodes} nodes, {final_edges} edges")

    # Defensive normalization: ensure every edge 'face' value is a string
    # matching ModuleFaces enum names (e.g., 'LEFT', 'FRONT'). This avoids
    # cases where an Enum instance slips through and causes KeyError in the
    # constructor usage ModuleFaces[face].
    for _u, _v, attrs in robot_graph.edges(data=True):
        face_raw = attrs.get("face", "FRONT")
        try:
            # Enum instances have 'name'
            face_name = face_raw.name  # type: ignore[attr-defined]
        except Exception:
            face_name = str(face_raw).upper()
        attrs["face"] = face_name

    # Save the graph to a file
    if save_graph is True:
        logger.info(f"Saving robot graph to {DATA / 'robot_graph.json'}")
        save_graph_as_json(
            robot_graph,
            DATA / "robot_graph.json",
        )

    # Print all nodes
    logger.info("Constructing MuJoCo specification from graph...")
    return construct_mjspec_from_graph(robot_graph)


def insert_hinges_between_blocks(
    graph: "DiGraph[Any]",
    *,
    conn_p_matrices: npt.NDArray[np.float32] | None = None,
) -> "DiGraph[Any]":
    """Deterministically insert hinge modules between edges based on connection probabilities.

    Uses connection probability matrix to rank edges and insert hinges on top-K edges
    while respecting NUM_OF_MODULES and preserving leaf nodes as blocks.
    """
    logger.info("Starting hinge insertion process...")
    G = graph
    G2 = nx.DiGraph()

    # Copy original nodes
    for n, attrs in G.nodes(data=True):
        G2.add_node(n, **attrs)

    # Prepare candidates (all edges can potentially have hinges inserted, allowing hinge-hinge connections)
    candidates: list[tuple[int, int, dict[str, Any], float]] = []
    for u, v, attrs in G.edges(data=True):
        # Allow hinge-hinge connections - no restrictions on node types
        # If 'v' is a leaf (no children), it is an extremity and must remain a block
        if G.out_degree(v) == 0:
            # Leaf nodes keep their direct connection
            new_attrs = dict(attrs)
            try:
                new_attrs["face"] = attrs.get("face", "FRONT").name  # type: ignore[attr-defined]
            except Exception:
                new_attrs["face"] = str(attrs.get("face", "FRONT")).upper()
            G2.add_edge(u, v, **new_attrs)
        else:
            # Get connection probability score for this edge (if available)
            score = 0.0
            if conn_p_matrices is not None and u < conn_p_matrices.shape[0] and v < conn_p_matrices.shape[1]:
                # Average across all face connections for this u->v pair
                score = float(np.mean(conn_p_matrices[u, v, :]))
            candidates.append((u, v, dict(attrs), score))

    # Budget: how many new nodes (hinges) we can add
    existing_nodes = G2.number_of_nodes()
    max_new_nodes = max(0, NUM_OF_MODULES - existing_nodes)
    logger.info(f"Hinge insertion: {len(candidates)} candidates, budget for {max_new_nodes} new hinges")

    # Decide which candidates will get hinges (deterministic, top-K by score)
    if len(candidates) == 0 or max_new_nodes == 0:
        logger.info("No hinges to insert (no candidates or no budget)")
        # nothing to do or no budget -> copy remaining candidate edges directly
        for u, v, attrs, _ in candidates:
            safe_face_raw = attrs.get("face", "FRONT")
            try:
                safe_face = safe_face_raw.name  # type: ignore[attr-defined]
            except Exception:
                safe_face = str(safe_face_raw).upper()
            new_attrs = dict(attrs)
            new_attrs["face"] = safe_face
            G2.add_edge(u, v, **new_attrs)
        return G2

    # Sort candidates by score (descending) and select top K
    candidates_sorted = sorted(candidates, key=lambda x: x[3], reverse=True)
    num_hinges_to_insert = min(len(candidates_sorted), max_new_nodes)
    selected_set = set(range(num_hinges_to_insert))
    logger.info(f"Inserting {num_hinges_to_insert} hinges based on connection probabilities")

    # Rotation options (deterministic choice based on edge index)
    rotations = HINGE_ROTATION_CHOICES

    # Determine next id (assumes integer node ids)
    next_id = (max(G2.nodes) + 1) if G2.number_of_nodes() > 0 else 1

    hinges_inserted = 0
    for i, (u, v, attrs, _score) in enumerate(candidates_sorted):
        # Normalize face to a string key matching ModuleFaces enum names
        face_raw = attrs.get("face", "FRONT")
        # If face is already an enum, convert to its name; else uppercase string
        try:
            face = face_raw.name  # type: ignore[attr-defined]
        except Exception:
            face = str(face_raw).upper()
        
        if i in selected_set:
            # Insert a deterministic hinge node
            h = next_id
            next_id += 1
            hinges_inserted += 1
            # Choose rotation deterministically based on edge index
            hinge_rotation = rotations[i % len(rotations)]
            # Hinge modules only provide a FRONT site; enforce this.
            hinge_child_face = "FRONT"
            G2.add_node(h, type="HINGE", rotation=hinge_rotation)
            # ensure faces stored are enum-name strings
            G2.add_edge(u, h, face=face)
            G2.add_edge(h, v, face=str(hinge_child_face).upper())
        else:
            # keep original connection, but ensure face is normalized
            safe_face_raw = attrs.get("face", "FRONT")
            try:
                safe_face = safe_face_raw.name  # type: ignore[attr-defined]
            except Exception:
                safe_face = str(safe_face_raw).upper()
            new_attrs = dict(attrs)
            new_attrs["face"] = safe_face
            G2.add_edge(u, v, **new_attrs)
    
    logger.info(f"Successfully inserted {hinges_inserted} hinges")

    # Safety: If we exceeded NUM_OF_MODULES, collapse pass-through hinges only (do not remove leaves)
    over_by = G2.number_of_nodes() - NUM_OF_MODULES
    if over_by > 0:
        logger.warning(f"Exceeded NUM_OF_MODULES by {over_by}, collapsing pass-through hinges...")
        collapsed = 0
        for n in list(G2.nodes()):
            if over_by <= 0:
                break
            if G2.nodes[n].get("type") == "HINGE" and G2.in_degree(n) == 1 and G2.out_degree(n) == 1:
                u = next(G2.predecessors(n))
                v = next(G2.successors(n))
                face_uv = G2.edges[(u, n)].get("face", "FRONT")
                G2.add_edge(u, v, face=face_uv)
                G2.remove_node(n)
                over_by -= 1
                collapsed += 1
        logger.info(f"Collapsed {collapsed} pass-through hinges")

    logger.info(f"Final graph after hinge insertion: {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")
    return G2


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


# --- Controller parameterization helpers --- #
def _nn_param_sizes(nn: "NN") -> tuple[int, int, int, int]:
    """Return (D, in, hid, out) where D is total number of parameters.

    Shapes:
    - w1: (in, hid)
    - w2: (hid, hid)
    - w3: (hid, out)
    """
    inp, hid, out = nn.input_size, nn.hidden_size, nn.output_size
    D = inp * hid + hid * hid + hid * out
    return D, inp, hid, out


def _unflatten_weights(
    x: np.ndarray,
    inp: int,
    hid: int,
    out: int,
) -> tuple[Vector, Vector, Vector]:
    """Convert flat vector into three weight matrices with expected shapes."""
    assert x.ndim == 1
    idx = 0
    w1_sz = inp * hid
    w2_sz = hid * hid
    w3_sz = hid * out

    w1 = x[idx : idx + w1_sz].reshape((inp, hid)); idx += w1_sz
    w2 = x[idx : idx + w2_sz].reshape((hid, hid)); idx += w2_sz
    w3 = x[idx : idx + w3_sz].reshape((hid, out)); idx += w3_sz
    return w1, w2, w3


def _flatten_weights(weights: tuple[Vector, Vector, Vector]) -> np.ndarray:
    """Flatten a tuple of weight matrices into a 1D vector."""
    w1, w2, w3 = weights
    return np.concatenate([w1.ravel(), w2.ravel(), w3.ravel()])


def _evaluate_controller_candidate(
    x: np.ndarray,
    robot: CoreModule,
    inp: int,
    hid: int,
    out: int,
    generation: int,
) -> float:
    """Helper function for parallel evaluation of controller candidates.
    
    Returns distance to minimize (positive values).
    """
    try:
        # Create a new NN instance for this worker
        nn = NN(robot)
        weights = _unflatten_weights(x, inp, hid, out)
        nn.set_controller_weights(weights)
        fit = evaluate(robot, nn, plot_and_record=False)
        return -float(fit)  # Return distance (to minimize)
    except Exception as e:
        logger.error(f"[Gen {generation}] Error evaluating candidate: {e}")
        return float('inf')  # Return worst possible fitness on error


def _objective_minimize_distance( # type: ignore[no-untyped-def]
    x: np.ndarray,
    robot: CoreModule,
    nn: "NN",
    generation: int = 0,
) -> float:
    """CMA objective that returns distance (to be minimized).

    We compute -fitness because evaluate() returns negative distance.
    """
    # Set weights from candidate vector and evaluate
    D, inp, hid, out = _nn_param_sizes(nn)
    if x.shape[0] != D:
        # Resize defensively if optimizer passes unexpected shapes
        x = x[:D]
    weights = _unflatten_weights(x, inp, hid, out)
    nn.set_controller_weights(weights)
    fit = evaluate(robot, nn, plot_and_record=False)
    logger.debug(f"[Gen {generation}] Candidate evaluation: fitness={fit:.4f}, distance={-fit:.4f}")
    return -float(fit)


def train_controller_with_cma(
    *,
    robot: CoreModule,
    population_size: int = 12,
    sigma0: float = 0.5,
    max_generations: int = 30,
) -> tuple["NN", float, np.ndarray]:
    """Optimize controller weights for a fixed robot using CMA-ES.

    Returns (best_nn, best_fitness, best_vector).
    """
    logger.info("="*60)
    logger.info("Starting controller training with CMA-ES")
    logger.info(f"Population size: {population_size}, Sigma0: {sigma0}, Max generations: {max_generations}")
    
    if cma is None:
        raise RuntimeError(
            "pycma is not available. Please install 'cma' to use CMA-ES optimization.",
        )

    # Build a network for this robot (to infer sizes)
    logger.info("Building neural network for robot...")
    nn = NN(robot)
    nn.random_controller()  # seed initial weights
    x0 = _flatten_weights(nn.weights)
    logger.info(f"Controller has {len(x0)} parameters (input={nn.input_size}, hidden={nn.hidden_size}, output={nn.output_size})")

    # CMA options
    opts: dict[str, Any] = {
        "seed": int(RNG.integers(0, 10_000_000)),
        "popsize": population_size,
        "maxiter": max_generations,
        "verbose": -9,
    }

    # Initialize the strategy
    logger.info("Initializing CMA-ES strategy...")
    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)

    best_val = math.inf
    best_x = x0.copy()

    # Optimize
    gen = 0
    logger.info(f"Starting CMA-ES optimization loop with {NUM_PARALLEL_WORKERS} parallel workers...")
    
    # Get network dimensions for parallel evaluation
    _D, inp, hid, out = _nn_param_sizes(nn)
    
    while not es.stop():
        gen += 1
        X = es.ask()
        
        # Evaluate all candidates in parallel
        if NUM_PARALLEL_WORKERS > 1:
            eval_func = partial(_evaluate_controller_candidate, 
                              robot=robot, inp=inp, hid=hid, out=out, generation=gen)
            with Pool(NUM_PARALLEL_WORKERS) as pool:
                fitnesses = pool.map(eval_func, X)
        else:
            # Sequential fallback
            fitnesses = [_evaluate_controller_candidate(np.asarray(xi, dtype=float), 
                                                       robot, inp, hid, out, gen) 
                        for xi in X]
        
        # Track best
        for xi, val in zip(X, fitnesses):
            if val < best_val:
                best_val = val
                best_x = np.asarray(xi, dtype=float).copy()

        es.tell(X, fitnesses)

        # Optional: log a little progress
        if gen % 5 == 0:
            logger.info(f"[CMA Gen {gen}/{max_generations}] best_distance={best_val:.4f} median={np.median(fitnesses):.4f}")
            console.log(f"[CMA] gen={gen}/{max_generations} best_distance={best_val:.4f}")

        # Manual guard on generations (opts['maxiter'] also applies)
        if gen >= max_generations:
            break

    es.result  # finalize
    logger.info(f"CMA-ES optimization completed after {gen} generations")
    logger.info(f"Best distance achieved: {best_val:.4f}")

    # Build the best NN to return
    best_weights = _unflatten_weights(best_x, nn.input_size, nn.hidden_size, nn.output_size)
    nn.set_controller_weights(best_weights)
    best_fitness = -best_val  # convert back to our fitness sign
    logger.info(f"Best fitness (negative distance): {best_fitness:.4f}")
    logger.info("="*60)
    return nn, best_fitness, best_x


class NN:
    def __init__(self, robot: CoreModule) -> None:
        _, data, _ = quick_spawn(robot)

        # Get relevant info
        self.input_size = len(data.qpos.copy())
        self.hidden_size = HIDDEN_SIZE
        self.output_size = len(data.ctrl)
        self.fixed_input = False

        # Clear cache
        del data

    def random_controller(
        self,
    ) -> None:
        # Initialize the networks weights randomly
        # Normally, you would use the genes of an individual as the weights,
        # Here we set them randomly for simplicity.
        w1 = RNG.normal(
            loc=0.0138,
            scale=0.5,
            size=(self.input_size, self.hidden_size),
        )
        w2 = RNG.normal(
            loc=0.0138,
            scale=0.5,
            size=(self.hidden_size, self.hidden_size),
        )
        w3 = RNG.normal(
            loc=0.0138,
            scale=0.5,
            size=(self.hidden_size, self.output_size),
        )
        self.weights = (w1, w2, w3)

    def set_controller_weights(
        self,
        weights: tuple[Vector, Vector, Vector],
    ) -> None:
        self.weights = weights

    def forward(
        self,
        model: mj.MjModel,
        data: mj.MjData,
    ) -> npt.NDArray[np.float64]:
        # Prepare inputs
        if getattr(self, "fixed_input", False):
            # Concatenate qpos + qvel, then pad/trim to self.input_size
            feats = np.concatenate([np.array(data.qpos), np.array(data.qvel)])
            if feats.shape[0] < self.input_size:
                feats = np.pad(feats, (0, self.input_size - feats.shape[0]))
            else:
                feats = feats[: self.input_size]
            inputs = feats
        else:
            # Default: use qpos directly
            inputs = data.qpos

        # Run the inputs through the lays of the network.
        layer1 = np.tanh(np.dot(inputs, self.weights[0]))
        layer2 = np.tanh(np.dot(layer1, self.weights[1]))
        outputs = np.tanh(np.dot(layer2, self.weights[2]))

        # Scale the outputs
        return outputs * np.pi


# ========= Co-evolution support ========= #
def _nn_param_sizes_fixed(inp: int, hid: int, out: int) -> int:
    return inp * hid + hid * hid + hid * out


def _split_genome(
    x: np.ndarray,
    *,
    body_gene_len: int,
    ctrl_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    assert x.ndim == 1
    xb = x[:body_gene_len]
    xc = x[body_gene_len : body_gene_len + ctrl_len]
    return xb, xc


def _build_robot_from_body_genes(xb: np.ndarray) -> tuple[CoreModule, dict[str, Any]]:
    """Build robot and return body + diversity metrics."""
    logger.info("Building robot from body genes...")
    # Expect xb to be 3*GENOTYPE_SIZE flat [type, conn, rot]
    if xb.size < 3 * GENOTYPE_SIZE:
        # pad gracefully
        pad = np.zeros(3 * GENOTYPE_SIZE - xb.size, dtype=xb.dtype)
        xb = np.concatenate([xb, pad])
        logger.debug(f"Padded body genes to {xb.size} elements")
    # Map real-valued genes to probability space via sigmoid
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    type_p = _sigmoid(xb[0:GENOTYPE_SIZE]).astype(np.float32)
    conn_p = _sigmoid(xb[GENOTYPE_SIZE : 2 * GENOTYPE_SIZE]).astype(np.float32)
    rot_p = _sigmoid(xb[2 * GENOTYPE_SIZE : 3 * GENOTYPE_SIZE]).astype(np.float32)
    
    # Build genotype and robot
    genotype = [type_p, conn_p, rot_p]
    
    # Use NUM_OF_MODULES directly to maximize module count
    base_modules = NUM_OF_MODULES
    nde = NeuralDevelopmentalEncoding(number_of_modules=base_modules)
    p_matrices = nde.forward(genotype)
    hpd = HighProbabilityDecoder(base_modules)
    robot_graph = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )
    
    # Count diversity metrics before hinge insertion
    num_blocks_before = sum(1 for n in robot_graph.nodes() if robot_graph.nodes[n].get("type") == "BRICK")
    num_edges_before = robot_graph.number_of_edges()
    
    # Ensure leaves are BRICK
    for n in list(robot_graph.nodes()):
        if robot_graph.out_degree(n) == 0 and robot_graph.nodes[n].get("type") != "CORE":
            robot_graph.nodes[n]["type"] = "BRICK"
    
    # Insert hinges deterministically
    robot_graph = insert_hinges_between_blocks(robot_graph, conn_p_matrices=p_matrices[1])
    
    # Count final metrics
    num_hinges = sum(1 for n in robot_graph.nodes() if robot_graph.nodes[n].get("type") == "HINGE")
    num_modules_total = robot_graph.number_of_nodes()
    
    # Normalize faces
    for _u, _v, attrs in robot_graph.edges(data=True):
        face_raw = attrs.get("face", "FRONT")
        try:
            face_name = face_raw.name  # type: ignore[attr-defined]
        except Exception:
            face_name = str(face_raw).upper()
        attrs["face"] = face_name
    
    robot = construct_mjspec_from_graph(robot_graph)
    
    diversity_metrics = {
        "num_hinges": num_hinges,
        "num_modules": num_modules_total,
        "num_blocks_before": num_blocks_before,
        "num_edges": num_edges_before,
    }
    
    logger.info(f"Robot built: {num_modules_total} modules, {num_hinges} hinges, {num_blocks_before} blocks")
    
    return robot, diversity_metrics


def _make_controller_for_robot_with_padding(robot: CoreModule, xc: np.ndarray) -> "NN":
    # Create NN sized to the robot
    nn = NN(robot)
    inp, hid, out = COEV_INPUT_SIZE, HIDDEN_SIZE, min(COEV_MAX_ACTUATORS, nn.output_size)
    # Build weight shapes for fixed IO, then adapt outputs to robot
    D = _nn_param_sizes_fixed(inp, hid, out)
    if xc.size < D:
        xc = np.concatenate([xc, np.zeros(D - xc.size)])
    w1, w2, w3 = _unflatten_weights(xc[:D], inp, hid, out)

    # If robot has fewer actuators than out, slice; if more, tile/truncate
    if nn.output_size <= out:
        w3_adapt = w3[:, : nn.output_size]
    else:
        # Expand columns by tiling to match actual actuators
        reps = int(np.ceil(nn.output_size / out))
        w3_tiled = np.tile(w3, (1, reps))
        w3_adapt = w3_tiled[:, : nn.output_size]

    # Inputs: use first COEV_INPUT_SIZE entries of qpos/qvel via NN.forward reading from data.qpos
    # Our NN class uses data.qpos length to set input_size; override its sizes to fixed input
    nn.input_size = inp
    nn.hidden_size = hid
    nn.output_size = nn.output_size  # unchanged
    # For w1, we need matching input dimension; tile/trim rows appropriately
    if w1.shape[0] != inp:
        # reshape by tiling rows to reach inp
        reps = int(np.ceil(inp / w1.shape[0]))
        w1 = np.tile(w1, (reps, 1))[:inp, :]
    nn.set_controller_weights((w1, w2, w3_adapt))
    nn.fixed_input = True
    return nn


def _evaluate_coevolution_candidate_worker(
    args: tuple[np.ndarray, int, set[str]],
) -> tuple[float, str, bool]:
    """Worker function for parallel co-evolution evaluation.
    
    Returns (fitness, morphology_signature, is_novel).
    Note: seen_morphologies is passed per-candidate to avoid shared state issues.
    """
    x, generation, local_seen = args
    try:
        body_len = 3 * GENOTYPE_SIZE
        ctrl_len = _nn_param_sizes_fixed(COEV_INPUT_SIZE, HIDDEN_SIZE, COEV_MAX_ACTUATORS)
        xb, xc = _split_genome(x, body_gene_len=body_len, ctrl_len=ctrl_len)
        
        robot, diversity_metrics = _build_robot_from_body_genes(xb)
        nn = _make_controller_for_robot_with_padding(robot, xc)
        
        # Quick controller polish (simplified for parallel - no CMA polish in parallel workers)
        # Use random perturbation polish only
        best_local_nn = nn
        best_local_fit = evaluate(
            robot,
            best_local_nn,
            plot_and_record=False,
            duration_override=COEV_POLISH_EVAL_DURATION,
        )
        for _ in range(COEV_POLISH_STEPS):
            w1, w2, w3 = best_local_nn.weights
            for _j in range(COEV_POLISH_POPSIZE):
                c1 = w1 + RNG.normal(0.0, COEV_POLISH_SIGMA, size=w1.shape)
                c2 = w2 + RNG.normal(0.0, COEV_POLISH_SIGMA, size=w2.shape)
                c3 = w3 + RNG.normal(0.0, COEV_POLISH_SIGMA, size=w3.shape)
                trial = _make_controller_for_robot_with_padding(robot, _flatten_weights((c1, c2, c3)))
                fit_try = evaluate(
                    robot,
                    trial,
                    plot_and_record=False,
                    duration_override=COEV_POLISH_EVAL_DURATION,
                )
                if fit_try > best_local_fit:
                    best_local_fit = fit_try
                    best_local_nn = trial
        nn = best_local_nn
        
        # Repeated evaluations for noise reduction
        fitnesses_raw: list[float] = []
        for _i in range(COEV_NUM_EVALS_PER_CANDIDATE):
            fit = evaluate(robot, nn, plot_and_record=False)
            fitnesses_raw.append(fit)
        
        # Use median for robustness
        raw_fitness = float(np.median(fitnesses_raw))
        
        # Compute diversity info (actual novelty will be determined in main thread)
        morph_sig = f"{diversity_metrics['num_hinges']}_{diversity_metrics['num_modules']}"
        is_novel = morph_sig not in local_seen
        
        # Return negative because CMA minimizes
        return -(raw_fitness), morph_sig, is_novel
        
    except Exception as e:
        logger.error(f"[Gen {generation}] Error in parallel coevolution evaluation: {e}")
        return float('inf'), "error", False


def _objective_coevolution(x: np.ndarray, seen_morphologies: set[str], generation: int = 0) -> float:
    # Joint genome: [body_genes (3*GENOTYPE_SIZE), controller_weights (fixed-D)]
    body_len = 3 * GENOTYPE_SIZE
    ctrl_len = _nn_param_sizes_fixed(COEV_INPUT_SIZE, HIDDEN_SIZE, COEV_MAX_ACTUATORS)
    xb, xc = _split_genome(x, body_gene_len=body_len, ctrl_len=ctrl_len)
    logger.debug(f"[Gen {generation}] Evaluating co-evolution candidate...")
    robot, diversity_metrics = _build_robot_from_body_genes(xb)
    nn = _make_controller_for_robot_with_padding(robot, xc)

    # Quick controller polish using CMA-ES if enabled
    if COEV_POLISH_USE_CMA and cma is not None:
        logger.debug(f"[Gen {generation}] Polishing controller with mini-CMA-ES...")
        # Mini-CMA on controller weights only
        ctrl_weights_flat = _flatten_weights(nn.weights)
        polish_opts: dict[str, Any] = {
            "seed": int(RNG.integers(0, 10_000_000)),
            "popsize": COEV_POLISH_POPSIZE,
            "maxiter": COEV_POLISH_STEPS,
            "verbose": -9,
        }
        polish_es = cma.CMAEvolutionStrategy(ctrl_weights_flat.tolist(), COEV_POLISH_SIGMA, polish_opts)
        best_polish_fit = -math.inf
        best_polish_weights = ctrl_weights_flat.copy()
        
        polish_gen = 0
        while not polish_es.stop() and polish_gen < COEV_POLISH_STEPS:
            polish_gen += 1
            X_polish = polish_es.ask()
            polish_fitnesses: list[float] = []
            for xi_polish in X_polish:
                trial_nn = _make_controller_for_robot_with_padding(robot, np.asarray(xi_polish, dtype=float))
                fit_try = evaluate(
                    robot,
                    trial_nn,
                    plot_and_record=False,
                    duration_override=COEV_POLISH_EVAL_DURATION,
                )
                polish_fitnesses.append(fit_try)
                if fit_try > best_polish_fit:
                    best_polish_fit = fit_try
                    best_polish_weights = np.asarray(xi_polish, dtype=float).copy()
            polish_es.tell(X_polish, polish_fitnesses)
        
        logger.debug(f"[Gen {generation}] Polish completed: best_fit={best_polish_fit:.4f}")
        nn = _make_controller_for_robot_with_padding(robot, best_polish_weights)
    else:
        # Fallback: random perturbation polish
        logger.debug(f"[Gen {generation}] Polishing controller with random perturbations...")
        best_local_nn = nn
        best_local_fit = evaluate(
            robot,
            best_local_nn,
            plot_and_record=False,
            duration_override=COEV_POLISH_EVAL_DURATION,
        )
        for _ in range(COEV_POLISH_STEPS):
            w1, w2, w3 = best_local_nn.weights
            for _j in range(COEV_POLISH_POPSIZE):
                c1 = w1 + RNG.normal(0.0, COEV_POLISH_SIGMA, size=w1.shape)
                c2 = w2 + RNG.normal(0.0, COEV_POLISH_SIGMA, size=w2.shape)
                c3 = w3 + RNG.normal(0.0, COEV_POLISH_SIGMA, size=w3.shape)
                trial = _make_controller_for_robot_with_padding(robot, _flatten_weights((c1, c2, c3)))
                fit_try = evaluate(
                    robot,
                    trial,
                    plot_and_record=False,
                    duration_override=COEV_POLISH_EVAL_DURATION,
                )
                if fit_try > best_local_fit:
                    best_local_fit = fit_try
                    best_local_nn = trial
        logger.debug(f"[Gen {generation}] Polish completed: best_fit={best_local_fit:.4f}")
        nn = best_local_nn

    # Repeated evaluations for noise reduction
    fitnesses_raw: list[float] = []
    for i in range(COEV_NUM_EVALS_PER_CANDIDATE):
        fit = evaluate(robot, nn, plot_and_record=False)
        fitnesses_raw.append(fit)
        logger.debug(f"[Gen {generation}] Evaluation {i+1}/{COEV_NUM_EVALS_PER_CANDIDATE}: fitness={fit:.4f}")
    
    # Use median for robustness
    raw_fitness = float(np.median(fitnesses_raw))
    
    # Compute diversity bonus
    morph_sig = f"{diversity_metrics['num_hinges']}_{diversity_metrics['num_modules']}"
    is_novel = morph_sig not in seen_morphologies
    seen_morphologies.add(morph_sig)
    diversity_bonus = COEV_DIVERSITY_WEIGHT if is_novel else 0.0
    
    logger.debug(f"[Gen {generation}] Morph signature: {morph_sig}, Novel: {is_novel}, Diversity bonus: {diversity_bonus:.2f}")
    
    # Combined fitness (negative because CMA minimizes)
    return -(raw_fitness + diversity_bonus)


def train_coevolution_with_cma(
    *,
    population_size: int | None = None,
    sigma0: float | None = None,
    max_generations: int | None = None,
) -> tuple[CoreModule, "NN", float, np.ndarray]:
    logger.info("="*80)
    logger.info("STARTING CO-EVOLUTION WITH CMA-ES")
    logger.info("="*80)
    
    if cma is None:
        raise RuntimeError("pycma is not available. Please install 'cma'.")

    pop = population_size or CMA_POPSIZE
    s0 = sigma0 or CMA_SIGMA0
    gens = max_generations or CMA_MAX_GENERATIONS

    logger.info(f"Population size: {pop}")
    logger.info(f"Initial sigma: {s0}")
    logger.info(f"Max generations: {gens}")
    logger.info(f"Diversity weight: {COEV_DIVERSITY_WEIGHT}")
    logger.info(f"Evaluations per candidate: {COEV_NUM_EVALS_PER_CANDIDATE}")

    body_len = 3 * GENOTYPE_SIZE
    ctrl_len = _nn_param_sizes_fixed(COEV_INPUT_SIZE, HIDDEN_SIZE, COEV_MAX_ACTUATORS)
    D = body_len + ctrl_len
    logger.info(f"Total genome size: {D} (body={body_len}, controller={ctrl_len})")

    # Initialize genome around small random values
    x0 = RNG.normal(0.0, 0.1, size=D)
    opts: dict[str, Any] = {
        "seed": int(RNG.integers(0, 10_000_000)),
        "popsize": pop,
        "maxiter": gens,
        "verbose": -9,
    }
    logger.info("Initializing CMA-ES strategy...")
    es = cma.CMAEvolutionStrategy(x0.tolist(), s0, opts)

    best_val = math.inf
    best_x = x0.copy()
    gen = 0
    
    # Track seen morphologies for diversity bonus
    seen_morphologies: set[str] = set()
    
    logger.info(f"Starting co-evolution optimization loop with {NUM_PARALLEL_WORKERS} parallel workers...")
    logger.info("-"*80)
    
    while not es.stop():
        gen += 1
        logger.info(f"="*80)
        logger.info(f"GENERATION {gen}/{gens}")
        logger.info(f"="*80)
        logger.info(f"Asking CMA-ES for {pop} candidates...")
        X = es.ask()
        
        # Evaluate population in parallel
        if NUM_PARALLEL_WORKERS > 1:
            # Prepare arguments for parallel evaluation
            # Each worker gets a copy of seen morphologies to check novelty
            eval_args = [(np.asarray(xi, dtype=float), gen, seen_morphologies.copy()) for xi in X]
            
            with Pool(NUM_PARALLEL_WORKERS) as pool:
                results = pool.map(_evaluate_coevolution_candidate_worker, eval_args)
            
            # Process results and update seen morphologies
            fitnesses: list[float] = []
            for (raw_fit, morph_sig, _is_novel), xi in zip(results, X):
                # Check actual novelty against main thread's seen_morphologies
                actually_novel = morph_sig not in seen_morphologies
                seen_morphologies.add(morph_sig)
                
                # Apply diversity bonus
                diversity_bonus = COEV_DIVERSITY_WEIGHT if actually_novel else 0.0
                final_fitness = raw_fit - diversity_bonus  # raw_fit is already negative
                fitnesses.append(final_fitness)
                
                if final_fitness < best_val:
                    best_val = final_fitness
                    best_x = np.asarray(xi, dtype=float).copy()
                    logger.info(f"[Gen {gen}] NEW BEST found! fitness={-best_val:.4f}")
        else:
            # Sequential fallback
            fitnesses = []
            for idx, xi in enumerate(X):
                logger.debug(f"[Gen {gen}] Evaluating candidate {idx+1}/{len(X)}")
                val = _objective_coevolution(np.asarray(xi, dtype=float), seen_morphologies, generation=gen)
                fitnesses.append(val)
                if val < best_val:
                    best_val = val
                    best_x = np.asarray(xi, dtype=float).copy()
                    logger.info(f"[Gen {gen}] NEW BEST found! fitness={-best_val:.4f}")
                best_x = np.asarray(xi, dtype=float).copy()
                logger.info(f"[Gen {gen}] NEW BEST found! fitness={-best_val:.4f}")
        
        es.tell(X, fitnesses)
        
        if gen % LOG_EVERY_N_GEN == 0:
            logger.info(
                f"[CoEv Gen {gen}] best_distance={-best_val:.4f} "
                f"median={-float(np.median(fitnesses)):.4f} pop={len(fitnesses)} "
                f"unique_morphs={len(seen_morphologies)}",
            )
            console.log(
                f"[CoEv] gen={gen} best_distance={-best_val:.4f} "
                f"median={-float(np.median(fitnesses)):.4f} pop={len(fitnesses)} "
                f"unique_morphs={len(seen_morphologies)}",
            )
        if gen >= gens:
            break

    logger.info("-"*80)
    logger.info(f"Co-evolution optimization completed after {gen} generations")
    logger.info(f"Total unique morphologies discovered: {len(seen_morphologies)}")

    # Decode best into robot and controller
    logger.info("Decoding best genome into robot and controller...")
    xb, xc = _split_genome(best_x, body_gene_len=body_len, ctrl_len=ctrl_len)
    robot, diversity_metrics = _build_robot_from_body_genes(xb)
    nn = _make_controller_for_robot_with_padding(robot, xc)
    best_fitness = -best_val
    
    logger.info(
        f"[CoEv] Final best morphology: hinges={diversity_metrics['num_hinges']} "
        f"modules={diversity_metrics['num_modules']} fitness={best_fitness:.4f}",
    )
    console.log(
        f"[CoEv] Final best morphology: hinges={diversity_metrics['num_hinges']} "
        f"modules={diversity_metrics['num_modules']} fitness={best_fitness:.4f}",
    )
    
    logger.info("="*80)
    logger.info("CO-EVOLUTION COMPLETED")
    logger.info("="*80)
    
    return robot, nn, best_fitness, best_x
    nn = _make_controller_for_robot_with_padding(robot, xc)
    best_fitness = -best_val
    return robot, nn, best_fitness, best_x


def run(
    model: mj.MjModel,
    data: mj.MjData,
    duration: int = 15,
    mode: str = "viewer",
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
            path_to_video_folder = str(DATA / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)
            print(f"Saving video to: {path_to_video_folder}")
            # Render with video recorder
            video_renderer(
                model,
                data,
                duration=duration,
                video_recorder=video_recorder,
            )
        case "tracking":
            # This records a video of the simulation
            path_to_video_folder = str(DATA / "videos")
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
        case _:
            # Unknown mode: default to simple runner
            simple_runner(model, data, duration=duration)


def evaluate(
    robot: CoreModule,
    nn: NN,
    *,
    plot_and_record: bool = False,
    duration_override: int | None = None,
) -> float:
    logger.debug("Starting evaluation...")
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
    logger.debug("Spawning robot in simulation environment...")
    model, data, world = quick_spawn(robot)

    # Pass the model and data to the tracker
    controller.tracker.setup(world.spec, data)

    # Set the control callback function
    mj.set_mjcb_control(controller.set_control)

    # Choose the mode
    duration = EVALUATION_DURATION if duration_override is None else duration_override
    logger.debug(f"Running simulation for {duration} seconds...")
    if plot_and_record is True:
        logger.info("Running simulation with video recording...")
        # Run the simulation
        run(
            model,
            data,
            duration=duration,
            mode="video",
        )

        # Show the tracked history
        logger.info("Generating trajectory plot...")
        show_xpos_history(tracker.history["xpos"][0])
    else:
        # Run the simulation
        run(
            model,
            data,
            duration=duration,
            mode="simple",
        )

    # Calculate and print the fitness
    fitness = fitness_function(tracker.history["xpos"][0])
    logger.debug(f"Evaluation complete. Fitness: {fitness:.4f}")
    return fitness


def main() -> None:
    """Entry point: run co-evolution of body and controller with CMA-ES."""
    logger.info("="*80)
    logger.info("MAIN ENTRY POINT")
    logger.info("="*80)
    logger.info("Starting co-evolution experiment...")
    
    try:
        robot, best_nn, best_fitness, _ = train_coevolution_with_cma()
    except RuntimeError as e:
        logger.error(f"RuntimeError: {e}")
        console.log(str(e))
        return
    except Exception as e:
        logger.exception(f"Unexpected error during co-evolution: {e}")
        console.log(f"Error: {e}")
        return
    
    logger.info(f"Co-evolution finished. Best fitness: {best_fitness:.4f}")
    console.log(f"Co-evolution finished. Best fitness: {best_fitness:.4f}")
    
    logger.info("Running final evaluation with video recording...")
    final_fitness = evaluate(robot, best_nn, plot_and_record=True)
    logger.info(f"Final evaluation fitness: {final_fitness:.4f}")
    
    logger.info("="*80)
    logger.info(f"EXECUTION COMPLETED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log saved to: {log_file_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
