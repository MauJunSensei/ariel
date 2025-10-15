"""Assignment 3: Evolve body via NDE+HPD GA, then CMA-ES controller.

Phases
1) Body Evolution (NDE inputs + High-Probability Decoder):
   - Uses the GA loop from assignments/A3_v2/A3.py to evolve three real-valued
     chromosomes that feed the Neural Developmental Encoding (NDE) network
     and are decoded via the HighProbabilityDecoder (HPD) into a morphology.
   - Selection = elitism + tournament; mutation applied to NDE chromosomes.
   - Saves best morphology graph (JSON + PNG) to __data__/A3_merge/.

2) Controller Evolution (CMA-ES sinusoidal CPG):
   - On the selected best morphology, optimize a sinusoidal controller with
     CMA-ES (per-actuator amplitudes and phases + global frequency).
   - Saves best controller parameters (best_theta.npy) and can visualize.

"""

from __future__ import annotations

# Standard library
from pathlib import Path
from typing import Any, Literal
from concurrent.futures import ProcessPoolExecutor, as_completed

import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# Local libraries
from ariel import console
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena, SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
    draw_graph,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding

# Ensure we can import the shared config from A3_v2
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
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
    NUM_OF_MODULES,
    ENFORCE_GUARDRAILS,
    REQUIRE_MIN_HINGES,
    REQUIRE_MAX_HINGES,
    MUTATION_STRENGTH,
    CROSSOVER_RATE,
    PHASE1_NUM_SAMPLES,
    PHASE1_SIM_SECONDS,
    PHASE1_SPAWN_POS,
    PHASE1_REQUIRE_HINGE_FIRST_LIMB,
    PHASE1_TOTAL_HINGE_MIN,
    PHASE1_TOTAL_HINGE_MAX,
    PHASE2_TOP_K,
    PHASE2_MIN_FORWARD_DISTANCE,
    GA_BASELINE_AMPLITUDE,
    GA_BASELINE_FREQUENCY,
    GA_CTRL_STEP,
    GA_SAVE_STEP,
    GA_USE_RANDOM_CONTROLLER,
    CMA_CONTROLLER_TYPE,
    CMA_MAX_EVALS,
    CMA_POPSIZE,
    CMA_SIGMA0,
)

try:
    import cma  # Controller CMA-ES
except Exception:  # pragma: no cover
    cma = None  # type: ignore


# Types
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# RNG
SEED = 42
RNG = np.random.default_rng(SEED)

# Data dirs
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path(__file__).parent
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# World constants
SPAWN_POS = [-0.8, 0, 0.1]


class Individual:
    def __init__(
        self,
        body_genes: list[np.ndarray],
    ) -> None:
        self.body_genes = body_genes
        self.graph = None  # networkx.DiGraph[Any] | None
        self.fitness: float | None = None


def generate_body_genes() -> list[np.ndarray]:
    # Sample in [0,1] to avoid saturating the untrained NDE MLP
    type_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
    conn_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
    rot_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
    return [type_p_genes, conn_p_genes, rot_p_genes]


def generate_body(ind: Individual, genes: list[np.ndarray] | None = None) -> None:
    import networkx as nx
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_types, p_conns, p_rots = nde.forward(genes if genes is not None else ind.body_genes)
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    g = hpd.probability_matrices_to_graph(p_types, p_conns, p_rots)
    
    if ENFORCE_GUARDRAILS:
        # Drop NONE nodes
        to_remove = [n for n, d in g.nodes(data=True) if str(d.get("type", "")).upper() == "NONE"]
        if to_remove:
            g.remove_nodes_from(to_remove)
        if g.number_of_nodes() == 0:
            ind.graph = None
            return
        
        # Keep only component reachable from CORE
        core_nodes = [n for n, d in g.nodes(data=True) if str(d.get("type", "")).upper() == "CORE"]
        if core_nodes:
            core_n = core_nodes[0]
            reachable = nx.descendants(g, core_n) | {core_n}
            g = g.subgraph(reachable).copy()
        if g.number_of_nodes() == 0:
            ind.graph = None
            return
        
        # Require minimum and maximum hinges
        num_hinges = sum(1 for _, d in g.nodes(data=True) if str(d.get("type", "")).upper() == "HINGE")
        if num_hinges < REQUIRE_MIN_HINGES or num_hinges > REQUIRE_MAX_HINGES:
            ind.graph = None
            return
    
    ind.graph = g


def fitness_function(history: list[tuple[float, float, float]]) -> float:
    if not history:
        return -1e6
    pos = np.array(history)
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    course_len = max(FIT_SECTION_THRESHOLDS[-1], 1.0)
    fwd = np.clip(np.max(x) / course_len, 0.0, 1.0)
    lateral = float(np.mean(np.abs(y))) if y.size > 0 else 0.0
    dz = np.diff(z) if z.size > 1 else np.array([0.0])
    bounce = float(np.mean(np.abs(dz)))
    dx = np.diff(x) if x.size > 1 else np.array([0.0])
    ddx = np.diff(dx) if dx.size > 1 else np.array([0.0])
    smooth = float(np.mean(np.abs(ddx)))
    bonus = 0.0
    xmax = float(np.max(x))
    for thr, b in zip(FIT_SECTION_THRESHOLDS, FIT_SECTION_BONUSES):
        if xmax >= thr:
            bonus += b
    score = (
        FIT_W_FORWARD * fwd
        - FIT_W_LATERAL * lateral
        - FIT_W_BOUNCE * bounce
        - FIT_W_SMOOTH * smooth
        + bonus
    )
    return float(score)


def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
    *,
    amps: np.ndarray,
    phases: np.ndarray,
    freq: float,
) -> npt.NDArray[np.float64]:
    t = float(data.time)
    targets = amps * np.sin(2 * np.pi * freq * t + phases)
    return np.clip(targets, -np.pi / 2, np.pi / 2)


def experiment(
    robot: Any,
    controller: Controller,
    duration: int = 15,
    mode: ViewerTypes = "simple",
) -> None:
    mj.set_mjcb_control(None)
    world = OlympicArena()
    # Use bounding box correction as before; small gap not necessary
    world.spawn(robot.spec, position=SPAWN_POS, correct_collision_with_floor=True)
    model = world.spec.compile()
    data = mj.MjData(model)
    # Increase stability: smaller timestep, Euler integrator, add damping
    # Keep default integrator/timestep/damping (previous stable config)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    controller.tracker.setup(world.spec, data)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    mj.set_mjcb_control(lambda m, d: controller.set_control(m, d, *args, **kwargs))
    match mode:
        case "simple":
            simple_runner(model, data, duration=duration)
        case "frame":
            single_frame_renderer(model, data, save=True, save_path=str(DATA / "robot.png"))
        case "video":
            video_recorder = VideoRecorder(output_folder=str(DATA / "videos"))
            video_renderer(model, data, duration=duration, video_recorder=video_recorder)
        case "launcher":
            viewer.launch(model=model, data=data)
        case "no_control":
            mj.set_mjcb_control(None)
            viewer.launch(model=model, data=data)


def evaluate(individual: Individual) -> float:
    if individual.graph is None:
        generate_body(individual)
    if individual.graph is None:
        return -1e9
    robot = construct_mjspec_from_graph(individual.graph)

    # Fast reject actuator-less bodies
    try:
        model_chk = robot.spec.compile()
        if model_chk.nu == 0:
            return -1e9
    except Exception:
        return -1e9

    # Baseline sinusoidal controller to probe morphology viability (headless)
    # Simple uniform oscillation like A3_clean.py for consistent evaluation
    def baseline_cb(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
        t = float(data.time)
        targets = GA_BASELINE_AMPLITUDE * np.sin(2 * np.pi * GA_BASELINE_FREQUENCY * t)
        return np.clip(np.full(model.nu, targets), -np.pi / 2, np.pi / 2)

    ctrl = Controller(
        controller_callback_function=baseline_cb,
        time_steps_per_ctrl_step=GA_CTRL_STEP,
        time_steps_per_save=GA_SAVE_STEP,
        tracker=Tracker(mj.mjtObj.mjOBJ_BODY, "core"),
    )
    # Force simple (no rendering) for faster GA evaluation
    experiment(robot=robot, controller=ctrl, duration=RUN_DURATION, mode="simple")
    return fitness_function(ctrl.tracker.history["xpos"][0])


def evaluate_all(population: list[Individual]) -> None:
    if not PARALLEL_EVAL:
        for ind in population:
            ind.fitness = evaluate(ind)
        return
    with ProcessPoolExecutor() as ex:
        futures = {ex.submit(evaluate, ind): ind for ind in population}
        for f in as_completed(futures):
            ind = futures[f]
            try:
                ind.fitness = f.result()
            except Exception as e:
                console.log(f"Evaluation failed: {e}")
                ind.fitness = -1e9


def select_survivors(population: list[Individual], num_survivors: int) -> list[Individual]:
    valid = [ind for ind in population if ind.fitness is not None]
    if not valid:
        raise ValueError("No individuals with valid fitness to select from.")
    ranked = sorted(valid, key=lambda ind: float(ind.fitness or -1e9), reverse=True)
    elites = ranked[: min(ELITISM_K, num_survivors)]
    survivors: list[Individual] = elites.copy()
    pool: list[Individual] = [ind for ind in ranked if ind not in set(survivors)]
    while len(survivors) < num_survivors and pool:
        k = min(len(pool), TOURNAMENT_SIZE)
        idxs = RNG.choice(len(pool), size=k, replace=False)
        tourney = [pool[i] for i in idxs]
        winner = max(tourney, key=lambda ind: float(ind.fitness or -1e9))
        survivors.append(winner)
        pool.remove(winner)
    return survivors


def crossover(parent1: Individual, parent2: Individual) -> Individual:
    """Uniform crossover: each gene randomly chosen from parent1 or parent2."""
    child_genes = []
    for arr1, arr2 in zip(parent1.body_genes, parent2.body_genes):
        mask = RNG.random(arr1.shape) < 0.5
        child_arr = np.where(mask, arr1, arr2).copy()
        child_genes.append(child_arr)
    child = Individual(body_genes=child_genes)
    generate_body(child, genes=child_genes)
    return child


def mutate(individual: Individual) -> None:
    # Mutate inside [0,1] with configurable strength
    for i, arr in enumerate(individual.body_genes):
        arr2 = arr.copy()
        mask = RNG.random(arr2.shape) < MUTATION_RATE
        if mask.any():
            noise = RNG.normal(loc=0.0, scale=MUTATION_STRENGTH, size=arr2.shape).astype(arr2.dtype)
            arr2[mask] = np.clip(arr2[mask] + noise[mask], 0.0, 1.0)
        individual.body_genes[i] = arr2
    generate_body(individual, genes=individual.body_genes)
    individual.fitness = None


def _count_hinges(g: Any) -> int:
    return sum(1 for _, d in g.nodes(data=True) if str(d.get("type", "")).upper() == "HINGE")


def _core_children(g: Any) -> list[tuple[int, dict[str, Any]]]:
    core_nodes = [n for n, d in g.nodes(data=True) if str(d.get("type", "")).upper() == "CORE"]
    if not core_nodes:
        return []
    core = core_nodes[0]
    return [(v, g.nodes[v]) for _, v in g.out_edges(core)]


def _is_hinge_first_limb(g: Any) -> bool:
    """Return True if at least one limb from core begins with a hinge, or a hinge appears at depth 2.
    This is less strict than requiring all limbs to start with a hinge.
    """
    children = _core_children(g)
    if not children:
        return False
    # Accept if any direct child is a hinge
    for _, attrs in children:
        if str(attrs.get("type", "")).upper() == "HINGE":
            return True
    # Otherwise, check grandchildren one step away
    for node_id, _ in children:
        for _, gc_id in g.out_edges(node_id):
            gc_attrs = g.nodes[gc_id]
            if str(gc_attrs.get("type", "")).upper() == "HINGE":
                return True
    return False


def _graph_is_viable_for_motion(g: Any) -> tuple[bool, str]:
    num_hinges = _count_hinges(g)
    if num_hinges < PHASE1_TOTAL_HINGE_MIN:
        return False, f"too_few_hinges:{num_hinges}"
    if num_hinges > PHASE1_TOTAL_HINGE_MAX:
        return False, f"too_many_hinges:{num_hinges}"
    if PHASE1_REQUIRE_HINGE_FIRST_LIMB and not _is_hinge_first_limb(g):
        return False, "no_hinge_first_limb"
    return True, "ok"


def _count_module_types(g: Any) -> tuple[int, int, int]:
    """Count CORE, HINGE, BRICK modules in graph."""
    num_core = sum(1 for _, d in g.nodes(data=True) if str(d.get("type", "")).upper() == "CORE")
    num_hinge = sum(1 for _, d in g.nodes(data=True) if str(d.get("type", "")).upper() == "HINGE")
    num_brick = sum(1 for _, d in g.nodes(data=True) if str(d.get("type", "")).upper() == "BRICK")
    return num_core, num_hinge, num_brick


def _evaluate_displacement(core: Any, seconds: float) -> tuple[float, float]:
    # Screen on SimpleFlat for stability and speed
    world = SimpleFlatWorld()
    try:
        world.spawn(core.spec, position=PHASE1_SPAWN_POS, correct_collision_with_floor=True)
        model = world.spec.compile()
    except Exception:
        return -np.inf, -np.inf
    data = mj.MjData(model)
    # Deterministic gentle multi-phase oscillation
    phases = None
    def cb(m: mj.MjModel, d: mj.MjData) -> None:
        nonlocal phases
        if phases is None:
            phases = np.linspace(0.0, 2 * np.pi, m.nu, endpoint=False)
        t = float(d.time)
        targets = GA_BASELINE_AMPLITUDE * np.sin(2 * np.pi * GA_BASELINE_FREQUENCY * t + phases)
        d.ctrl[:] = np.clip(targets, -np.pi / 2, np.pi / 2)
    if model.nu == 0:
        return -np.inf, -np.inf
    mj.set_mjcb_control(cb)
    dt = model.opt.timestep
    steps = max(1, int(seconds / dt))
    try:
        core_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "core")
        x0 = float(data.xpos[core_bid, 0]); y0 = float(data.xpos[core_bid, 1])
    except Exception:
        x0 = float(data.qpos[0]) if len(data.qpos) > 0 else 0.0
        y0 = float(data.qpos[1]) if len(data.qpos) > 1 else 0.0
    for _ in range(steps):
        mj.mj_step(model, data)
    try:
        core_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "core")
        x1 = float(data.xpos[core_bid, 0]); y1 = float(data.xpos[core_bid, 1])
    except Exception:
        x1 = float(data.qpos[0]) if len(data.qpos) > 0 else 0.0
        y1 = float(data.qpos[1]) if len(data.qpos) > 1 else 0.0
    return x1 - x0, float(np.hypot(x1 - x0, y1 - y0))


def sample_bodies(num_samples: int) -> list[Individual]:
    console.rule("Phase 1: Random Morphology Sampling (NDE→HPD)")
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    individuals: list[Individual] = []
    kept = 0
    for i in range(num_samples):
        genes = [RNG.random(GENOTYPE_SIZE).astype(np.float32) for _ in range(3)]
        ind = Individual(body_genes=genes)
        try:
            p_types, p_conns, p_rots = nde.forward(genes)
            g = hpd.probability_matrices_to_graph(p_types, p_conns, p_rots)
        except Exception:
            continue
        # Guardrails similar to generate_body
        if ENFORCE_GUARDRAILS:
            import networkx as nx
            to_remove = [n for n, d in g.nodes(data=True) if str(d.get("type", "")).upper() == "NONE"]
            if to_remove:
                g.remove_nodes_from(to_remove)
            if g.number_of_nodes() == 0:
                console.log({"reject": "all_none"}); continue
            core_nodes = [n for n, d in g.nodes(data=True) if str(d.get("type", "")).upper() == "CORE"]
            if core_nodes:
                core_n = core_nodes[0]
                reachable = nx.descendants(g, core_n) | {core_n}
                g = g.subgraph(reachable).copy()
            if g.number_of_nodes() == 0:
                console.log({"reject": "not_core_reachable"}); continue
        # Motion viability filters
        ok, reason = _graph_is_viable_for_motion(g)
        if not ok:
            if reason == "no_hinge_first_limb":
                # brief debug types of core children to aid tuning
                cc = [(nid, str(g.nodes[nid].get("type", "")).upper()) for nid, _ in _core_children(g)]
                console.log({"reject": reason, "core_children": cc})
            else:
                console.log({"reject": reason})
            continue
        ind.graph = g
        individuals.append(ind)
        kept += 1
        if kept % 50 == 0:
            console.log({"kept": kept, "sampled": i + 1})
    console.log({"sampled_total": num_samples, "kept": kept})
    return individuals


def run_body_evolution() -> tuple[float, list[Individual]]:
    # Phase 1 replaced by assignment-compliant random sampling + viability check
    inds = sample_bodies(PHASE1_NUM_SAMPLES)
    # Evaluate displacement with short sim
    scored: list[tuple[float, float, Individual]] = []  # (dx, dist, ind)
    records: list[tuple[int, float, float, int, int, int]] = []
    
    for idx, ind in enumerate(inds):
        if ind.graph is None:
            records.append((idx, -np.inf, -np.inf, 0, 0, 0))
            continue
        try:
            core = construct_mjspec_from_graph(ind.graph)
        except Exception:
            records.append((idx, -np.inf, -np.inf, 0, 0, 0))
            continue
        dx, dist = _evaluate_displacement(core, PHASE1_SIM_SECONDS)
        scored.append((dx, dist, ind))
        c, h, b = _count_module_types(ind.graph)
        records.append((idx, float(dx), float(dist), c, h, b))
    
    # Save CSV log
    import csv
    csv_path = DATA / "phase1_samples.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "dx", "dist", "num_core", "num_hinge", "num_brick"])
        writer.writerows(records)
    console.log({"saved_csv": str(csv_path)})
    # Prefer forward movers; if none, use absolute displacement
    movers = [(dx, dist, ind) for dx, dist, ind in scored if dx > 0.0]
    if movers:
        movers.sort(key=lambda t: t[0], reverse=True)
        top = movers[:PHASE2_TOP_K]
        console.log({"phase1_done": True, "candidates_forward": len(movers), "selected": len(top)})
    else:
        scored.sort(key=lambda t: t[1], reverse=True)
        top = scored[:PHASE2_TOP_K]
        console.log({"phase1_done": True, "candidates_any": len(scored), "selected": len(top), "note": "no forward movers"})
    top_candidates = [ind for _, _, ind in top]
    best_fit = float(top[0][0]) if top else -1e12
    # Save top graph preview if available
    if top_candidates and top_candidates[0].graph is not None:
        json_path = DATA / "best_robot_graph.json"
        png_path = DATA / "best_robot_graph.png"
        save_graph_as_json(top_candidates[0].graph, save_file=json_path)
        draw_graph(top_candidates[0].graph, title=f"Best graph (dx={best_fit:.3f})", save_file=png_path)
        console.log({"saved_graph": str(json_path), "saved_image": str(png_path)})
    return best_fit, top_candidates


def train_controller_cma(best_individual: Individual) -> tuple[float, np.ndarray]:
    console.rule("Phase 2: Controller Training (CMA-ES)")
    if best_individual.graph is None:
        raise ValueError("Best individual has no graph")
    # Build a probe model to determine controller dimensions
    # Probe model for dimensions; robust to errors
    try:
        core = construct_mjspec_from_graph(best_individual.graph)
        world_probe = OlympicArena()
        # try preferred spawn
        try:
            world_probe.spawn(core.spec, position=PHASE1_SPAWN_POS, correct_collision_with_floor=True)
            model_probe = world_probe.spec.compile()
        except Exception:
            # retry slightly above
            pos2 = list(PHASE1_SPAWN_POS)
            pos2[2] = float(pos2[2]) + 0.05
            world_probe = OlympicArena()
            world_probe.spawn(core.spec, position=pos2, correct_collision_with_floor=True)
            model_probe = world_probe.spec.compile()
    except Exception as e:
        console.log({"phase2_error": f"probe_compile_failed: {e}"})
        return -np.inf, np.array([])
    if cma is None:
        console.log({"phase2_error": "cma_not_installed"})
        return -np.inf, np.array([])
    if model_probe.nu == 0:
        console.log({"phase2_error": "no_actuators"})
        return -np.inf, np.array([])
    input_size = len(mj.MjData(model_probe).qpos)
    hidden_size = CONTROLLER_HIDDEN_SIZE
    output_size = model_probe.nu
    # Controller type selection
    if CMA_CONTROLLER_TYPE == "cpg":
        # Per-actuator amplitudes and phases + global freq
        param_dim = output_size * 2 + 1
    else:
        # NN: w1 (in*hid), w2 (hid*hid), w3 (hid*out)
        param_dim = input_size * hidden_size + hidden_size * hidden_size + hidden_size * output_size

    def _decode_weights(vec: np.ndarray) -> dict[str, np.ndarray]:
        v = np.asarray(vec).astype(np.float64)
        if v.size != param_dim:
            v = v[:param_dim] if v.size > param_dim else np.pad(v, (0, param_dim - v.size))
        if CMA_CONTROLLER_TYPE == "cpg":
            return {"theta": v}
        p = 0
        w1 = v[p : p + input_size * hidden_size].reshape((input_size, hidden_size)); p += input_size * hidden_size
        w2 = v[p : p + hidden_size * hidden_size].reshape((hidden_size, hidden_size)); p += hidden_size * hidden_size
        w3 = v[p : p + hidden_size * output_size].reshape((hidden_size, output_size))
        return {"w1": w1, "w2": w2, "w3": w3}

    def objective(vec: np.ndarray) -> float:
        # Build fresh robot and world per evaluation to ensure clean state
        try:
            core_eval = construct_mjspec_from_graph(best_individual.graph)
            world = OlympicArena()
            try:
                world.spawn(core_eval.spec, position=PHASE1_SPAWN_POS, correct_collision_with_floor=True)
                model = world.spec.compile()
            except Exception:
                pos2 = list(PHASE1_SPAWN_POS)
                pos2[2] = float(pos2[2]) + 0.05
                world = OlympicArena()
                world.spawn(core_eval.spec, position=pos2, correct_collision_with_floor=True)
                model = world.spec.compile()
            data = mj.MjData(model)
            mj.mj_resetData(model, data)
        except Exception as e:
            console.log({"phase2_eval_error": str(e)})
            return 1e9

        weights = _decode_weights(vec)

        def cb(m: mj.MjModel, d: mj.MjData) -> None:
            if CMA_CONTROLLER_TYPE == "cpg":
                vec = weights["theta"]
                amps_raw = vec[:output_size]
                phases_raw = vec[output_size: output_size * 2]
                freq_raw = vec[-1]
                amps = np.tanh(amps_raw) * np.pi
                phases = np.tanh(phases_raw) * np.pi
                freq = 1.5 + np.tanh(freq_raw) * 2.0
                t = float(d.time)
                outputs = amps * np.sin(2 * np.pi * freq * t + phases)
            else:
                inputs = d.qpos
                layer1 = np.tanh(inputs @ weights["w1"]) 
                layer2 = np.tanh(layer1 @ weights["w2"]) 
                outputs = np.tanh(layer2 @ weights["w3"]) * np.pi
            d.ctrl[:] = np.clip(outputs, -np.pi / 2, np.pi / 2)

        mj.set_mjcb_control(cb)
        dt = model.opt.timestep
        steps = int(RUN_DURATION / dt)
        # Track core body position history for full fitness
        history: list[tuple[float, float, float]] = []
        try:
            core_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "core")
        except Exception:
            core_bid = -1
        for k in range(steps):
            mj.mj_step(model, data)
            if not np.isfinite(data.qpos).all():
                return 1e9
            if k % 10 == 0:
                if core_bid >= 0:
                    px, py, pz = data.xpos[core_bid, 0], data.xpos[core_bid, 1], data.xpos[core_bid, 2]
                else:
                    # fallback to first 3 qpos if unavailable
                    px = float(data.qpos[0]) if len(data.qpos) > 0 else 0.0
                    py = float(data.qpos[1]) if len(data.qpos) > 1 else 0.0
                    pz = float(data.qpos[2]) if len(data.qpos) > 2 else 0.0
                history.append((float(px), float(py), float(pz)))
        fit = fitness_function(history)
        return -fit

    x0 = np.zeros(param_dim)
    sigma0 = CMA_SIGMA0
    pop = CMA_POPSIZE if CMA_POPSIZE is not None else max(8, min(24, 4 + int(np.log(param_dim + 1) * 3)))
    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, {"popsize": int(pop), "seed": int(SEED), "verbose": -1})
    console.log({"phase2_start": True, "param_dim": int(param_dim), "popsize": int(pop)})
    best_f = -np.inf
    best_theta: np.ndarray | None = None
    evals = 0
    while not es.stop() and evals < CMA_MAX_EVALS:
        sols = es.ask()
        losses = []
        for s in sols:
            loss = objective(s)
            fit = -loss
            if fit > best_f:
                best_f = fit
                best_theta = np.array(s)
            losses.append(loss)
        es.tell(sols, losses)
        evals += len(sols)
        if evals % (pop * 5) == 0:
            console.log({"evals": evals, "best_f": float(best_f)})
    if best_theta is None:
        return -np.inf, np.array([])
    best_weights = _decode_weights(best_theta)
    if CMA_CONTROLLER_TYPE == "cpg":
        np.save(DATA / "best_cpg_theta.npy", best_weights["theta"])  # type: ignore[index]
        console.log({"saved": str(DATA / "best_cpg_theta.npy")})
    else:
        np.savez(DATA / "best_nn_weights.npz", **{k: v for k, v in best_weights.items()})
        console.log({"saved": str(DATA / "best_nn_weights.npz")})
    return best_f, best_theta


def main() -> None:
    best_fit, candidates = run_body_evolution()
    if not candidates:
        console.log({"error": "no_candidates_found"})
        return
    # Preflight Phase 2: filter candidates that cannot compile or have no actuators
    valid_candidates: list[Individual] = []
    for ind in candidates:
        try:
            core = construct_mjspec_from_graph(ind.graph)  # type: ignore[arg-type]
            model = core.spec.compile()
            if model.nu == 0:
                continue
            valid_candidates.append(ind)
        except Exception:
            continue
    if not valid_candidates:
        console.log({"error": "no_valid_phase2_candidates"})
        return
    # Phase 2: train each candidate and pick overall best
    console.rule("Phase 2: Controller Training (CMA-ES)")
    console.log({"training_candidates": len(valid_candidates)})
    results: list[tuple[float, Individual, np.ndarray]] = []
    for idx, cand in enumerate(valid_candidates):
        console.log({"training_candidate": idx + 1, "phase1_fitness": float(cand.fitness or -1e12)})
        ctrl_fit, theta = train_controller_cma(cand)
        results.append((ctrl_fit, cand, theta))
        console.log({"candidate": idx + 1, "phase2_fitness": float(ctrl_fit)})
    
    # Pick overall best
    best_result = max(results, key=lambda r: r[0])
    best_ctrl_fit, best_cand, best_theta = best_result
    console.log({"phase2_done": True, "best_ctrl_fitness": float(best_ctrl_fit), "from_candidate": valid_candidates.index(best_cand) + 1})
    
    # Save overall best controller
    if CMA_CONTROLLER_TYPE == "cpg":
        np.save(DATA / "overall_best_cpg_theta.npy", best_theta)
    else:
        # decode and save as npz
        input_size = len(best_theta) // (CONTROLLER_HIDDEN_SIZE * 2 + CONTROLLER_HIDDEN_SIZE)  # approximate
        # for simplicity just save raw theta
        np.save(DATA / "overall_best_nn_theta.npy", best_theta)
    console.log({"saved_overall_best": True})


if __name__ == "__main__":
    main()


