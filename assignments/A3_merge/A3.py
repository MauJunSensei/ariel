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
from ariel.simulation.environments import OlympicArena
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
    PHASE2_TOP_K,
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
CWD = Path.cwd()
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
    world = OlympicArena(load_precompiled=False)
    world.spawn(robot.spec, position=SPAWN_POS, correct_collision_with_floor=True)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
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
    def baseline_cb(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
        t = float(data.time)
        amps = np.full(model.nu, GA_BASELINE_AMPLITUDE)
        # Add random phase offsets for each individual to create diversity
        if GA_USE_RANDOM_CONTROLLER:
            phase_seed = hash(tuple(individual.body_genes[0].flat[:5])) % 100000
            phase_rng = np.random.default_rng(phase_seed)
            phases = phase_rng.uniform(0.0, 2*np.pi, model.nu)
            # Also vary frequency slightly
            freq_variation = phase_rng.uniform(0.8, 1.2)
            freq = GA_BASELINE_FREQUENCY * freq_variation
        else:
            phases = np.linspace(0.0, np.pi, model.nu, dtype=np.float64)
            freq = GA_BASELINE_FREQUENCY
        targets = amps * np.sin(2 * np.pi * freq * t + phases)
        return np.clip(targets, -np.pi / 2, np.pi / 2)

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


def run_body_evolution() -> tuple[float, list[Individual]]:
    console.rule("Phase 1: Body Evolution (GA)")
    console.log({"population": POPULATION_SIZE, "generations": GENERATIONS})
    population = [Individual(body_genes=generate_body_genes()) for _ in range(POPULATION_SIZE)]
    for ind in population:
        generate_body(ind)
    
    # Track all unique individuals across all generations
    candidates: dict[int, Individual] = {}  # keyed by id to avoid duplicates
    
    for gen in range(GENERATIONS):
        console.log(f"Generation {gen + 1}/{GENERATIONS}")
        evaluate_all(population)
        fitness_vals = [float(ind.fitness) for ind in population if ind.fitness is not None]
        if fitness_vals:
            gen_best_fit = max(fitness_vals)
            console.log({"best_fitness": gen_best_fit})
            # Collect ALL individuals with valid fitness and graph
            for ind in population:
                if ind.fitness is not None and ind.graph is not None:
                    ind_id = id(ind)
                    if ind_id not in candidates:
                        cand = Individual(body_genes=[arr.copy() for arr in ind.body_genes])
                        cand.graph = ind.graph
                        cand.fitness = ind.fitness
                        candidates[ind_id] = cand
        survivors = select_survivors(population, num_survivors=POPULATION_SIZE // 2)
        offspring: list[Individual] = []
        for ind in survivors:
            child = Individual(body_genes=[arr.copy() for arr in ind.body_genes])
            generate_body(child, child.body_genes)
            mutate(child)
            offspring.append(child)
        population = survivors + offspring
    
    # Sort all candidates by fitness and take top K
    cand_list = sorted(candidates.values(), key=lambda c: float(c.fitness or -1e12), reverse=True)
    top_candidates = cand_list[:PHASE2_TOP_K]
    console.log({"phase1_done": True, "total_candidates": len(cand_list), "top_k_selected": len(top_candidates)})
    
    if top_candidates:
        best_fit = float(top_candidates[0].fitness or -1e12)
        # Save top candidate graph
        if top_candidates[0].graph is not None:
            json_path = DATA / "best_robot_graph.json"
            png_path = DATA / "best_robot_graph.png"
            save_graph_as_json(top_candidates[0].graph, save_file=json_path)
            draw_graph(top_candidates[0].graph, title=f"Best graph (fit={best_fit:.3f})", save_file=png_path)
            console.log({"saved_graph": str(json_path), "saved_image": str(png_path)})
    else:
        best_fit = -1e12
    
    return best_fit, top_candidates


def train_controller_cma(best_individual: Individual) -> tuple[float, np.ndarray]:
    console.rule("Phase 2: Controller Training (CMA-ES)")
    if best_individual.graph is None:
        raise ValueError("Best individual has no graph")
    # Build a probe model to determine controller dimensions
    core = construct_mjspec_from_graph(best_individual.graph)
    world_probe = OlympicArena()  # use precompiled arena assets
    world_probe.spawn(core.spec, position=SPAWN_POS, correct_collision_with_floor=False)
    model_probe = world_probe.spec.compile()
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
            core_eval = construct_mjspec_from_graph(best_individual.graph)  # rebuild robot spec
            world = OlympicArena()
            world.spawn(core_eval.spec, position=SPAWN_POS, correct_collision_with_floor=False)
            model = world.spec.compile()
            data = mj.MjData(model)
            mj.mj_resetData(model, data)
        except Exception:
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
    
    # Phase 2: train each candidate and pick overall best
    console.rule("Phase 2: Controller Training (CMA-ES)")
    console.log({"training_candidates": len(candidates)})
    results: list[tuple[float, Individual, np.ndarray]] = []
    for idx, cand in enumerate(candidates):
        console.log({"training_candidate": idx + 1, "phase1_fitness": float(cand.fitness or -1e12)})
        ctrl_fit, theta = train_controller_cma(cand)
        results.append((ctrl_fit, cand, theta))
        console.log({"candidate": idx + 1, "phase2_fitness": float(ctrl_fit)})
    
    # Pick overall best
    best_result = max(results, key=lambda r: r[0])
    best_ctrl_fit, best_cand, best_theta = best_result
    console.log({"phase2_done": True, "best_ctrl_fitness": float(best_ctrl_fit), "from_candidate": candidates.index(best_cand) + 1})
    
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


