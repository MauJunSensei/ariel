"""Assignment 3: Evolve robot bodies and controllers for OlympicArena."""

from pathlib import Path
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer
import cma

from ariel import console
from ariel.simulation.environments import OlympicArena
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule

# --- SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)
SPAWN_POS = [-0.8, 0, 0.1]

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)


def build_robot_from_genotype(genotype: list[npt.NDArray[np.float32]]) -> CoreModule:
    """Build a robot directly using modules (gecko-style), parameterized by genotype.
    
    Genotype interpretation:
    - genotype[0][0:4]: limb lengths (1-3 segments per limb)
    - genotype[0][4:8]: hinge rotation offsets (0, 90, 180, 270 degrees)
    - genotype[1][0:4]: whether limb has hinge at base (>0.5 = yes)
    """
    g0, g1, _ = genotype
    
    core = CoreModule(index=0)
    idx = 1
    
    # 4 limbs: FRONT, LEFT, RIGHT, BACK
    faces = [ModuleFaces.FRONT, ModuleFaces.LEFT, ModuleFaces.RIGHT, ModuleFaces.BACK]
    base_rotations = [0, 90, -90, 180]
    
    for i, (face, base_rot) in enumerate(zip(faces, base_rotations)):
        limb_len = int(1 + (g0[i] * 2))  # 1-3 segments
        limb_len = max(1, min(3, limb_len))
        
        rot_offset = int(g0[4 + i] * 4) * 90  # 0, 90, 180, 270
        has_hinge = g1[i] > 0.5
        
        prev_module = core
        prev_face = face
        
        for seg in range(limb_len):
            if seg == 0 and has_hinge:
                hinge = HingeModule(index=idx)
                hinge.rotate(base_rot + rot_offset)
                idx += 1
                prev_module.sites[prev_face].attach_body(hinge.body, prefix=f"limb{i}_seg{seg}_")
                prev_module = hinge
                prev_face = ModuleFaces.FRONT
            else:
                brick = BrickModule(index=idx)
                brick.rotate(base_rot if seg == 0 else 0)
                idx += 1
                prev_module.sites[prev_face].attach_body(brick.body, prefix=f"limb{i}_seg{seg}_")
                prev_module = brick
                prev_face = ModuleFaces.FRONT
                
    return core


def quick_viability_score(core: CoreModule, run_seconds: float = 1.2) -> float:
    """Score a body by running a short simulation with gentle oscillatory control."""
    try:
        world = OlympicArena()
        world.spawn(core.spec, spawn_position=SPAWN_POS.copy(), small_gap=0.02, correct_for_bounding_box=False)
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
    
    return abs(x1 - x0)


def sample_bodies(total_trials: int = 500, top_k: int = 50, seeds: list[int] | None = None) -> list[tuple[float, CoreModule]]:
    """Generate and screen robot bodies, return top performers."""
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    candidates: list[tuple[float, CoreModule]] = []
    per_seed = max(1, total_trials // len(seeds))
    
    attempts = 0
    seed_idx = 0
    max_attempts = total_trials * 10
    
    while len(candidates) < top_k and attempts < max_attempts:
        s = seeds[seed_idx % len(seeds)]
        seed_idx += 1
        local_rng = np.random.default_rng(s + attempts)
        attempts += 1
        
        genotype = [
            local_rng.random(64).astype(np.float32),
            local_rng.random(64).astype(np.float32),
            local_rng.random(64).astype(np.float32),
        ]
        
        try:
            core = build_robot_from_genotype(genotype)
            # Test compile
            _test_model = core.spec.compile()
            if _test_model.nu == 0:
                continue
            # Test in arena
            world = OlympicArena()
            world.spawn(core.spec, spawn_position=SPAWN_POS.copy(), small_gap=0.02, correct_for_bounding_box=False)
            model = world.spec.compile()
            if model.nu == 0:
                continue
        except Exception as e:
            if attempts <= 3:
                console.log({"sample_error": str(e)[:100]})
            continue
            
        try:
            score = quick_viability_score(core)
            if np.isfinite(score):
                candidates.append((score, core))
        except Exception as e:
            if attempts <= 3:
                console.log({"viability_error": str(e)[:100]})
            continue

    candidates.sort(key=lambda x: x[0], reverse=True)
    console.log({"screening_complete": len(candidates), "attempts": attempts})
    return candidates[:top_k]


def train_body_with_cma(core: CoreModule, duration: float = 10.0, budget: int = 200) -> tuple[float, np.ndarray]:
    """Train a controller for a given body using CMA-ES."""
    try:
        world = OlympicArena()
        world.spawn(core.spec, spawn_position=SPAWN_POS.copy(), small_gap=0.02, correct_for_bounding_box=False)
        model = world.spec.compile()
    except Exception as e:
        console.log({"train_error": str(e)})
        return -np.inf, np.array([])
        
    if model.nu == 0:
        return -np.inf, np.array([])

    out_size = model.nu
    param_dim = out_size * 2 + 1  # amps, phases, freq

    def sinusoid_objective(vec: np.ndarray) -> float:
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
        amps = np.tanh(amps_raw) * (np.pi / 2)
        phases = np.tanh(phases_raw) * np.pi
        freq = 1.0 + np.tanh(freq_raw) * 1.0

        def cb(m: mj.MjModel, d: mj.MjData) -> None:
            t = float(d.time)
            targets = amps * np.sin(2 * np.pi * freq * t + phases)
            d.ctrl += targets * 0.01
            d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

        mj.set_mjcb_control(cb)
        dt = model.opt.timestep
        steps = int(duration / dt)
        
        try:
            core_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "core")
            x0 = float(data.xpos[core_bid, 0])
        except Exception:
            x0 = float(data.qpos[0]) if len(data.qpos) > 0 else 0.0
        
        for _ in range(steps):
            mj.mj_step(model, data)
            if not np.isfinite(data.qpos).all():
                return 1e9
        
        try:
            x1 = float(data.xpos[core_bid, 0])
        except Exception:
            x1 = float(data.qpos[0]) if len(data.qpos) > 0 else 0.0
        
        return -(x1 - x0)  # minimize negative displacement

    x0 = np.zeros(param_dim)
    sigma0 = 0.5
    pop = max(4, min(20, 4 + int(np.log(param_dim + 1) * 3)))
    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, {"popsize": pop, "seed": int(SEED)})
    
    best_f = -np.inf
    best_theta = None
    evals = 0

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
    
    if best_theta is None:
        return -np.inf, np.array([])
    
    return best_f, best_theta


def main() -> None:
    """Entry point."""
    # Stage 1: Sample and screen bodies
    console.rule("Stage 1: Body Sampling")
    candidates = sample_bodies(total_trials=500, top_k=50)
    
    if not candidates:
        console.log({"error": "no viable bodies"})
        return
    
    # Stage 2: Train top bodies with CMA-ES
    console.rule("Stage 2: Controller Training")
    best_overall = (-np.inf, None, None)  # (fitness, theta, core)
    
    for idx, (viability_score, core) in enumerate(candidates[:10]):  # Train top 10
        console.log({"training_body": idx, "viability": float(viability_score)})
        f, theta = train_body_with_cma(core, duration=10.0, budget=200)
        console.log({"body": idx, "fitness": float(f)})
        
        if f > best_overall[0]:
            best_overall = (f, theta, core)
    
    best_f, best_theta, best_core = best_overall
    console.log({"best_fitness": float(best_f)})
    
    # Save results
    if best_theta is not None:
        np.save(DATA / "best_theta.npy", best_theta)
        console.log({"saved": str(DATA / "best_theta.npy")})
    
    # Visualize best
    if best_core is not None and best_theta is not None and best_f > -np.inf:
        try:
            world = OlympicArena()
            world.spawn(best_core.spec, spawn_position=SPAWN_POS.copy())
            model = world.spec.compile()
            data = mj.MjData(model)
            
            amps_raw = best_theta[:model.nu]
            phases_raw = best_theta[model.nu: model.nu * 2]
            freq_raw = best_theta[-1]
            amps = np.tanh(amps_raw) * (np.pi / 2)
            phases = np.tanh(phases_raw) * np.pi
            freq = 1.0 + np.tanh(freq_raw) * 1.0
            
            def best_controller(m: mj.MjModel, d: mj.MjData) -> None:
                t = float(d.time)
                targets = amps * np.sin(2 * np.pi * freq * t + phases)
                d.ctrl += targets * 0.01
                d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)
            
            mj.set_mjcb_control(best_controller)
            viewer.launch(model=model, data=data)
        except Exception as e:
            console.log({"viewer_error": str(e)})


if __name__ == "__main__":
    main()
