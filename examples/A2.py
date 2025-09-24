# Third-party libraries
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder

from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.simulation.environments.rugged_heightmap import RuggedTerrainWorld
from ariel.simulation.environments.amphiteater_heightmap import AmphitheatreTerrainWorld
from ariel.simulation.environments.boxy_heightmap import BoxyRugged
from ariel.simulation.environments.crater_heightmap import CraterTerrainWorld
from ariel.simulation.environments.djoser_pyramid import PyramidWorld
from ariel.simulation.environments.simple_tilted_world import TiltedFlatWorld

import cma
import concurrent.futures
import os
from datetime import datetime
import csv
import json
from pathlib import Path

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# Keep track of data / history
HISTORY = []


def build_world_and_model(env_name: str, mujoco_module=None):
    """Create a world, spawn a gecko, compile model, create data and bind tracking geoms.

    Returns: (world, model, data, to_track, nu)
    """
    if mujoco_module is None:
        mujoco_module = mujoco
    # choose environment class by name
    # supported names: flat, rugged, amphiteater/amphitheatre, boxy, crater, pyramid, tilted
    key = (env_name or "").lower()
    if key == "rugged":
        world = RuggedTerrainWorld()
    elif key in ("amphiteater", "amphitheatre"):
        world = AmphitheatreTerrainWorld()
    elif key == "boxy":                     # Falls off the edge, resulting in weird final graph
        world = BoxyRugged()
    elif key == "crater":
        world = CraterTerrainWorld()

    # # Robot is spawned inside the pyramid, so disabled for now
    # elif key == "pyramid":                  
    #     print("Note: Pyramid environment doesn't seem to work well...")
    #     world = PyramidWorld()

    elif key == "tilted":
        world = TiltedFlatWorld()
    else:
        # default to simple flat world
        world = SimpleFlatWorld()
    
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco_module.MjData(model)
    geoms = world.spec.worldbody.find_all(mujoco_module.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    nu = model.nu
    return world, model, data, to_track, nu


def decode_params(x, nu):
    """Normalize/reshape parameter vector and return (amps, phases, freq).

    x may be any sequence; if its length doesn't match expected it will be
    padded or trimmed to (2*nu + 1).
    """
    x = np.asarray(x)
    expected = int(nu * 2 + 1)
    if x.size != expected:
        if x.size > expected:
            x = x[:expected]
        else:
            pad = np.zeros(expected - x.size)
            x = np.concatenate([x, pad])
    amps_raw = x[:nu]
    phases_raw = x[nu: nu * 2]
    freq_raw = x[-1]

    amps = np.tanh(amps_raw) * (np.pi / 2)
    phases = np.tanh(phases_raw) * np.pi
    freq = 1.0 + np.tanh(freq_raw) * 1.0
    return amps, phases, freq

def show_qpos_history(history:list, env_name: str = None):
    # Guard: empty history
    if not history:
        print("No trajectory history to plot or save.")
        return

    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Create figure and axis
    plt.figure(figsize=(10, 6))

    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], 'b-', label='Path')

    # Draw start and end with larger markers so they are visible on top
    start = pos_data[0]
    end = pos_data[-1]
    plt.scatter([start[0]], [start[1]], c='green', s=80, edgecolors='k', zorder=5, label='Start')
    plt.scatter([end[0]], [end[1]], c='red', s=80, edgecolors='k', zorder=6, label='End')

    # Add labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    if env_name:
        plt.title(f'Robot Path in XY Plane ({env_name} environment)')
    else:
        plt.title('Robot Path in XY Plane')
    plt.legend()
    plt.grid(True)

    # Set equal aspect ratio and compute axis limits from data extents
    plt.axis('equal')
    xmin, ymin = pos_data[:, 0].min(), pos_data[:, 1].min()
    xmax, ymax = pos_data[:, 0].max(), pos_data[:, 1].max()
    # Add a margin so markers aren't on the edge
    xpad = max(0.1, 0.05 * (xmax - xmin) if xmax > xmin else 0.3)
    ypad = max(0.1, 0.05 * (ymax - ymin) if ymax > ymin else 0.3)
    plt.xlim(xmin - xpad, xmax + xpad)
    plt.ylim(ymin - ypad, ymax + ypad)

    # Ensure __data__ directory exists next to this script
    try:
        base_dir = os.path.dirname(__file__) or os.getcwd()
    except Exception:
        base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "__data__")
    os.makedirs(data_dir, exist_ok=True)

    # Build filename: <env>_YYYYmmdd_HHMMSS.png
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_tag = (env_name or "env").replace(" ", "_")
    filename = f"{env_tag}_{now}.png"
    filepath = os.path.join(data_dir, filename)

    try:
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        print(f"Saved trajectory plot to: {filepath}")
    except Exception:
        pass

    plt.show()


def evaluate_params_worker(x, duration: float, steps_per_loop: int, seed: int, eval_idx: int, env_name: str):
    """Worker function to evaluate a parameter vector in a separate process.

    Returns (score: float, history: list of positions)
    """
    import numpy as _np
    import mujoco as _mujoco
    # initialize RNG for reproducibility per worker
    try:
        _np.random.seed(int(seed) + int(eval_idx))
    except Exception:
        pass

    # Local history for this worker
    local_history = []

    # Build world/model and bind tracking geoms
    world, model, data, to_track, nu = build_world_and_model(env_name, mujoco_module=_mujoco)

    # Decode and normalise params
    amps, phases, freq = decode_params(x, nu)

    def controller(m: _mujoco.MjModel, d: _mujoco.MjData) -> None:
        """
        Create sine-wave control signals from the given amplitudes, phases, and frequency.
        Send those signals to the robot's motors based on the current simulation time.
        If there is a tracked object, record its position.
        """
        t = float(d.time)
        targets = amps * _np.sin(2 * _np.pi * freq * t + phases)
        delta = 0.05
        d.ctrl += targets * delta
        d.ctrl = _np.clip(d.ctrl, -_np.pi / 2, _np.pi / 2)
        if len(to_track) > 0:
            local_history.append(to_track[0].xpos.copy())

    # Run
    _mujoco.set_mjcb_control(controller)
    _mujoco.mj_resetData(model, data)
    while data.time < duration:
        _mujoco.mj_step(model, data, nstep=steps_per_loop)
    _mujoco.set_mjcb_control(None)

    if len(local_history) == 0:
        return 0.0, []
    # Score = start_y - min_y (furthest forward along -Y)
    start_y = local_history[0][1]
    ys = [_p[1] for _p in local_history]
    min_y = float(min(ys))
    score = float(start_y - min_y)
    return score, local_history

def main():
    mujoco.set_mjcb_control(None)  # DO NOT REMOVE

    # ---- Fixed experiment config (edit here, no CLI) ----
    env_label = "Rugged"  # choices: "SimpleFlat" or "Rugged"
    base_seed = 42
    runs = 3
    generations = 40
    duration = 60.0  # seconds per evaluation
    steps_per_loop = 1
    pop_size = 20
    out_root = Path("results")
    # ----------------------------------------------------

    env_key = "flat" if env_label == "SimpleFlat" else "rugged"

    # Infer action dimension
    _, _, _, _, nu = build_world_and_model(env_key)
    param_dim = nu * 2 + 1

    # Static metadata
    meta = {
        "env": env_label,
        "algo": "cmaes",
        "duration_s": duration,
        "population_size": pop_size,
        "controller": "per-joint sinusoid (amp, phase, global freq)",
        "fitness": "delta_y = start_y - min_y",
        "param_dim": int(param_dim),
    }
    try:
        import subprocess
        meta["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        meta["git_commit"] = None

    for run_idx in range(runs):
        seed = int(base_seed) + run_idx
        np.random.seed(seed)

        # CMA-ES init
        x0 = np.zeros(param_dim)
        sigma0 = 0.5
        es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, {'popsize': pop_size, 'seed': int(seed)})

        # Output dirs per seed
        run_dir = out_root / env_label / "cmaes" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.csv"
        final_path = run_dir / "final.json"
        params_path = run_dir / "params.json"

        with params_path.open("w") as f:
            json.dump({**meta, "seed": seed, "generations": generations}, f, indent=2)

        with metrics_path.open("w", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=["generation", "evaluations", "best_fitness", "mean_fitness", "timestamp"]) 
            writer.writeheader()

            eval_count = 0
            best_overall = -np.inf
            best_solution = None

            max_workers = os.cpu_count() or 1
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as exe:
                while not es.stop() and es.countiter < generations:
                    solutions = es.ask()
                    futures = [exe.submit(evaluate_params_worker, s, duration, steps_per_loop, seed, idx, env_key) for idx, s in enumerate(solutions)]
                    losses = []
                    fitnesses = []
                    for fut, s in zip(futures, solutions):
                        score, _ = fut.result()
                        eval_count += 1
                        fitnesses.append(score)
                        losses.append(-score)
                        if score > best_overall:
                            best_overall = score
                            best_solution = np.array(s)
                    es.tell(solutions, losses)
                    mean_fit = float(np.mean(fitnesses)) if fitnesses else None
                    writer.writerow({
                        "generation": es.countiter,
                        "evaluations": eval_count,
                        "best_fitness": float(best_overall),
                        "mean_fitness": "" if mean_fit is None else float(mean_fit),
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    })

        with final_path.open("w") as f:
            json.dump({
                "final_best_fitness": float(best_overall),
                "env": env_label,
                "algo": "cmaes",
                "seed": int(seed),
                "generations": int(generations),
            }, f, indent=2)

        # Replay best for the last run only
        if (run_idx == runs - 1) and (best_solution is not None):
            world, model, data, to_track, _ = build_world_and_model(env_key)
            amps, phases, freq = decode_params(best_solution, nu)

            def viz_controller(m: mujoco.MjModel, d: mujoco.MjData) -> None:
                t = float(d.time)
                targets = amps * np.sin(2 * np.pi * freq * t + phases)
                delta = 0.05
                d.ctrl += targets * delta
                d.ctrl = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)
                if len(to_track) > 0:
                    HISTORY.append(to_track[0].xpos.copy())

            HISTORY.clear()
            mujoco.set_mjcb_control(viz_controller)
            viewer.launch(model=model, data=data)
            show_qpos_history(HISTORY, env_name=env_label)
            mujoco.set_mjcb_control(None)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for numpy and CMA-ES (default: 42)")
    parser.add_argument("--runs", type=int, default=3, help="Number of independent runs (seeds) to execute (default: 3)")
    parser.add_argument(
        "--env",
        type=str,
        default="flat",
        choices=["flat", "rugged", "amphiteater", "amphitheatre", "boxy", "crater", "pyramid", "tilted"],
        help="Environment to use (default: flat)",
    )
    parser.add_argument("--video", action="store_true", help="Record video of best solution instead of launching interactive viewer")
    args = parser.parse_args()
    # Attach as attribute so main can read it (avoids changing signature)
    main._cli_seed = int(args.seed)
    main._cli_runs = int(args.runs)
    main._cli_env = str(args.env)
    main._cli_video = bool(args.video)
    main()


