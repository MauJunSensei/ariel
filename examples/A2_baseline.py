# Third-party libraries
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
import os
from datetime import datetime
import csv
import json
from pathlib import Path

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.runners import simple_runner
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.simulation.environments.rugged_heightmap import RuggedTerrainWorld



# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# Keep track of data / history
HISTORY = []

# -------- Minimal experiment config (edit here) --------
ENV_LABEL = "Rugged"  # "SimpleFlat" or "Rugged"
BASE_SEED = 42
RUNS = 3                   # run the baseline exactly 3 times
DURATION_S = 60.0          # fixed horizon per run
OUT_ROOT = Path("results")
# ------------------------------------------------------

# Baseline task & fitness definition
# Task: directed forward locomotion on SimpleFlatWorld
# Fitness: forward displacement Δx of the robot core over the run duration
def compute_forward_displacement(history:list) -> float:
    if not history:
        return 0.0
    pos_data = np.array(history)
    delta_x = float(pos_data[-1, 0] - pos_data[0, 0])
    return delta_x

def compute_forward_delta_y(history:list) -> float:
    if not history:
        return 0.0
    pos = np.array(history)
    start_y = float(pos[0, 1])
    min_y = float(np.min(pos[:, 1]))
    return start_y - min_y

def save_baseline_fitness(delta_x: float) -> None:
    """Append fitness to CSV under __data__/A2_template/ for later plotting."""
    try:
        base_dir = os.path.dirname(__file__) or os.getcwd()
    except Exception:
        base_dir = os.getcwd()
    out_dir = os.path.join(base_dir, "__data__", "A2_template")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "baseline_random.csv")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header_needed = not os.path.exists(csv_path)
    try:
        with open(csv_path, "a") as f:
            if header_needed:
                f.write("timestamp,delta_x\n")
            f.write(f"{timestamp},{delta_x}\n")
        print(f"Saved baseline fitness to: {csv_path}")
    except Exception:
        pass

def random_move(model, data, to_track) -> None:
    """Generate random movements for the robot's joints.
    
    The mujoco.set_mjcb_control() function will always give 
    model and data as inputs to the function. Even if you don't use them,
    you need to have them as inputs.

    Parameters
    ----------

    model : mujoco.MjModel
        The MuJoCo model of the robot.
    data : mujoco.MjData
        The MuJoCo data of the robot.

    Returns
    -------
    None
        This function modifies the data.ctrl in place.
    """

    # Get the number of joints
    num_joints = model.nu 
    
    # Hinges take values between -pi/2 and pi/2
    hinge_range = np.pi/2
    rand_moves = np.random.uniform(low= -hinge_range, # -pi/2
                                   high=hinge_range, # pi/2
                                   size=num_joints) 

    # There are 2 ways to make movements:
    # 1. Set the control values directly (this might result in junky physics)
    # data.ctrl = rand_moves

    # 2. Add to the control values with a delta (this results in smoother physics)
    delta = 0.05
    data.ctrl += rand_moves * delta 

    # Bound the control values to be within the hinge limits.
    # If a value goes outside the bounds it might result in jittery movement.
    data.ctrl = np.clip(data.ctrl, -np.pi/2, np.pi/2)

    # Save movement to history
    HISTORY.append(to_track[0].xpos.copy())

    ##############################################
    #
    # Take all the above into consideration when creating your controller
    # The input size, output size, output range
    # Your network might return ranges [-1,1], so you will need to scale it
    # to the expected [-pi/2, pi/2] range.
    # 
    # Or you might not need a delta and use the direct controller outputs
    #
    ##############################################

def show_qpos_history(history:list):
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], 'b-', label='Path')
    plt.plot(pos_data[0, 0], pos_data[0, 1], 'go', label='Start')
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], 'ro', label='End')
    
    # Add labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position') 
    plt.title('Robot Path in XY Plane')
    plt.legend()
    plt.grid(True)
    
    # Set equal aspect ratio and center at (0,0)
    plt.axis('equal')
    max_range = max(abs(pos_data).max(), 0.3)  # At least 1.0 to avoid empty plots
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)
    
    # Save figure alongside CSV for convenience
    try:
        base_dir = os.path.dirname(__file__) or os.getcwd()
    except Exception:
        base_dir = os.getcwd()
    out_dir = os.path.join(base_dir, "__data__", "A2_template")
    os.makedirs(out_dir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(out_dir, f"trajectory_{now}.png")
    try:
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"Saved trajectory plot to: {fig_path}")
    except Exception:
        pass
    plt.show()

def main():
    """Baseline random controller: run exactly RUNS times and save results like A2."""
    mujoco.set_mjcb_control(None)  # DO NOT REMOVE

    env_key = "flat" if ENV_LABEL == "SimpleFlat" else "rugged"
    algo_label = "baseline"
    run_root = OUT_ROOT / ENV_LABEL / algo_label
    run_root.mkdir(parents=True, exist_ok=True)

    for run_idx in range(RUNS):
        seed = BASE_SEED + run_idx
        np.random.seed(seed)

        # Build world and robot (retain original lines)
        world = SimpleFlatWorld() if env_key == "flat" else RuggedTerrainWorld()
        gecko_core = gecko()     # DO NOT CHANGE
        world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
        model = world.spec.compile()
        data = mujoco.MjData(model) # type: ignore

        geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
        to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

        # Run a single fixed-duration episode
        HISTORY.clear()
        mujoco.set_mjcb_control(lambda m, d: random_move(m, d, to_track))
        simple_runner(model=model, data=data, duration=DURATION_S)
        mujoco.set_mjcb_control(None)

        # Compute fitness (Δy), save like A2
        dy = compute_forward_delta_y(HISTORY)

        seed_dir = run_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = seed_dir / "metrics.csv"
        final_path = seed_dir / "final.json"
        params_path = seed_dir / "params.json"

        with params_path.open("w") as f:
            json.dump({
                "env": ENV_LABEL,
                "algo": algo_label,
                "seed": int(seed),
                "generations": 1,
                "duration_s": float(DURATION_S),
                "controller": "random delta",
                "fitness": "delta_y = start_y - min_y",
            }, f, indent=2)

        with metrics_path.open("w", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=["generation", "evaluations", "best_fitness", "mean_fitness", "timestamp"]) 
            writer.writeheader()
            writer.writerow({
                "generation": 1,
                "evaluations": 1,
                "best_fitness": float(dy),
                "mean_fitness": "",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })

        with final_path.open("w") as f:
            json.dump({
                "final_best_fitness": float(dy),
                "env": ENV_LABEL,
                "algo": algo_label,
                "seed": int(seed),
                "generations": 1,
            }, f, indent=2)

    print(f"Baseline runs saved under: {run_root}")
    # If you want to record a video of your simulation, you can use the video renderer.

    # # Non-default VideoRecorder options
    # PATH_TO_VIDEO_FOLDER = "./__videos__"
    # video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)

    # # Render with video recorder
    # video_renderer(
    #     model,
    #     data,
    #     duration=30,
    #     video_recorder=video_recorder,
    # )

if __name__ == "__main__":
    main()


