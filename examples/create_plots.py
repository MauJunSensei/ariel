from glob import glob
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths and constants
RESULTS = Path("results")
REPORTS = Path("reports")
ENVS = ("SimpleFlat", "Rugged")


def seeds_for(env: str, algo: str) -> list[int]:
    paths = sorted(glob(str(RESULTS / env / algo / "seed_*")))
    return [int(Path(p).name.split("_")[-1]) for p in paths]


def load_runs(env: str, algo: str, seeds: list[int]) -> list[pd.DataFrame]:
    runs: list[pd.DataFrame] = []
    for seed in seeds:
        f = RESULTS / env / algo / f"seed_{seed}" / "metrics.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["seed"] = seed
            runs.append(df)
        else:
            print(f"[warn] missing metrics: {f}")
    return runs


def mean_curve(env: str, algo: str) -> tuple[np.ndarray, np.ndarray] | None:
    seeds = seeds_for(env, algo)
    runs = load_runs(env, algo, seeds)
    if not runs:
        print(f"[warn] no runs for {env}/{algo}")
        return None
    merged = None
    for i, df in enumerate(runs):
        cur = df[["generation", "best_fitness"]].rename(columns={"best_fitness": f"r{i}"})
        merged = cur if merged is None else pd.merge(merged, cur, on="generation", how="inner")
    gens = merged["generation"].to_numpy()
    vals = merged.drop(columns=["generation"]).to_numpy()
    return gens, vals.mean(axis=1)


def final_from_run(env: str, algo: str, seed: int) -> float | None:
    f_final = RESULTS / env / algo / f"seed_{seed}" / "final.json"
    f_metrics = RESULTS / env / algo / f"seed_{seed}" / "metrics.csv"
    if f_final.exists():
        with f_final.open() as fh:
            d = json.load(fh)
        v = d.get("final_best_fitness")
        if v is not None:
            return float(v)
    if f_metrics.exists():
        df = pd.read_csv(f_metrics)
        if len(df):
            return float(df["best_fitness"].iloc[-1])
    print(f"[warn] no final for {env}/{algo}/seed_{seed}")
    return None


def _baseline_algo_for(env: str) -> str | None:
    for a in ("baseline", "random"):
        if seeds_for(env, a):
            return a
    return None


def baseline_mean(env: str) -> float | None:
    for algo in ("baseline", "random"):
        seeds = seeds_for(env, algo)
        if not seeds:
            continue
        vals: list[float] = []
        for s in seeds:
            f = RESULTS / env / algo / f"seed_{s}" / "metrics.csv"
            if f.exists():
                df = pd.read_csv(f)
                vals.extend(df["best_fitness"].tolist())
        if vals:
            return float(np.mean(vals))
    return None


def figure_triptych_envs() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    # Panels A & B: per-env CMA-ES mean vs Random/baseline mean (dashed)
    for ax, env in zip(axes[:2], ENVS):
        c = mean_curve(env, "cmaes")
        if c is not None:
            gx, gy = c
            ax.plot(gx, gy, label=f"CMA-ES {env}", linewidth=2)
            mu = baseline_mean(env)
            if mu is not None:
                ax.plot([gx.min(), gx.max()], [mu, mu], "--", label=f"Random {env}", linewidth=1.5)
        ax.set_title(env)
        ax.set_xlabel("Generation")
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("Best fitness (Δy, m)")

    # Panel C: CMA-ES only, Flat vs Rugged
    cf = mean_curve("SimpleFlat", "cmaes")
    cr = mean_curve("Rugged", "cmaes")
    if cf is not None:
        g, y = cf
        axes[2].plot(g, y, label="CMA-ES SimpleFlat", linewidth=2)
    if cr is not None:
        g2, y2 = cr
        axes[2].plot(g2, y2, label="CMA-ES Rugged", linewidth=2)
    axes[2].set_xlabel("Generation")
    axes[2].legend(loc="best", fontsize=8)

    # Consistent x-limits: intersection of available ranges
    xmins, xmaxs = [], []
    for ax in axes:
        xs = [ln.get_xdata() for ln in ax.get_lines()]
        if xs:
            arr = np.concatenate(xs)
            xmins.append(arr.min())
            xmaxs.append(arr.max())
    if xmins and xmaxs:
        xmin, xmax = float(np.max(xmins)), float(np.min(xmaxs))
        if xmin < xmax:
            for ax in axes:
                ax.set_xlim(xmin, xmax)

    fig.tight_layout()
    fig.savefig(REPORTS / "fig_triptych_envs.png", dpi=200)
    fig.savefig(REPORTS / "fig_triptych_envs.pdf")
    plt.close(fig)


def figure_2_and_summary() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for env in ENVS:
        # CMA-ES
        for s in seeds_for(env, "cmaes"):
            v = final_from_run(env, "cmaes", s)
            if v is not None:
                rows.append({"env": env, "algo": "cmaes", "seed": s, "final_best": v})
        # Baseline/Random
        b = _baseline_algo_for(env)
        if b:
            for s in seeds_for(env, b):
                v = final_from_run(env, b, s)
                if v is not None:
                    rows.append({"env": env, "algo": b, "seed": s, "final_best": v})
    df = pd.DataFrame(rows)

    # Summary CSV
    stats = []
    if not df.empty:
        for (env, algo), sub in df.groupby(["env", "algo"], sort=False):
            n = len(sub)
            mean = float(sub["final_best"].mean()) if n else 0.0
            sd = float(sub["final_best"].std(ddof=1)) if n > 1 else 0.0
            ci = 1.96 * sd / np.sqrt(n) if n else 0.0
            stats.append({
                "env": env,
                "algo": algo,
                "n": n,
                "mean": mean,
                "sd": sd,
                "ci_lower": mean - ci,
                "ci_upper": mean + ci,
                "seeds": ",".join(map(str, sub["seed"]))
            })
    pd.DataFrame(stats).to_csv(REPORTS / "summary_stats.csv", index=False)

    # Boxplot + jitter (order fixed, use available data)
    fig, ax = plt.subplots(figsize=(8, 4))
    order_labels = [
        ("SimpleFlat", "cmaes", "CMA-ES\nFlat"),
        ("Rugged", "cmaes", "CMA-ES\nRugged"),
        ("SimpleFlat", _baseline_algo_for("SimpleFlat"), "Random\nFlat"),
        ("Rugged", _baseline_algo_for("Rugged"), "Random\nRugged"),
    ]
    data, labels = [], []
    for env, algo, lab in order_labels:
        if (algo is not None) and (not df.empty):
            arr = df[(df.env == env) & (df.algo == algo)]["final_best"].to_numpy()
        else:
            arr = np.array([])
        data.append(arr)
        labels.append(lab)

    any_data = any(a.size for a in data)
    if any_data:
        ax.boxplot(data, labels=labels, showfliers=False)
        for i, arr in enumerate(data, start=1):
            if arr.size:
                x = np.random.uniform(i - 0.15, i + 0.15, size=arr.size)
                ax.scatter(x, arr, alpha=0.7, s=12, color="#1f77b4")
    else:
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(labels)
    ax.set_ylabel("Final best fitness (Δy, m)")
    fig.tight_layout()
    fig.savefig(REPORTS / "fig2_final_performance.png", dpi=200)
    fig.savefig(REPORTS / "fig2_final_performance.pdf")
    plt.close(fig)


def main() -> None:
    figure_triptych_envs()
    figure_2_and_summary()


if __name__ == "__main__":
    main()


