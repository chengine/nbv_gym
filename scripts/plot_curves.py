#!/usr/bin/env python3
"""
Plot PSNR/LPIPS/SSIM curves per scene from curves_long.csv across multiple entities.
- METHODS_ALLOWLIST controls which methods appear.
- If multiple runs exist for the same (scene, method), we plot mean with a shaded spread (±1 std).
- Colors per method and dashed linestyles are configurable.

Outputs:
  <OUT_DIR>/plots/<scene>_curves.png
  <OUT_DIR>/plots/<scene>_curves.pdf
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== CONFIG (edit me) ======================

ENTITIES: List[str] = [
    "navlab",
    "chengine-stanford-university",
]

# Curves CSV per entity (produced by your wandb export)
CSV_PATH_FMT = "../results/{entity}/wandb_exports/curves_long.csv"

OUT_DIR = "../results/collated"

# Metrics to plot (must match the strings in the curves CSV)
METRICS = ["/psnr", "/ssim", "/lpips"]  # order = subplot order  (PSNR, SSIM, LPIPS)

# Methods to include (None => include all)
METHODS_ALLOWLIST: Optional[List[str]] = ["coverage", "fisher_info", "bayes-rays", "random", "all"]

# Scene renames (to match your paper nomenclature)
SCENE_RENAMES: Dict[str, str] = {
    "chair_3pm": "chair",
    "space_laces_4pm": "space laces",
    "shiny_statue_6pm": "shiny statue",
}

# (Optional) only plot these scenes after renaming (None => all)
SCENE_WHITELIST: Optional[List[str]] = None

# Method display renames (legend labels)
METHOD_RENAMES: Dict[str, str] = {
    "coverage": "Coverage (Ours)",
    "fisher_info": "FisherRF",
    "bayes-rays": "Bayes' Rays",
    "random": "Random",
    "all": "All-Images Oracle",
}

# Hex colors for each method (display name after rename)
METHOD_COLORS: Dict[str, str] = {
    "Coverage (Ours)": "#2CA02C",   # green
    "FisherRF": "#1F77B4",          # blue
    "Bayes' Rays": "#D62728",       # red
    "Random": "#7F7F7F",            # gray
    "All-Images Oracle": "#000000", # black
}

# Which methods (after rename) should be dashed
DASHED_METHODS: List[str] = ["Random"]

# Plot aesthetics
FIGSIZE = (28, 9.2)         # width, height (inches)
LINEWIDTH = 6.
SPREAD_ALPHA = 0.25         # shading transparency for ±std
GRID_ALPHA = 0.10
FONTS = {
    "title": 0,             # set 0 to skip scene title (you can add later in post)
    "axis": 30,
    "tick": 30,
    "legend": 20,
}
LEGEND_OUTSIDE = True       # place legend to the right
LEGEND_COLS = 1

# Optional filtering of incomplete runs
REQUIRE_FINISHED = True     # requires 'state' column in CSV
MIN_FINAL_STEP   = 20000      # e.g., 20000 to require runs reached at least this step
MAX_STEP         = None      # e.g., 30000 to truncate curves to at most this step

# Spread type: 'std' or 'iqr'
SPREAD_TYPE = "std"

# ============================================================

REQ_COLS = {"scene", "uncertainty", "metric", "step", "value"}
RUN_ID_CANDIDATES = ["run", "run_id", "run_name", "run_hash", "runId", "run_name_full", "run_uid"]


def _read_entity_curves(entity: str) -> pd.DataFrame:
    path = CSV_PATH_FMT.format(entity=entity)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing curves CSV for entity '{entity}': {path}")
    df = pd.read_csv(path)
    missing = REQ_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df.copy()
    df["entity"] = entity

    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["step", "value"])

    # Normalize strings
    df["scene"] = df["scene"].astype(str)
    df["uncertainty"] = df["uncertainty"].astype(str)
    df["metric"] = df["metric"].astype(str)

    # Find or synthesize run_id
    run_col = None
    for c in RUN_ID_CANDIDATES:
        if c in df.columns:
            run_col = c
            break
    if run_col is None:
        df["_synth_key"] = df[["entity", "scene", "uncertainty"]].astype(str).agg("|".join, axis=1)
        df["run_uid"] = df["_synth_key"].factorize()[0].astype(str)
        run_col = "run_uid"
    df["run_id"] = df[run_col].astype(str)

    return df[["entity", "scene", "uncertainty", "metric", "step", "value", "run_id"] + ([ "state"] if "state" in df.columns else [])]


def _apply_renames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Scene -> SceneRenamed
    df["Scene"] = df["scene"].map(SCENE_RENAMES).fillna(df["scene"])
    # Method -> MethodDisplay
    df["Method"] = df["uncertainty"].map(METHOD_RENAMES).fillna(df["uncertainty"])
    return df


def _filter_methods(df: pd.DataFrame) -> pd.DataFrame:
    if not METHODS_ALLOWLIST:
        return df
    allow = set(METHODS_ALLOWLIST)
    return df[df["uncertainty"].isin(allow)].copy()


def _maybe_filter_incomplete(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if REQUIRE_FINISHED and "state" in d.columns:
        d = d[d["state"].astype(str).str.lower().eq("finished")]
    if MIN_FINAL_STEP is not None:
        d = (
            d.groupby(["entity", "run_id", "Scene", "Method", "metric"], dropna=False)
             .filter(lambda g: pd.to_numeric(g["step"], errors="coerce").max() >= MIN_FINAL_STEP)
             .reset_index(drop=True)
        )
    if MAX_STEP is not None:
        d = d[d["step"] <= MAX_STEP].copy()
    return d


def _agg_runs_to_band(run_list: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Given a list of per-run frames (with columns step, value),
    unify to a common grid, interpolate each run, return:
       grid, mean, low_bound, high_bound
    where bounds are ±std if SPREAD_TYPE='std', else 25/75 percentiles if 'iqr'.
    """
    # union of all steps
    grid = np.unique(np.concatenate([r["step"].values for r in run_list])).astype(float)
    grid.sort()

    # interpolate each run onto grid
    curves = []
    for r in run_list:
        xs = r["step"].values.astype(float)
        ys = r["value"].values.astype(float)
        if xs.size == 0:
            continue
        if np.unique(xs).size == 1:
            yi = np.full_like(grid, ys[-1])
        else:
            yi = np.interp(grid, xs, ys)
        curves.append(yi)

    if not curves:
        return grid, np.array([]), None, None

    A = np.vstack(curves)  # shape: [n_runs, len(grid)]
    mean = A.mean(axis=0)

    if SPREAD_TYPE == "iqr":
        low = np.percentile(A, 25, axis=0)
        high = np.percentile(A, 75, axis=0)
    else:  # std
        std = A.std(axis=0)
        low = mean - std
        high = mean + std

    return grid, mean, low, high


def _pretty_metric_name(metric: str) -> str:
    if metric == "/psnr": return "PSNR ↑"
    if metric == "/ssim": return "SSIM ↑"
    if metric == "/lpips": return "LPIPS ↓"
    return metric


def _plot_scene(scene: str, sdf: pd.DataFrame, outdir: str):
    # One figure with 3 subplots (PSNR/SSIM/LPIPS)
    fig, axes = plt.subplots(1, len(METRICS), figsize=FIGSIZE, constrained_layout=True)

    for ax, metric in zip(axes, METRICS):
        mdf = sdf[sdf["metric"] == metric]
        if mdf.empty:
            ax.set_visible(False)
            continue

        # group by method, then by run
        for method, msub in mdf.groupby("Method", dropna=False):
            # collect runs
            runs = [g[1][["step", "value"]].sort_values("step") for g in msub.groupby("run_id", dropna=False)]
            if not runs:
                continue

            grid, mean, low, high = _agg_runs_to_band(runs)
            if mean.size == 0:
                continue

            # style
            disp = str(method)
            color = METHOD_COLORS.get(disp, None)
            ls = "--" if disp in DASHED_METHODS else "-"
            lw = LINEWIDTH

            # x in thousands
            xk = grid / 1000.0

            ax.plot(xk, mean, label=disp, color=color, linestyle=ls, linewidth=lw)
            if low is not None and high is not None and color is not None:
                ax.fill_between(xk, low, high, color=color, alpha=SPREAD_ALPHA, linewidth=0)

        # cosmetics
        ax.grid(True, alpha=GRID_ALPHA, linewidth=0.8)
        ax.set_xlabel("Step (K)", fontsize=FONTS["axis"])
        ax.set_ylabel(_pretty_metric_name(metric), fontsize=FONTS["axis"])
        ax.tick_params(axis="both", labelsize=FONTS["tick"])

        # thicker spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    # legend
    handles, labels = axes[0].get_legend_handles_labels()
    if LEGEND_OUTSIDE:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5),
                   frameon=False, fontsize=FONTS["legend"], ncol=LEGEND_COLS)
    else:
        axes[0].legend(frameon=False, fontsize=FONTS["legend"], ncol=LEGEND_COLS)

    # optional title (off by default; add later in post)
    if FONTS["title"] > 0:
        fig.suptitle(scene, fontsize=FONTS["title"], y=1.02)

    # save
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, f"{scene.replace(' ', '_')}_curves")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load + concat entities
    frames = []
    for e in ENTITIES:
        frames.append(_read_entity_curves(e))
    df = pd.concat(frames, ignore_index=True)

    # Filter methods (raw names)
    df = _filter_methods(df)

    # Apply renames
    df = _apply_renames(df)

    # Filter incomplete or to step limit
    df = _maybe_filter_incomplete(df)

    # Only desired metrics
    df = df[df["metric"].isin(METRICS)].copy()

    # Optional scene whitelist
    if SCENE_WHITELIST:
        df = df[df["Scene"].isin(SCENE_WHITELIST)].copy()

    # Plot per scene
    plots_dir = os.path.join(OUT_DIR, "plots")
    for scene, sdf in df.groupby("Scene", dropna=False):
        _plot_scene(scene, sdf, plots_dir)

    print(f"[OK] Wrote plots to {plots_dir}")

if __name__ == "__main__":
    main()
