#!/usr/bin/env python3
"""
Summarize NBV metrics across multiple W&B entities, including:
  - Per-run FINAL (value at max step)
  - Per-run AUC vs random baseline (method - random), with linear interpolation

Outputs:
  - summary_all.csv            (means/stds for FINAL & AUC; all scenes/datasets)
  - counts_matrix.csv          (# runs per Scene x Method)
  - summary_<dataset>.csv      (one per dataset key: e.g., captures, tanks_and_temples, other)

Config at the top:
  - ENTITIES: list of entities to collate
  - CSV_PATH_FMT: where to read curves CSV for each entity
  - METHODS_ALLOWLIST: restrict which methods/uncertainty to include (optional)
  - METRICS: list of metric names expected in the curves CSV
  - SCENE_RENAMES, DATASET_GROUPS: rename scenes and group into datasets
  - BASELINE_SCOPE: how to form the random baseline ("scene_global" or "scene_entity")
  - AUC_SCALE: divide AUC by this (e.g., 1000.0) for nicer numbers
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# ===================== CONFIG (edit me) ======================

ENTITIES: List[str] = [
    "navlab",
    "chengine-stanford-university",
]

# Curves CSV per entity (produced by your wandb export)
# Must contain columns: scene, uncertainty, metric, step, value
CSV_PATH_FMT = "../results/{entity}/wandb_exports/curves_long.csv"

OUT_DIR = "../results/collated"

# Methods to include (None or [] => include all)
METHODS_ALLOWLIST: Optional[List[str]] = ["coverage", "fisher_info", "bayes-rays", "random", "all"]
# e.g., ["coverage", "fig", "view_fig", "random"]

# Metrics to summarize (as present in the curves CSV)
METRICS = ["/psnr", "/lpips", "/ssim"]
# Direction (only used if you later want signed AUC or ranking; here we stick to method - random)
HIGHER_IS_BETTER = {"/psnr": True, "/lpips": False, "/ssim": True}

# Scene renames (applied BEFORE dataset grouping)
SCENE_RENAMES: Dict[str, str] = {
    "chair_3pm": "chair",
    "space_laces_4pm": "space laces",
    "shiny_statue_6pm": "shiny statue",
}

# Dataset grouping by (renamed) scene
DATASET_GROUPS: Dict[str, List[str]] = {
    "tanks_and_temples": ["caterpillar", "train", "ignatius"],
    "captures": ["shiny statue", "space laces", "chair"],
    "mipnerf360": ["bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill"],
    # anything not listed falls into "other"
}

# Optional: restrict to scenes after renaming (None => keep all)
SCENE_WHITELIST: Optional[List[str]] = None

# How to build the random baseline curve for AUC:
#  - "scene_global": average all random runs for that (scene, metric) across ALL entities
#  - "scene_entity": average random runs within the SAME entity (scene, metric)
BASELINE_SCOPE = "scene_global"

# Divide AUC by this constant for nicer magnitudes (e.g., 1000.0 to match your HTML displays)
AUC_SCALE = 1000.0

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

    # Coerce types, clean
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["step", "value"])

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
        # synthesize a per-series run id
        df["_synth_key"] = df[["entity", "scene", "uncertainty"]].astype(str).agg("|".join, axis=1)
        df["run_uid"] = df["_synth_key"].factorize()[0].astype(str)
        run_col = "run_uid"
    df["run_id"] = df[run_col].astype(str)

    return df[["entity", "scene", "uncertainty", "metric", "step", "value", "run_id"]]


def _apply_scene_renames_and_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["scene_renamed"] = df["scene"].map(SCENE_RENAMES).fillna(df["scene"])
    # dataset map
    scene_to_dataset = {}
    for ds, scenes in DATASET_GROUPS.items():
        for s in scenes:
            scene_to_dataset[s] = ds
    df["dataset"] = df["scene_renamed"].map(scene_to_dataset).fillna("other")
    if SCENE_WHITELIST:
        df = df[df["scene_renamed"].isin(SCENE_WHITELIST)]
    return df


def _final_per_run(df: pd.DataFrame) -> pd.DataFrame:
    """
    FINAL per (entity, run_id, scene_renamed, uncertainty, metric) = value at max step.
    """
    grp_cols = ["entity", "run_id", "scene_renamed", "uncertainty", "metric"]
    idx = df.groupby(grp_cols, dropna=False)["step"].idxmax()
    finals = df.loc[idx, grp_cols + ["value"]].rename(columns={"value": "final"})
    return finals.reset_index(drop=True)


def _build_baselines(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build random baseline curves.

    Returns a tidy frame with columns:
      key_cols + step + baseline_value

    key_cols depend on BASELINE_SCOPE:
      - scene_global: ["scene_renamed", "metric"]
      - scene_entity: ["entity", "scene_renamed", "metric"]

    Each baseline series is the mean over all random runs (within the scope) at each *unified step grid*.
    We build the unified step grid by taking the union of steps across random runs for that key.
    """
    df = all_df[all_df["uncertainty"].str.lower() == "random"].copy()
    if df.empty:
        # no random baseline present
        return pd.DataFrame(columns=["entity", "scene_renamed", "metric", "step", "baseline_value"])

    # scope key
    if BASELINE_SCOPE == "scene_entity":
        key_cols = ["entity", "scene_renamed", "metric"]
    else:
        key_cols = ["scene_renamed", "metric"]

    baselines = []

    for key, sub in df.groupby(key_cols, dropna=False):
        # gather unique step grid from all random runs in this group
        grid = np.sort(sub["step"].unique().astype(float))
        # average across random runs at each step (if some runs are missing a step, fill by interpolation per run first)
        # -> do per-run interpolation to the union grid, then mean across runs
        run_interp = []
        for run_id, run_df in sub.groupby("run_id", dropna=False):
            xs = run_df["step"].values.astype(float)
            ys = run_df["value"].values.astype(float)
            # guard degenerate runs
            if len(xs) == 0:
                continue
            if len(np.unique(xs)) == 1:
                # constant value across grid
                y_interp = np.full_like(grid, ys[-1])
            else:
                y_interp = np.interp(grid, xs, ys)
            run_interp.append(y_interp)

        if not run_interp:
            continue
        mean_curve = np.mean(np.vstack(run_interp), axis=0)
        key_dict = dict(zip(key_cols, key if isinstance(key, tuple) else (key,)))
        out = pd.DataFrame({**key_dict, "step": grid, "baseline_value": mean_curve})
        baselines.append(out)

    if not baselines:
        return pd.DataFrame(columns=key_cols + ["step", "baseline_value"])

    return pd.concat(baselines, ignore_index=True)


def _auc_per_run(all_df: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """
    Compute AUC (method - random baseline) per run for each metric.

    Returns columns:
      entity, run_id, scene_renamed, uncertainty, metric, auc
    """
    if BASELINE_SCOPE == "scene_entity":
        base_keys = ["entity", "scene_renamed", "metric"]
    else:
        base_keys = ["scene_renamed", "metric"]

    # Make sure baseline lookups are fast
    baselines = baselines.copy()
    baselines.set_index(base_keys + ["step"], inplace=True)

    rows = []
    for (entity, run_id, scene, method, metric), sub in all_df.groupby(
        ["entity", "run_id", "scene_renamed", "uncertainty", "metric"], dropna=False
    ):
        # Skip random: define AUC = 0
        if str(method).lower() == "random":
            rows.append((entity, run_id, scene, method, metric, 0.0))
            continue

        xs = sub["step"].values.astype(float)
        ys = sub["value"].values.astype(float)
        if len(xs) < 2:
            # not enough points => auc zero
            rows.append((entity, run_id, scene, method, metric, 0.0))
            continue

        # fetch baseline series for this (scene, metric) [and entity if scene_entity]
        key = (entity, scene, metric) if BASELINE_SCOPE == "scene_entity" else (scene, metric)

        # Pull baseline steps & values
        try:
            # get all index rows matching base key
            base_slice = baselines.loc[key]
            base_steps = base_slice.index.get_level_values("step").values.astype(float)
            base_vals  = base_slice["baseline_value"].values.astype(float)
        except KeyError:
            # no baseline available => auc zero
            rows.append((entity, run_id, scene, method, metric, 0.0))
            continue

        # Common grid = union of steps
        grid = np.unique(np.concatenate([xs, base_steps]))
        # interpolate both onto grid
        y_m = np.interp(grid, xs, ys) if len(np.unique(xs)) > 1 else np.full_like(grid, ys[-1])
        y_b = np.interp(grid, base_steps, base_vals) if len(np.unique(base_steps)) > 1 else np.full_like(grid, base_vals[-1])

        diff = y_m - y_b
        auc = np.trapz(diff, grid)

        rows.append((entity, run_id, scene, method, metric, auc))

    out = pd.DataFrame(rows, columns=["entity", "run_id", "scene_renamed", "uncertainty", "metric", "auc"])
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # -------- Load + concat entities --------
    frames = [_read_entity_curves(e) for e in ENTITIES]
    df = pd.concat(frames, ignore_index=True)

    # --- Drop incomplete runs here (Option B) ---
    # Set either/both of these:
    REQUIRE_FINISHED = True        # True => keep only runs with state == "finished" (if column exists)
    MIN_FINAL_STEP   = 29000         # e.g., 20000 to require runs reached at least this step

    # If your curves CSV carries run state, filter by it.
    # (If it doesn't, this no-ops because 'state' isn't present.)
    if REQUIRE_FINISHED and "state" in df.columns:
        df = df[df["state"].astype(str).str.lower().eq("finished")]

    # Filter by maximum logged step per (entity, run_id, scene, method, metric)
    if MIN_FINAL_STEP is not None:
        df = (
            df.groupby(["entity", "run_id", "scene", "uncertainty", "metric"], dropna=False)
            .filter(lambda g: pd.to_numeric(g["step"], errors="coerce").max() >= MIN_FINAL_STEP)
            .reset_index(drop=True)
        )
    # --- End Option B filter ---


    # Filter metrics / methods
    df = df[df["metric"].isin(METRICS)].copy()
    if METHODS_ALLOWLIST:
        df = df[df["uncertainty"].isin(METHODS_ALLOWLIST)].copy()

    # Renames + dataset
    df = _apply_scene_renames_and_group(df)

    # -------- Per-run FINAL --------
    finals = _final_per_run(df)   # entity, run_id, scene_renamed, uncertainty, metric, final

    # Attach dataset mapping
    scene_ds = df[["scene_renamed", "dataset"]].drop_duplicates()
    finals = finals.merge(scene_ds, on="scene_renamed", how="left")

    # -------- Per-run AUC vs random --------
    baselines = _build_baselines(df)   # baseline curves per (scope key, step)
    auc_df = _auc_per_run(df, baselines)  # entity, run_id, scene_renamed, uncertainty, metric, auc
    # scale AUC (for nicer magnitudes, e.g., /1000)
    if AUC_SCALE and AUC_SCALE != 1.0:
        auc_df["auc"] = auc_df["auc"] / float(AUC_SCALE)

    # attach dataset
    auc_df = auc_df.merge(scene_ds, on="scene_renamed", how="left")

    # -------- Aggregate across runs: mean/std/count --------
    # FINAL
    agg_final = (
        finals.groupby(["dataset", "scene_renamed", "uncertainty", "metric"], dropna=False)["final"]
        .agg(mean="mean", std="std", n_runs="count")
        .reset_index()
    )
    # AUC
    agg_auc = (
        auc_df.groupby(["dataset", "scene_renamed", "uncertainty", "metric"], dropna=False)["auc"]
        .agg(mean="mean", std="std", n_runs="count")
        .reset_index()
    )

    # -------- Counts matrix (# runs per Scene x Method) --------
    counts = (
        finals.drop_duplicates(subset=["run_id", "scene_renamed", "uncertainty"])
              .groupby(["scene_renamed", "uncertainty"], dropna=False)["run_id"]
              .nunique()
              .unstack(fill_value=0)
              .sort_index()
    )
    counts.index.name = "Scene"
    counts.columns.name = "Method"

    # -------- Produce wide summary with FINAL & AUC --------
    metric_key = {"/psnr": "PSNR", "/lpips": "LPIPS", "/ssim": "SSIM"}

    def _widen(agg: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
        t = agg.copy()
        t["metric_key"] = t["metric"].map(metric_key).fillna(t["metric"])
        mean_w = t.pivot_table(index=["dataset", "scene_renamed", "uncertainty"],
                               columns="metric_key", values="mean")
        std_w  = t.pivot_table(index=["dataset", "scene_renamed", "uncertainty"],
                               columns="metric_key", values="std")
        n_w    = t.pivot_table(index=["dataset", "scene_renamed", "uncertainty"],
                               columns="metric_key", values="n_runs")
        mean_w.columns = [f"{prefix} {c}_mean" for c in mean_w.columns]
        std_w.columns  = [f"{prefix} {c}_std"  for c in std_w.columns]
        n_w.columns    = [f"{prefix} {c}_n"    for c in n_w.columns]
        out = pd.concat([mean_w, std_w, n_w], axis=1).reset_index()
        return out

    wide_final = _widen(agg_final, "final", "Final")
    wide_auc   = _widen(agg_auc,   "auc",   "AUC")

    summary = pd.merge(wide_final, wide_auc,
                       on=["dataset", "scene_renamed", "uncertainty"],
                       how="outer").fillna(np.nan)

    # Friendly column names
    summary = summary.rename(columns={
        "scene_renamed": "Scene",
        "uncertainty": "Method",
        "dataset": "Dataset",
    })

    # Sort rows
    order_ds = list(DATASET_GROUPS.keys()) + ["other"]
    summary["Dataset"] = pd.Categorical(summary["Dataset"], categories=order_ds, ordered=True)
    summary = summary.sort_values(["Dataset", "Scene", "Method"]).reset_index(drop=True)

    # -------- Save outputs --------
    os.makedirs(OUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUT_DIR, "summary_all.csv")
    counts_path  = os.path.join(OUT_DIR, "counts_matrix.csv")
    summary.to_csv(summary_path, index=False, float_format="%.4f")
    counts.to_csv(counts_path)

    # Per-dataset CSVs
    for ds in summary["Dataset"].dropna().unique():
        sub = summary[summary["Dataset"] == ds].copy()
        if sub.empty:
            continue
        out = os.path.join(OUT_DIR, f"summary_{ds}.csv")
        sub.to_csv(out, index=False, float_format="%.4f")

    print(f"[OK] Wrote:\n  {summary_path}\n  {counts_path}")
    for ds in summary["Dataset"].dropna().unique():
        print(f"  {os.path.join(OUT_DIR, f'summary_{ds}.csv')}")


if __name__ == "__main__":
    main()
