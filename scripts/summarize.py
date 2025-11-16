#!/usr/bin/env python3
"""
Summarize NBV metrics across multiple W&B entities, including:
  - Per-run FINAL (value at max step or max step <= cutoff)
  - Per-run AUC vs random baseline (method - random), with linear interpolation
  - Per-run MAX IMPROVEMENT over random baseline at any step (direction-aware by default)

Outputs (FULL, i.e., no cutoff):
  - summary_all.csv            (means/stds for FINAL, AUC, MAXΔ; all scenes/datasets)
  - counts_matrix.csv          (# runs per Scene x Method)
  - summary_<dataset>.csv      (one per dataset key)

Outputs (for each cutoff S in STEP_CUTOFFS):
  - summary_all_upto_S.csv
  - counts_matrix_upto_S.csv
  - summary_<dataset>_upto_S.csv

Config at the top:
  - ENTITIES, CSV_PATH_FMT, METHODS_ALLOWLIST, METRICS, renames, dataset groups
  - BASELINE_SCOPE: "scene_global" or "scene_entity"
  - AUC_SCALE: divide AUC by this (e.g., 1000.0) for nicer numbers
  - STEP_CUTOFFS: e.g., [15000, 25000, 30000]
  - IMPROVEMENT_DIRECTION_AWARE: True => uses sign that makes "bigger is better"
  - MAX_IMPROVE_SCALE: optional display scaling for MaxΔ (default 1.0)
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

# Metrics to summarize (as present in the curves CSV)
METRICS = ["/psnr", "/lpips", "/ssim"]
# Direction (used for direction-aware MaxΔ; AUC stays method - random by design)
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

# How to build the random baseline curve for AUC & MaxΔ:
#  - "scene_global": average all random runs for that (scene, metric) across ALL entities
#  - "scene_entity": average random runs within the SAME entity (scene, metric)
BASELINE_SCOPE = "scene_global"

# Divide AUC by this constant for nicer magnitudes (e.g., 1000.0 to match your HTML displays)
AUC_SCALE = 1000.0

# ==== NEW: compute summaries up to these step cutoffs (in addition to full) ====
STEP_CUTOFFS: List[int] = [5000, 10000, 15000, 20000, 25000, 29000]

# ==== NEW: Max improvement over random options ====
# If True: diff = method - random for higher-is-better metrics; diff = random - method for lower-is-better
# If False: diff = method - random for all metrics (like AUC)
IMPROVEMENT_DIRECTION_AWARE = False
# Optional display scaling for MaxΔ
MAX_IMPROVE_SCALE = 1.0

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


def _filter_by_cutoff(df: pd.DataFrame, cutoff: Optional[int]) -> pd.DataFrame:
    """Return rows with step <= cutoff. If cutoff is None, return df."""
    if cutoff is None:
        return df
    d = df[df["step"] <= cutoff].copy()
    # drop (entity, run_id, scene_renamed, uncertainty, metric) groups that became empty
    # (groupby/filter not needed because the boolean mask already does that)
    return d


def _final_per_run(df: pd.DataFrame) -> pd.DataFrame:
    """
    FINAL per (entity, run_id, scene_renamed, uncertainty, metric) = value at max step (within df).
    """
    grp_cols = ["entity", "run_id", "scene_renamed", "uncertainty", "metric"]
    if df.empty:
        return pd.DataFrame(columns=grp_cols + ["final"])
    idx = df.groupby(grp_cols, dropna=False)["step"].idxmax()
    finals = df.loc[idx, grp_cols + ["value"]].rename(columns={"value": "final"})
    return finals.reset_index(drop=True)


def _build_baselines(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build random baseline curves (within the provided df, which may already be cutoff-filtered).

    Returns a tidy frame with columns:
      key_cols + step + baseline_value

    key_cols depend on BASELINE_SCOPE:
      - scene_global: ["scene_renamed", "metric"]
      - scene_entity: ["entity", "scene_renamed", "metric"]

    Each baseline series is the mean over all random runs (within the scope) at each *unified step grid*.
    """
    df = all_df[all_df["uncertainty"].str.lower() == "random"].copy()
    if df.empty:
        # no random baseline present
        if BASELINE_SCOPE == "scene_entity":
            cols = ["entity", "scene_renamed", "metric", "step", "baseline_value"]
        else:
            cols = ["scene_renamed", "metric", "step", "baseline_value"]
        return pd.DataFrame(columns=cols)

    # scope key
    key_cols = ["entity", "scene_renamed", "metric"] if BASELINE_SCOPE == "scene_entity" else ["scene_renamed", "metric"]

    baselines = []

    for key, sub in df.groupby(key_cols, dropna=False):
        # gather unique step grid from all random runs in this group
        grid = np.sort(sub["step"].unique().astype(float))
        if grid.size == 0:
            continue
        # average across random runs at each step: per-run interp to union grid, then mean
        run_interp = []
        for _, run_df in sub.groupby("run_id", dropna=False):
            xs = run_df["step"].values.astype(float)
            ys = run_df["value"].values.astype(float)
            if len(xs) == 0:
                continue
            if len(np.unique(xs)) == 1:
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
    Compute AUC (method - random baseline) per run for each metric (within provided df).
    Returns: entity, run_id, scene_renamed, uncertainty, metric, auc
    """
    base_keys = ["entity", "scene_renamed", "metric"] if BASELINE_SCOPE == "scene_entity" \
                else ["scene_renamed", "metric"]

    if baselines.empty:
        return pd.DataFrame(columns=["entity", "run_id", "scene_renamed", "uncertainty", "metric", "auc"])

    baselines = baselines.copy().set_index(base_keys + ["step"])

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
            rows.append((entity, run_id, scene, method, metric, 0.0))
            continue

        key = (entity, scene, metric) if BASELINE_SCOPE == "scene_entity" else (scene, metric)

        try:
            base_slice = baselines.loc[key]
            base_steps = base_slice.index.get_level_values("step").values.astype(float)
            base_vals  = base_slice["baseline_value"].values.astype(float)
        except KeyError:
            rows.append((entity, run_id, scene, method, metric, 0.0))
            continue

        grid = np.unique(np.concatenate([xs, base_steps]))
        y_m = np.interp(grid, xs, ys) if len(np.unique(xs)) > 1 else np.full_like(grid, ys[-1])
        y_b = np.interp(grid, base_steps, base_vals) if len(np.unique(base_steps)) > 1 else np.full_like(grid, base_vals[-1])

        diff = y_m - y_b
        auc = np.trapz(diff, grid)

        rows.append((entity, run_id, scene, method, metric, auc))

    out = pd.DataFrame(rows, columns=["entity", "run_id", "scene_renamed", "uncertainty", "metric", "auc"])
    return out


def _max_improvement_per_run(all_df: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MaxΔ over random baseline at any step per run/metric.
    Direction-aware if IMPROVEMENT_DIRECTION_AWARE:
      - if HIGHER_IS_BETTER[metric] is True:  diff = method - baseline
      - else (lower better):                  diff = baseline - method
    Random runs get MaxΔ = 0.
    Returns: entity, run_id, scene_renamed, uncertainty, metric, max_improve
    """
    base_keys = ["entity", "scene_renamed", "metric"] if BASELINE_SCOPE == "scene_entity" \
                else ["scene_renamed", "metric"]

    if baselines.empty:
        return pd.DataFrame(columns=["entity", "run_id", "scene_renamed", "uncertainty", "metric", "max_improve"])

    baselines = baselines.copy().set_index(base_keys + ["step"])

    rows = []
    for (entity, run_id, scene, method, metric), sub in all_df.groupby(
        ["entity", "run_id", "scene_renamed", "uncertainty", "metric"], dropna=False
    ):
        if str(method).lower() == "random":
            rows.append((entity, run_id, scene, method, metric, 0.0))
            continue

        xs = sub["step"].values.astype(float)
        ys = sub["value"].values.astype(float)
        if xs.size == 0:
            rows.append((entity, run_id, scene, method, metric, 0.0))
            continue

        key = (entity, scene, metric) if BASELINE_SCOPE == "scene_entity" else (scene, metric)
        try:
            base_slice = baselines.loc[key]
            base_steps = base_slice.index.get_level_values("step").values.astype(float)
            base_vals  = base_slice["baseline_value"].values.astype(float)
        except KeyError:
            rows.append((entity, run_id, scene, method, metric, 0.0))
            continue

        grid = np.unique(np.concatenate([xs, base_steps]))
        y_m = np.interp(grid, xs, ys) if len(np.unique(xs)) > 1 else np.full_like(grid, ys[-1])
        y_b = np.interp(grid, base_steps, base_vals) if len(np.unique(base_steps)) > 1 else np.full_like(grid, base_vals[-1])

        if IMPROVEMENT_DIRECTION_AWARE:
            # choose sign so that larger = better improvement
            sign = 1.0 if HIGHER_IS_BETTER.get(metric, True) else -1.0
            diff = sign * (y_m - y_b)
        else:
            # raw method - random (like AUC)
            diff = (y_m - y_b)

        max_improve = float(np.max(diff)) if diff.size else 0.0
        rows.append((entity, run_id, scene, method, metric, max_improve))

    out = pd.DataFrame(rows, columns=["entity", "run_id", "scene_renamed", "uncertainty", "metric", "max_improve"])
    return out


def _aggregate_and_write(
    df_in: pd.DataFrame,
    out_dir: str,
    suffix: str = ""  # e.g., "_upto_15000"
):
    """Core aggregation & writing for a given (possibly cutoff-filtered) dataframe."""
    os.makedirs(out_dir, exist_ok=True)

    # Filter metrics / methods
    df = df_in[df_in["metric"].isin(METRICS)].copy()
    if METHODS_ALLOWLIST:
        df = df[df["uncertainty"].isin(METHODS_ALLOWLIST)].copy()

    # Renames + dataset
    df = _apply_scene_renames_and_group(df)

    # -------- Per-run FINAL --------
    finals = _final_per_run(df)   # entity, run_id, scene_renamed, uncertainty, metric, final

    # Attach dataset mapping
    scene_ds = df[["scene_renamed", "dataset"]].drop_duplicates()
    finals = finals.merge(scene_ds, on="scene_renamed", how="left")

    # -------- Build baselines (from the same filtered df) --------
    baselines = _build_baselines(df)

    # -------- Per-run AUC vs random --------
    auc_df = _auc_per_run(df, baselines)  # entity, run_id, scene_renamed, uncertainty, metric, auc
    if AUC_SCALE and AUC_SCALE != 1.0:
        auc_df["auc"] = auc_df["auc"] / float(AUC_SCALE)
    auc_df = auc_df.merge(scene_ds, on="scene_renamed", how="left")

    # -------- Per-run Max Improvement vs random --------
    max_df = _max_improvement_per_run(df, baselines)  # entity, run_id, scene_renamed, uncertainty, metric, max_improve
    if MAX_IMPROVE_SCALE and MAX_IMPROVE_SCALE != 1.0:
        max_df["max_improve"] = max_df["max_improve"] / float(MAX_IMPROVE_SCALE)
    max_df = max_df.merge(scene_ds, on="scene_renamed", how="left")

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
    # MAXΔ
    agg_max = (
        max_df.groupby(["dataset", "scene_renamed", "uncertainty", "metric"], dropna=False)["max_improve"]
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

    # -------- Produce wide summary with FINAL, AUC, MAXΔ --------
    metric_key = {"/psnr": "PSNR", "/lpips": "LPIPS", "/ssim": "SSIM"}

    def _widen(agg: pd.DataFrame, value_name: str, prefix: str) -> pd.DataFrame:
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
    wide_max   = _widen(agg_max,   "max_improve", "MaxΔ")

    summary = pd.merge(wide_final, wide_auc,
                       on=["dataset", "scene_renamed", "uncertainty"],
                       how="outer")
    summary = pd.merge(summary, wide_max,
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
    os.makedirs(out_dir, exist_ok=True)

    base_summary = os.path.join(out_dir, f"summary_all{suffix}.csv")
    base_counts  = os.path.join(out_dir, f"counts_matrix{suffix}.csv")
    summary.to_csv(base_summary, index=False, float_format="%.4f")
    counts.to_csv(base_counts)

    # Per-dataset CSVs
    for ds in summary["Dataset"].dropna().unique():
        sub = summary[summary["Dataset"] == ds].copy()
        if sub.empty:
            continue
        out = os.path.join(out_dir, f"summary_{ds}{suffix}.csv")
        sub.to_csv(out, index=False, float_format="%.4f")

    print(f"[OK] Wrote:")
    print(f"  {base_summary}")
    print(f"  {base_counts}")
    for ds in summary["Dataset"].dropna().unique():
        print(f"  {os.path.join(out_dir, f'summary_{ds}{suffix}.csv')}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # -------- Load + concat entities --------
    frames = [_read_entity_curves(e) for e in ENTITIES]
    df = pd.concat(frames, ignore_index=True)

    # --- Drop incomplete runs here (Option B) ---
    REQUIRE_FINISHED = True   # True => keep only runs with state == "finished" (if column exists)
    MIN_FINAL_STEP   = 29000  # e.g., 20000 to require runs reached at least this step

    if REQUIRE_FINISHED and "state" in df.columns:
        df = df[df["state"].astype(str).str.lower().eq("finished")]

    if MIN_FINAL_STEP is not None:
        df = (
            df.groupby(["entity", "run_id", "scene", "uncertainty", "metric"], dropna=False)
              .filter(lambda g: pd.to_numeric(g["step"], errors="coerce").max() >= MIN_FINAL_STEP)
              .reset_index(drop=True)
        )
    # --- End Option B filter ---

    # ==== FULL (no cutoff) ====
    _aggregate_and_write(df_in=_apply_scene_renames_and_group(df), out_dir=OUT_DIR, suffix="")

    # ==== STEP-CUTOFF LOOP ====
    for cutoff in STEP_CUTOFFS:
        df_c = _filter_by_cutoff(_apply_scene_renames_and_group(df), cutoff)
        # Note: _aggregate_and_write expects raw cols incl. scene_renamed; df is already renamed/grouped above
        # But it calls _apply_scene_renames_and_group again; to avoid duplicate work, we pass df BEFORE renaming.
        # For clarity, we'll pass the un-renamed df through the function (same as full), but filter first:
        df_cut = _filter_by_cutoff(df, cutoff)
        suffix = f"_upto_{int(cutoff)}"
        _aggregate_and_write(df_in=df_cut, out_dir=OUT_DIR, suffix=suffix)


if __name__ == "__main__":
    main()
