#!/usr/bin/env python3
"""
Unified WandB Analysis Pipeline

Combines:
- WandB data downloading (from wandb_export.py)
- Metrics aggregation (from summarize.py)
- Table generation (from build_html_from_summary.py)

Key feature: Separates runs by sweep parameters (KDTREE_FILTER, NUM_INITIAL_VIEWS,
LOAD_3D_POINTS) extracted from WandB run configs.

Usage:
    # Full pipeline
    python scripts/wandb_analysis_pipeline.py

    # Download only
    python scripts/wandb_analysis_pipeline.py --download-only

    # Aggregate existing data only
    python scripts/wandb_analysis_pipeline.py --aggregate-only

    # Filter by sweep params
    python scripts/wandb_analysis_pipeline.py --kdtree-filter false --num-initial-views 1 10

    # Force re-download
    python scripts/wandb_analysis_pipeline.py --force-reload

    # Re-check incomplete runs (below 30000 iterations)
    python scripts/wandb_analysis_pipeline.py --recheck-incomplete --min-iterations 30000
"""

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import wandb


# ===================== Configuration =====================

@dataclass
class PipelineConfig:
    entity: str = "chengine-stanford-university"
    project_prefixes: List[str] = field(default_factory=lambda: ["next-best-view__"])
    scenes: Optional[List[str]] = None
    metrics: List[str] = field(default_factory=lambda: ["/psnr", "/lpips", "/ssim"])
    auc_scale: float = 1000.0
    rate_limit_sleep: float = 0.25
    force_reload: bool = False
    download_only: bool = False
    aggregate_only: bool = False
    show_individual_scenes: bool = False
    output_format: str = "both"  # "csv", "markdown", or "both"

    # Sweep param filters (None means include all)
    kdtree_filter: Optional[List[bool]] = None
    num_initial_views: Optional[List[int]] = None
    load_3d_points: Optional[List[bool]] = None

    # Incomplete run re-checking
    min_iterations: Optional[int] = None  # Minimum iteration threshold for a run to be considered "complete"
    recheck_incomplete: bool = False  # Flag to enable re-checking of incomplete runs


# Default scenes to process
DEFAULT_SCENES = [
    "caterpillar", "train", "ignatius",
    "shiny_statue_6pm", "space_laces_4pm", "chair_3pm",
    "bicycle", "bonsai", "counter", "flowers", "garden",
    "kitchen", "room", "stump", "treehill",
]

# Metric key mapping
METRIC_KEY_MAP = {
    "/psnr": "Eval Images Metrics Dict (all images)/psnr",
    "/lpips": "Eval Images Metrics Dict (all images)/lpips",
    "/ssim": "Eval Images Metrics Dict (all images)/ssim",
}

# Direction for metrics (higher is better?)
HIGHER_IS_BETTER = {"/psnr": True, "/lpips": False, "/ssim": True}

# Run name parsing patterns
RUN_PATTERNS = [
    re.compile(r"^(?P<scene>[^_]+)__(?P<uncertainty>[^_]+)__(?P<selector>[^_]+)__(?P<ts>\d{8}-\d{4})$"),
    re.compile(r"^(?P<scene>[^_]+)__(?P<uncertainty>[^_]+)__(?P<ts>\d{8}-\d{4})$"),
]

# Scene renames for display
SCENE_RENAMES: Dict[str, str] = {
    "chair_3pm": "chair",
    "space_laces_4pm": "space laces",
    "shiny_statue_6pm": "shiny statue",
}

# Method renames for display
METHOD_RENAMES: Dict[str, str] = {
    "coverage": "Coverage (Ours)",
    "random": "Random",
    "fisher_info": "FisherRF",
    "bayes-rays": "Bayes' Rays",
    "all": "All Views",
}

# Method display order
METHOD_ORDER: List[str] = [
    "Bayes' Rays", "FisherRF", "Random", "All Views", "Coverage (Ours)",
]

# Our methods (for bold highlighting)
OUR_METHODS = ["Coverage (Ours)"]


# ===================== WandB Downloader =====================

class WandBDownloader:
    """Downloads data from WandB with sweep parameter extraction and incremental updates."""

    def __init__(self, config: PipelineConfig, out_dir: str):
        self.config = config
        self.out_dir = out_dir
        self.state_path = os.path.join(out_dir, "state.json")
        self.state = self._load_state()
        self.api = wandb.Api(timeout=60)

    def _load_state(self) -> dict:
        """Load incremental state from JSON file."""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_state(self) -> None:
        """Save state to JSON file."""
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=2, sort_keys=True)
        os.replace(tmp, self.state_path)

    def _compute_config_hash(self, run_config: dict) -> str:
        """Compute hash of relevant config fields for change detection."""
        relevant = {
            "kdtree_filter": self._extract_sweep_param(run_config, "kdtree_filter"),
            "num_initial_views": self._extract_sweep_param(run_config, "num_initial_views"),
            "load_3d_points": self._extract_sweep_param(run_config, "load_3d_points"),
        }
        config_str = json.dumps(relevant, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def _extract_sweep_param(self, config: dict, param: str) -> Optional[any]:
        """Extract sweep parameter from WandB run config."""
        try:
            if param == "kdtree_filter":
                return config.get("pipeline", {}).get("view_selection_use_kdtree_filter")
            elif param == "num_initial_views":
                return config.get("pipeline", {}).get("start_num_views")
            elif param == "load_3d_points":
                return config.get("pipeline", {}).get("datamanager", {}).get("dataparser", {}).get("load_3D_points")
        except (KeyError, TypeError, AttributeError):
            pass
        return None

    def _get_variant_name(self, kdtree: Optional[bool], init_views: Optional[int],
                          pts_3d: Optional[bool]) -> str:
        """Generate variant name from sweep parameters."""
        parts = []
        parts.append(f"kdtree_{1 if kdtree else 0}" if kdtree is not None else "kdtree_x")
        parts.append(f"init_{init_views}" if init_views is not None else "init_x")
        parts.append(f"pts_{1 if pts_3d else 0}" if pts_3d is not None else "pts_x")
        return "__".join(parts)

    def _should_process_run(self, run) -> bool:
        """Check if run needs processing based on state and force_reload."""
        if self.config.force_reload:
            return True

        rid = run.id
        updated_at = str(getattr(run, "updated_at", ""))
        config_hash = self._compute_config_hash(dict(run.config) if run.config else {})

        prev = self.state.get(rid)
        if not prev:
            return True

        # Check if we should recheck incomplete runs
        if self.config.recheck_incomplete and self.config.min_iterations is not None:
            prev_max_step = prev.get("max_step")
            if prev_max_step is not None and prev_max_step < self.config.min_iterations:
                return True  # Re-process incomplete run

        return prev.get("updated_at") != updated_at or prev.get("config_hash") != config_hash

    def _mark_processed(self, run, sweep_params: dict, max_step: Optional[int] = None) -> None:
        """Mark run as processed in state."""
        config_hash = self._compute_config_hash(dict(run.config) if run.config else {})
        self.state[run.id] = {
            "updated_at": str(getattr(run, "updated_at", "")),
            "config_hash": config_hash,
            "project": run.project,
            "name": run.name,
            "sweep_params": sweep_params,
            "max_step": max_step,
        }

    def _passes_sweep_filter(self, kdtree: Optional[bool], init_views: Optional[int],
                             pts_3d: Optional[bool]) -> bool:
        """Check if run passes sweep parameter filters."""
        if self.config.kdtree_filter is not None:
            if kdtree not in self.config.kdtree_filter:
                return False
        if self.config.num_initial_views is not None:
            if init_views not in self.config.num_initial_views:
                return False
        if self.config.load_3d_points is not None:
            if pts_3d not in self.config.load_3d_points:
                return False
        return True

    def _parse_run_name(self, name: str) -> Dict[str, Optional[str]]:
        """Parse run name into scene, uncertainty, selector, timestamp."""
        for pat in RUN_PATTERNS:
            m = pat.match(name or "")
            if m:
                d = m.groupdict()
                d.setdefault("selector", None)
                return d
        parts = (name or "").split("__")
        scene = parts[0] if parts else None
        uncertainty = parts[1] if len(parts) > 1 else None
        selector = parts[2] if len(parts) > 3 else (parts[2] if len(parts) == 4 else None)
        ts = parts[-1] if parts else None
        return {"scene": scene, "uncertainty": uncertainty, "selector": selector, "ts": ts}

    def _discover_projects(self) -> List[str]:
        """Discover projects matching any of the prefixes."""
        projects = []
        for p in self.api.projects(entity=self.config.entity):
            for prefix in self.config.project_prefixes:
                if p.name.startswith(prefix):
                    projects.append(p.name)
                    break  # Don't add same project multiple times
        return sorted(projects)

    def _project_scene(self, name: str) -> str:
        """Extract scene name from project name."""
        for prefix in self.config.project_prefixes:
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    def _union_history_keys(self, run, sample_rows: int = 200) -> set:
        """Sample history keys from a run."""
        keys = set()
        i = 0
        for row in run.scan_history():
            keys.update(row.keys())
            i += 1
            if i >= sample_rows:
                break
        return keys

    def _find_metric_keys(self, all_keys: Iterable[str]) -> Dict[str, str]:
        """Find metric keys matching desired suffixes."""
        found = {}
        for suf in self.config.metrics:
            candidates = [k for k in all_keys if k.endswith(suf) and
                         "Eval Images Metrics Dict" in k]
            if not candidates:
                candidates = [k for k in all_keys if k.endswith(suf)]
            if candidates:
                found[suf] = sorted(candidates, key=len)[0]
        return found

    def _fetch_metrics(self, run, metric_key_map: Dict[str, str]) -> pd.DataFrame:
        """Fetch metrics from run history in long format."""
        if not metric_key_map:
            return pd.DataFrame()

        wanted = list(metric_key_map.values())
        rows = []
        for row in run.scan_history(keys=wanted + ["_step", "_timestamp"]):
            step = row.get("_step")
            ts = row.get("_timestamp")
            if isinstance(ts, (int, float)):
                ts = pd.to_datetime(ts, unit="s", utc=True)
            for suf, fkey in metric_key_map.items():
                if fkey in row and row[fkey] is not None:
                    rows.append({"step": step, "timestamp": ts, "metric": suf, "value": row[fkey]})

        return pd.DataFrame(rows)

    def download(self) -> pd.DataFrame:
        """Download data from WandB and return curves dataframe with sweep params."""
        os.makedirs(self.out_dir, exist_ok=True)

        projects = self._discover_projects()
        if self.config.scenes:
            projects = [p for p in projects if self._project_scene(p) in self.config.scenes]

        if not projects:
            print("No matching projects found.")
            return pd.DataFrame()

        all_curves = []
        processed_count = 0
        skipped_count = 0

        for proj in projects:
            print(f"Processing project: {proj}")
            runs = self.api.runs(f"{self.config.entity}/{proj}")

            for run in runs:
                time.sleep(self.config.rate_limit_sleep)

                # Extract sweep params from config
                run_config = dict(run.config) if run.config else {}
                kdtree = self._extract_sweep_param(run_config, "kdtree_filter")
                init_views = self._extract_sweep_param(run_config, "num_initial_views")
                pts_3d = self._extract_sweep_param(run_config, "load_3d_points")

                # Check sweep filters
                if not self._passes_sweep_filter(kdtree, init_views, pts_3d):
                    continue

                # Check if already processed
                if not self._should_process_run(run):
                    skipped_count += 1
                    continue

                # Parse run info
                info = self._parse_run_name(run.name or "")
                scene = info.get("scene") or self._project_scene(proj)
                uncertainty = info.get("uncertainty")
                selector = info.get("selector")

                # Get variant name
                variant = self._get_variant_name(kdtree, init_views, pts_3d)

                # Discover and fetch metrics
                keys = self._union_history_keys(run, sample_rows=200)
                key_map = self._find_metric_keys(keys)

                max_step = None
                try:
                    curves = self._fetch_metrics(run, key_map)
                    if not curves.empty:
                        # Compute max_step from the fetched data
                        if "step" in curves.columns:
                            max_step = int(curves["step"].max())
                        curves.insert(0, "project", proj)
                        curves.insert(1, "scene", scene)
                        curves.insert(2, "run_id", run.id)
                        curves.insert(3, "run_name", run.name)
                        curves.insert(4, "uncertainty", uncertainty)
                        curves.insert(5, "selector", selector)
                        curves.insert(6, "variant", variant)
                        curves.insert(7, "kdtree_filter", kdtree)
                        curves.insert(8, "num_initial_views", init_views)
                        curves.insert(9, "load_3d_points", pts_3d)
                        all_curves.append(curves)
                        processed_count += 1
                except Exception as e:
                    print(f"[WARN] Failed to fetch curves for {run.id}: {e}")

                # Mark as processed with max_step
                sweep_params = {
                    "kdtree_filter": kdtree,
                    "num_initial_views": init_views,
                    "load_3d_points": pts_3d,
                }
                self._mark_processed(run, sweep_params, max_step=max_step)

        # Save state
        self._save_state()

        print(f"\nProcessed {processed_count} runs, skipped {skipped_count} unchanged runs")

        # Combine with existing data if not force_reload
        curves_csv = os.path.join(self.out_dir, "curves_with_params.csv")
        new_df = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()

        if self.config.force_reload or new_df.empty:
            combined = new_df
        else:
            if os.path.exists(curves_csv):
                try:
                    old_df = pd.read_csv(curves_csv)
                    if not new_df.empty:
                        processed_ids = set(new_df["run_id"].unique())
                        old_df = old_df[~old_df["run_id"].isin(processed_ids)]
                    combined = pd.concat([old_df, new_df], ignore_index=True)
                except Exception:
                    combined = new_df
            else:
                combined = new_df

        if not combined.empty:
            combined.to_csv(curves_csv, index=False)
            print(f"Saved: {curves_csv}")

        return combined


# ===================== Metrics Aggregator =====================

class MetricsAggregator:
    """Aggregates metrics: final values, AUC vs random baseline, max improvement."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def compute_final_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute final value (at max step) per run."""
        grp_cols = ["variant", "scene", "run_id", "uncertainty", "metric"]
        if df.empty:
            return pd.DataFrame(columns=grp_cols + ["final"])

        df = df.copy()
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["step", "value"])

        if df.empty:
            return pd.DataFrame(columns=grp_cols + ["final"])

        idx = df.groupby(grp_cols, dropna=False)["step"].idxmax()
        finals = df.loc[idx, grp_cols + ["value"]].rename(columns={"value": "final"})
        return finals.reset_index(drop=True)

    def build_baselines(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build random baseline curves averaged per (variant, scene, metric)."""
        random_df = df[df["uncertainty"].str.lower() == "random"].copy()
        if random_df.empty:
            return pd.DataFrame(columns=["variant", "scene", "metric", "step", "baseline_value"])

        random_df["step"] = pd.to_numeric(random_df["step"], errors="coerce")
        random_df["value"] = pd.to_numeric(random_df["value"], errors="coerce")
        random_df = random_df.dropna(subset=["step", "value"])

        key_cols = ["variant", "scene", "metric"]
        baselines = []

        for key, sub in random_df.groupby(key_cols, dropna=False):
            grid = np.sort(sub["step"].unique().astype(float))
            if grid.size == 0:
                continue

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

    def compute_auc(self, df: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
        """Compute AUC (method - random baseline) per run."""
        if baselines.empty:
            return pd.DataFrame(columns=["variant", "scene", "run_id", "uncertainty", "metric", "auc"])

        df = df.copy()
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["step", "value"])

        baselines = baselines.copy().set_index(["variant", "scene", "metric", "step"])

        rows = []
        for (variant, scene, run_id, method, metric), sub in df.groupby(
            ["variant", "scene", "run_id", "uncertainty", "metric"], dropna=False
        ):
            if str(method).lower() == "random":
                rows.append((variant, scene, run_id, method, metric, 0.0))
                continue

            xs = sub["step"].values.astype(float)
            ys = sub["value"].values.astype(float)
            if len(xs) < 2:
                rows.append((variant, scene, run_id, method, metric, 0.0))
                continue

            key = (variant, scene, metric)
            try:
                base_slice = baselines.loc[key]
                base_steps = base_slice.index.get_level_values("step").values.astype(float)
                base_vals = base_slice["baseline_value"].values.astype(float)
            except KeyError:
                rows.append((variant, scene, run_id, method, metric, 0.0))
                continue

            grid = np.unique(np.concatenate([xs, base_steps]))
            y_m = np.interp(grid, xs, ys) if len(np.unique(xs)) > 1 else np.full_like(grid, ys[-1])
            y_b = np.interp(grid, base_steps, base_vals) if len(np.unique(base_steps)) > 1 else np.full_like(grid, base_vals[-1])

            diff = y_m - y_b
            auc = np.trapz(diff, grid)

            rows.append((variant, scene, run_id, method, metric, auc))

        out = pd.DataFrame(rows, columns=["variant", "scene", "run_id", "uncertainty", "metric", "auc"])
        if self.config.auc_scale and self.config.auc_scale != 1.0:
            out["auc"] = out["auc"] / self.config.auc_scale
        return out

    def compute_max_improvement(self, df: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
        """Compute max improvement over baseline at any step."""
        if baselines.empty:
            return pd.DataFrame(columns=["variant", "scene", "run_id", "uncertainty", "metric", "max_improve"])

        df = df.copy()
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["step", "value"])

        baselines = baselines.copy().set_index(["variant", "scene", "metric", "step"])

        rows = []
        for (variant, scene, run_id, method, metric), sub in df.groupby(
            ["variant", "scene", "run_id", "uncertainty", "metric"], dropna=False
        ):
            if str(method).lower() == "random":
                rows.append((variant, scene, run_id, method, metric, 0.0))
                continue

            xs = sub["step"].values.astype(float)
            ys = sub["value"].values.astype(float)
            if xs.size == 0:
                rows.append((variant, scene, run_id, method, metric, 0.0))
                continue

            key = (variant, scene, metric)
            try:
                base_slice = baselines.loc[key]
                base_steps = base_slice.index.get_level_values("step").values.astype(float)
                base_vals = base_slice["baseline_value"].values.astype(float)
            except KeyError:
                rows.append((variant, scene, run_id, method, metric, 0.0))
                continue

            grid = np.unique(np.concatenate([xs, base_steps]))
            y_m = np.interp(grid, xs, ys) if len(np.unique(xs)) > 1 else np.full_like(grid, ys[-1])
            y_b = np.interp(grid, base_steps, base_vals) if len(np.unique(base_steps)) > 1 else np.full_like(grid, base_vals[-1])

            # Direction-aware: positive = improvement
            sign = 1.0 if HIGHER_IS_BETTER.get(metric, True) else -1.0
            diff = sign * (y_m - y_b)
            max_improve = float(np.max(diff)) if diff.size else 0.0

            rows.append((variant, scene, run_id, method, metric, max_improve))

        return pd.DataFrame(rows, columns=["variant", "scene", "run_id", "uncertainty", "metric", "max_improve"])

    def aggregate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregate all metrics across runs.
        Returns: (per_variant_summary, all_variants_summary)
        """
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Compute per-run metrics
        finals = self.compute_final_values(df)
        baselines = self.build_baselines(df)
        auc_df = self.compute_auc(df, baselines)
        max_df = self.compute_max_improvement(df, baselines)

        # Aggregate across runs: mean/std/count per (variant, scene, uncertainty, metric)
        def _agg(data: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
            grp_cols = ["variant", "scene", "uncertainty", "metric"]
            agg_result = (
                data.groupby(grp_cols, dropna=False)[value_col]
                .agg(mean="mean", std="std", n_runs="count")
                .reset_index()
            )
            # Pivot to wide format
            metric_key = {"/psnr": "PSNR", "/lpips": "LPIPS", "/ssim": "SSIM"}
            agg_result["metric_key"] = agg_result["metric"].map(metric_key).fillna(agg_result["metric"])

            mean_w = agg_result.pivot_table(
                index=["variant", "scene", "uncertainty"],
                columns="metric_key", values="mean"
            )
            std_w = agg_result.pivot_table(
                index=["variant", "scene", "uncertainty"],
                columns="metric_key", values="std"
            )
            n_w = agg_result.pivot_table(
                index=["variant", "scene", "uncertainty"],
                columns="metric_key", values="n_runs"
            )

            mean_w.columns = [f"{prefix}_{c}_mean" for c in mean_w.columns]
            std_w.columns = [f"{prefix}_{c}_std" for c in std_w.columns]
            n_w.columns = [f"{prefix}_{c}_n" for c in n_w.columns]

            return pd.concat([mean_w, std_w, n_w], axis=1).reset_index()

        wide_final = _agg(finals, "final", "Final")
        wide_auc = _agg(auc_df, "auc", "AUC")
        wide_max = _agg(max_df, "max_improve", "MaxDelta")

        # Merge
        summary = pd.merge(wide_final, wide_auc, on=["variant", "scene", "uncertainty"], how="outer")
        summary = pd.merge(summary, wide_max, on=["variant", "scene", "uncertainty"], how="outer")

        # Rename columns
        summary = summary.rename(columns={
            "scene": "Scene",
            "uncertainty": "Method",
            "variant": "Variant",
        })

        return summary


# ===================== Table Generator =====================

class TableGenerator:
    """Generates CSV and markdown tables from aggregated metrics."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.metrics = ["PSNR", "SSIM", "LPIPS"]

    def _format_value(self, val, decimals: int = 2) -> str:
        """Format a numeric value for display."""
        try:
            x = float(val)
            if np.isnan(x):
                return "-"
            return f"{x:.{decimals}f}"
        except Exception:
            return "-"

    def _rank_series(self, s: pd.Series, higher: bool) -> pd.Series:
        """Rank values (1 = best)."""
        return s.rank(ascending=not higher, method="min")

    def _apply_renames(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scene and method renames."""
        df = df.copy()
        if "Scene" in df.columns:
            df["Scene"] = df["Scene"].map(SCENE_RENAMES).fillna(df["Scene"])
        if "Method" in df.columns:
            df["Method"] = df["Method"].map(METHOD_RENAMES).fillna(df["Method"])
            # Apply method ordering
            existing = [m for m in METHOD_ORDER if m in df["Method"].values]
            others = [m for m in df["Method"].unique() if m not in METHOD_ORDER]
            final_order = existing + sorted(others)
            df["Method"] = pd.Categorical(df["Method"], categories=final_order, ordered=True)
        return df

    def generate_csv(self, df: pd.DataFrame, out_path: str) -> None:
        """Generate CSV output."""
        if df.empty:
            print(f"No data to write to {out_path}")
            return

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False, float_format="%.4f")
        print(f"Saved: {out_path}")

    def generate_markdown(self, df: pd.DataFrame, out_path: str, title: str = "Summary") -> None:
        """Generate markdown table output."""
        if df.empty:
            print(f"No data to write to {out_path}")
            return

        df = self._apply_renames(df.copy())

        # Sort by scene and method
        if "Scene" in df.columns:
            df = df.sort_values(["Scene", "Method"])

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        md_content = self._render_markdown(df, title)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"Saved: {out_path}")

    def _render_markdown(self, df: pd.DataFrame, title: str) -> str:
        """Render markdown table."""
        final_col = "Final_{m}_mean"
        auc_col = "AUC_{m}_mean"

        # Ensure numeric columns
        for m in self.metrics:
            for col in [final_col.format(m=m), auc_col.format(m=m)]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        lines = [f"# {title}", ""]

        # Build header
        header_parts = ["Scene", "Method"]
        for m in self.metrics:
            arrow = "↑" if HIGHER_IS_BETTER.get(f"/{m.lower()}", True) else "↓"
            header_parts.append(f"Final {m} {arrow}")
            header_parts.append(f"AUC {m} {arrow}")

        lines.append("| " + " | ".join(header_parts) + " |")
        lines.append("|" + "|".join(["---"] * len(header_parts)) + "|")

        # Per-scene rows
        scenes = list(dict.fromkeys(df["Scene"].tolist())) if "Scene" in df.columns else []

        for scene in scenes:
            scene_df = df[df["Scene"] == scene].copy()
            if pd.api.types.is_categorical_dtype(scene_df["Method"]):
                scene_df = scene_df.sort_values("Method")

            # Compute rankings for this scene
            ranks_final, ranks_auc = {}, {}
            for m in self.metrics:
                higher = HIGHER_IS_BETTER.get(f"/{m.lower()}", True)
                if final_col.format(m=m) in scene_df.columns:
                    ranks_final[m] = self._rank_series(scene_df[final_col.format(m=m)], higher)
                if auc_col.format(m=m) in scene_df.columns:
                    ranks_auc[m] = self._rank_series(scene_df[auc_col.format(m=m)], higher)

            for ridx, (_, row) in enumerate(scene_df.iterrows()):
                row_parts = []
                row_parts.append(scene if ridx == 0 else "")
                row_parts.append(str(row.get("Method", "")))

                for m in self.metrics:
                    v_final = row.get(final_col.format(m=m), np.nan)
                    v_auc = row.get(auc_col.format(m=m), np.nan)

                    is_best_final = m in ranks_final and ridx < len(ranks_final[m]) and ranks_final[m].iloc[ridx] == 1
                    is_best_auc = m in ranks_auc and ridx < len(ranks_auc[m]) and ranks_auc[m].iloc[ridx] == 1

                    # Mark best with **bold**
                    final_str = self._format_value(v_final)
                    auc_str = self._format_value(v_auc)

                    if is_best_final:
                        final_str = f"**{final_str}**"
                    if is_best_auc:
                        auc_str = f"**{auc_str}**"

                    row_parts.append(final_str)
                    row_parts.append(auc_str)

                lines.append("| " + " | ".join(row_parts) + " |")

        # Averages section
        lines.append("")
        lines.append("## Averages")
        lines.append("")
        lines.append("| " + " | ".join(["Method"] + [f"Final {m}" for m in self.metrics] + [f"AUC {m}" for m in self.metrics]) + " |")
        lines.append("|" + "|".join(["---"] * (1 + len(self.metrics) * 2)) + "|")

        # Compute averages
        agg_cols = []
        for m in self.metrics:
            if final_col.format(m=m) in df.columns:
                agg_cols.append(final_col.format(m=m))
            if auc_col.format(m=m) in df.columns:
                agg_cols.append(auc_col.format(m=m))

        if agg_cols and "Method" in df.columns:
            agg = df.groupby("Method", dropna=False)[agg_cols].mean(numeric_only=True).reset_index()
            if pd.api.types.is_categorical_dtype(agg["Method"]):
                agg = agg.sort_values("Method")

            # Compute rankings for averages
            ranks_final_avg, ranks_auc_avg = {}, {}
            for m in self.metrics:
                higher = HIGHER_IS_BETTER.get(f"/{m.lower()}", True)
                if final_col.format(m=m) in agg.columns:
                    ranks_final_avg[m] = self._rank_series(agg[final_col.format(m=m)], higher)
                if auc_col.format(m=m) in agg.columns:
                    ranks_auc_avg[m] = self._rank_series(agg[auc_col.format(m=m)], higher)

            for idx, (_, row) in enumerate(agg.iterrows()):
                row_parts = [str(row["Method"])]

                # Finals first
                for m in self.metrics:
                    vf = row.get(final_col.format(m=m), np.nan)
                    is_best = m in ranks_final_avg and idx < len(ranks_final_avg[m]) and ranks_final_avg[m].iloc[idx] == 1
                    val_str = self._format_value(vf)
                    if is_best:
                        val_str = f"**{val_str}**"
                    row_parts.append(val_str)

                # AUCs second
                for m in self.metrics:
                    va = row.get(auc_col.format(m=m), np.nan)
                    is_best = m in ranks_auc_avg and idx < len(ranks_auc_avg[m]) and ranks_auc_avg[m].iloc[idx] == 1
                    val_str = self._format_value(va)
                    if is_best:
                        val_str = f"**{val_str}**"
                    row_parts.append(val_str)

                lines.append("| " + " | ".join(row_parts) + " |")

        return "\n".join(lines) + "\n"


# ===================== Main Pipeline =====================

def run_pipeline(config: PipelineConfig) -> None:
    """Run the full analysis pipeline."""
    base_dir = f"../results/{config.entity}"
    wandb_dir = os.path.join(base_dir, "wandb_exports")
    analysis_dir = os.path.join(base_dir, "analysis")

    curves_df = None

    # Step 1: Download (unless aggregate-only)
    if not config.aggregate_only:
        print("\n=== Step 1: Downloading from WandB ===")
        downloader = WandBDownloader(config, wandb_dir)
        curves_df = downloader.download()
    else:
        # Load existing data
        curves_csv = os.path.join(wandb_dir, "curves_with_params.csv")
        if os.path.exists(curves_csv):
            curves_df = pd.read_csv(curves_csv)
            print(f"Loaded existing data from {curves_csv}")
        else:
            print(f"No existing data found at {curves_csv}")
            return

    if config.download_only:
        print("\n=== Download complete (--download-only) ===")
        return

    if curves_df is None or curves_df.empty:
        print("No data to aggregate.")
        return

    # Step 2: Aggregate metrics
    print("\n=== Step 2: Aggregating Metrics ===")
    aggregator = MetricsAggregator(config)
    summary_df = aggregator.aggregate(curves_df)

    if summary_df.empty:
        print("No aggregated data.")
        return

    # Step 3: Generate tables
    print("\n=== Step 3: Generating Tables ===")
    generator = TableGenerator(config)

    # Get unique variants
    variants = summary_df["Variant"].unique() if "Variant" in summary_df.columns else ["default"]

    for variant in variants:
        variant_dir = os.path.join(analysis_dir, "by_variant", variant)
        os.makedirs(variant_dir, exist_ok=True)

        variant_df = summary_df[summary_df["Variant"] == variant] if "Variant" in summary_df.columns else summary_df

        if config.output_format in ["csv", "both"]:
            generator.generate_csv(variant_df, os.path.join(variant_dir, "summary.csv"))

        if config.output_format in ["markdown", "both"]:
            generator.generate_markdown(variant_df, os.path.join(variant_dir, "summary.md"),
                                        title=f"Summary - {variant}")

    # Generate combined summary
    tables_dir = os.path.join(analysis_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    if config.output_format in ["csv", "both"]:
        generator.generate_csv(summary_df, os.path.join(tables_dir, "summary_all.csv"))

    if config.output_format in ["markdown", "both"]:
        generator.generate_markdown(summary_df, os.path.join(tables_dir, "summary_all.md"),
                                    title="Summary - All Variants")

    print("\n=== Pipeline Complete ===")


def parse_bool(value: str) -> bool:
    """Parse boolean string."""
    return value.lower() in ("true", "1", "yes", "t")


def main():
    parser = argparse.ArgumentParser(
        description="Unified WandB Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python scripts/wandb_analysis_pipeline.py

  # Specify project prefixes (matches PROJECT_NAME in sweep_scenes.py)
  python scripts/wandb_analysis_pipeline.py --project-prefixes next-best-view-rebuttal__

  # Multiple prefixes
  python scripts/wandb_analysis_pipeline.py --project-prefixes next-best-view__ next-best-view-rebuttal__

  # Download only
  python scripts/wandb_analysis_pipeline.py --download-only

  # Aggregate existing data only
  python scripts/wandb_analysis_pipeline.py --aggregate-only

  # Filter by sweep params
  python scripts/wandb_analysis_pipeline.py --kdtree-filter false --num-initial-views 1 10

  # Force re-download
  python scripts/wandb_analysis_pipeline.py --force-reload

  # Re-check incomplete runs (those below 30000 iterations)
  python scripts/wandb_analysis_pipeline.py --recheck-incomplete --min-iterations 30000

  # Combine with other options
  python scripts/wandb_analysis_pipeline.py --recheck-incomplete --min-iterations 30000 --aggregate-only
        """
    )

    parser.add_argument("--entity", default="chengine-stanford-university",
                       help="WandB entity name")
    parser.add_argument("--project-prefixes", nargs="+", default=["next-best-view__"],
                       help="Project prefix(es) to filter (e.g., 'next-best-view-rebuttal__')")
    parser.add_argument("--scenes", nargs="+", default=None,
                       help="Specific scenes to process")
    parser.add_argument("--metrics", nargs="+", default=["/psnr", "/lpips", "/ssim"],
                       help="Metrics to aggregate")
    parser.add_argument("--auc-scale", type=float, default=1000.0,
                       help="Scale factor for AUC values")

    parser.add_argument("--download-only", action="store_true",
                       help="Only download data, skip aggregation")
    parser.add_argument("--aggregate-only", action="store_true",
                       help="Only aggregate existing data, skip download")
    parser.add_argument("--force-reload", action="store_true",
                       help="Force re-download all runs")
    parser.add_argument("--show-individual-scenes", action="store_true",
                       help="Show individual scene rows in output")

    parser.add_argument("--min-iterations", type=int, default=None,
                       help="Minimum iterations for a run to be considered complete")
    parser.add_argument("--recheck-incomplete", action="store_true",
                       help="Re-check runs that were below min-iterations threshold")

    parser.add_argument("--format", dest="output_format", choices=["csv", "markdown", "both"],
                       default="both", help="Output format (csv, markdown, or both)")

    # Sweep parameter filters
    parser.add_argument("--kdtree-filter", nargs="+", type=str, default=None,
                       help="Filter by kdtree_filter values (true/false)")
    parser.add_argument("--num-initial-views", nargs="+", type=int, default=None,
                       help="Filter by num_initial_views values")
    parser.add_argument("--load-3d-points", nargs="+", type=str, default=None,
                       help="Filter by load_3d_points values (true/false)")

    args = parser.parse_args()

    # Build config
    config = PipelineConfig(
        entity=args.entity,
        project_prefixes=args.project_prefixes,
        scenes=args.scenes,
        metrics=args.metrics,
        auc_scale=args.auc_scale,
        force_reload=args.force_reload,
        download_only=args.download_only,
        aggregate_only=args.aggregate_only,
        show_individual_scenes=args.show_individual_scenes,
        output_format=args.output_format,
        min_iterations=args.min_iterations,
        recheck_incomplete=args.recheck_incomplete,
    )

    # Parse sweep filters
    if args.kdtree_filter:
        config.kdtree_filter = [parse_bool(v) for v in args.kdtree_filter]
    if args.num_initial_views:
        config.num_initial_views = args.num_initial_views
    if args.load_3d_points:
        config.load_3d_points = [parse_bool(v) for v in args.load_3d_points]

    run_pipeline(config)


if __name__ == "__main__":
    main()
