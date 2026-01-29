#!/usr/bin/env python3
"""
Analyze Locally Stored WandB Curves

Reads locally stored curves_with_params.csv (downloaded via wandb_analysis_pipeline.py),
filters runs by minimum step threshold to exclude incomplete/running experiments,
computes metrics (Final, AUC, MaxDelta), and generates summary tables.

Usage:
    # Basic usage (filter to runs with max step >= 30000)
    python scripts/analyze_local_curves.py

    # Custom min step threshold
    python scripts/analyze_local_curves.py --min-step 25000

    # Filter by sweep params
    python scripts/analyze_local_curves.py --num-initial-views 1 10 --kdtree-filter true

    # Group output by specific parameters
    python scripts/analyze_local_curves.py --group-by kdtree_filter num_initial_views

    # Custom input/output paths
    python scripts/analyze_local_curves.py --input path/to/curves.csv --output-dir path/to/output

    # Specific scenes only
    python scripts/analyze_local_curves.py --scenes caterpillar train
"""

import argparse
import os
from typing import List, Optional, Tuple

import pandas as pd

from wandb_analysis_pipeline import (
    MetricsAggregator,
    PipelineConfig,
    TableGenerator,
)


def filter_by_min_step(
    df: pd.DataFrame, min_step: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter runs where max step >= min_step.

    Args:
        df: DataFrame with curves data
        min_step: Minimum step threshold

    Returns:
        Tuple of (filtered_df, excluded_runs_df)
    """
    if df.empty:
        return df, pd.DataFrame()

    # Compute max step per run
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    max_steps = df.groupby("run_id")["step"].max().reset_index()
    max_steps.columns = ["run_id", "max_step"]

    # Split into valid and excluded
    valid_run_ids = max_steps[max_steps["max_step"] >= min_step]["run_id"]
    excluded_run_ids = max_steps[max_steps["max_step"] < min_step]["run_id"]

    # Get excluded runs info for reporting
    excluded_info = max_steps[max_steps["run_id"].isin(excluded_run_ids)].copy()
    if not excluded_info.empty:
        # Add run names and scenes
        run_info = df[["run_id", "run_name", "scene"]].drop_duplicates()
        excluded_info = excluded_info.merge(run_info, on="run_id", how="left")

    filtered_df = df[df["run_id"].isin(valid_run_ids)]

    return filtered_df, excluded_info


def apply_sweep_filters(
    df: pd.DataFrame,
    kdtree_filter: Optional[List[bool]] = None,
    num_initial_views: Optional[List[int]] = None,
    load_3d_points: Optional[List[bool]] = None,
) -> pd.DataFrame:
    """Apply sweep parameter filters to the dataframe."""
    if df.empty:
        return df

    mask = pd.Series([True] * len(df), index=df.index)

    if kdtree_filter is not None and "kdtree_filter" in df.columns:
        mask &= df["kdtree_filter"].isin(kdtree_filter)

    if num_initial_views is not None and "num_initial_views" in df.columns:
        mask &= df["num_initial_views"].isin(num_initial_views)

    if load_3d_points is not None and "load_3d_points" in df.columns:
        mask &= df["load_3d_points"].isin(load_3d_points)

    return df[mask]


def apply_scene_filter(df: pd.DataFrame, scenes: Optional[List[str]]) -> pd.DataFrame:
    """Filter to specific scenes."""
    if scenes is None or df.empty:
        return df
    return df[df["scene"].isin(scenes)]


def print_sweep_breakdown(df: pd.DataFrame) -> dict:
    """Print breakdown of runs by sweep parameters. Returns variant->run_count dict."""
    if df.empty:
        return {}

    print("\nSweep Parameter Breakdown:")

    # Get unique runs with their parameters
    param_cols = ["kdtree_filter", "num_initial_views", "load_3d_points"]
    available_cols = [c for c in param_cols if c in df.columns]
    run_params = df[["run_id"] + available_cols].drop_duplicates()
    total_runs = len(run_params)

    print(f"  Total runs: {total_runs}")

    # Count by each parameter
    if "kdtree_filter" in df.columns:
        kdtree_counts = run_params["kdtree_filter"].value_counts().to_dict()
        kdtree_str = ", ".join([f"{k}={v}" for k, v in sorted(kdtree_counts.items(), key=lambda x: str(x[0]))])
        print(f"  kdtree_filter: {kdtree_str}")

    if "num_initial_views" in df.columns:
        init_counts = run_params["num_initial_views"].value_counts().to_dict()
        init_str = ", ".join([f"{k}={v}" for k, v in sorted(init_counts.items(), key=lambda x: str(x[0]))])
        print(f"  num_initial_views: {init_str}")

    if "load_3d_points" in df.columns:
        pts_counts = run_params["load_3d_points"].value_counts().to_dict()
        pts_str = ", ".join([f"{k}={v}" for k, v in sorted(pts_counts.items(), key=lambda x: str(x[0]))])
        print(f"  load_3d_points: {pts_str}")

    # Count by combination (variant)
    variant_run_counts = {}
    print("\n  By combination (variant):")
    if "variant" in df.columns:
        variant_counts = run_params.merge(
            df[["run_id", "variant"]].drop_duplicates(), on="run_id"
        )["variant"].value_counts()
        for variant, count in sorted(variant_counts.items()):
            print(f"    {variant}: {count} runs")
            variant_run_counts[variant] = count

    return variant_run_counts


def get_group_key(row: pd.Series, group_by: List[str]) -> str:
    """Generate a group key from the specified columns."""
    parts = []
    for col in group_by:
        val = row.get(col, "unknown")
        if pd.isna(val):
            val = "none"
        parts.append(f"{col}={val}")
    return "__".join(parts)


def generate_variant_summary(
    summary_df: pd.DataFrame, variant_run_counts: dict, output_dir: str, output_format: str
) -> None:
    """
    Generate a summary table for each variant, averaging metrics across all scenes.
    Shows per-method statistics for each variant configuration.
    """
    import numpy as np

    if summary_df.empty or "Variant" not in summary_df.columns:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Identify metric columns (Final_*, AUC_*, MaxDelta_*)
    metric_cols = [c for c in summary_df.columns if any(
        c.startswith(prefix) for prefix in ["Final_", "AUC_", "MaxDelta_"]
    ) and c.endswith("_mean")]

    variants = sorted(summary_df["Variant"].unique())

    # Generate individual variant summaries
    all_variant_summaries = []

    for variant in variants:
        variant_df = summary_df[summary_df["Variant"] == variant].copy()
        run_count = variant_run_counts.get(variant, 0)
        n_scenes = variant_df["Scene"].nunique() if "Scene" in variant_df.columns else 0

        # Group by Method and compute mean/std across scenes
        if "Method" not in variant_df.columns:
            continue

        agg_dict = {}
        for col in metric_cols:
            agg_dict[col] = ["mean", "std", "count"]

        method_stats = variant_df.groupby("Method", observed=True)[metric_cols].agg(
            ["mean", "std", "count"]
        )

        # Flatten column names
        method_stats.columns = [f"{col}_{stat}" for col, stat in method_stats.columns]
        method_stats = method_stats.reset_index()

        # Add variant info
        method_stats.insert(0, "Variant", variant)
        method_stats.insert(1, "n_runs", run_count)
        method_stats.insert(2, "n_scenes", n_scenes)

        all_variant_summaries.append(method_stats)

        # Save individual variant summary
        variant_dir = os.path.join(output_dir, "variant_summaries")
        os.makedirs(variant_dir, exist_ok=True)

        # Parse variant for readable description
        parts = variant.split("__")
        desc_parts = []
        for part in parts:
            if part.startswith("kdtree_"):
                val = "True" if part == "kdtree_1" else "False"
                desc_parts.append(f"kdtree_filter={val}")
            elif part.startswith("init_"):
                desc_parts.append(f"num_initial_views={part.replace('init_', '')}")
            elif part.startswith("pts_"):
                val = "True" if part == "pts_1" else "False"
                desc_parts.append(f"load_3d_points={val}")
        variant_desc = ", ".join(desc_parts)

        if output_format in ["csv", "both"]:
            csv_path = os.path.join(variant_dir, f"{variant}.csv")
            method_stats.to_csv(csv_path, index=False, float_format="%.4f")
            print(f"Saved: {csv_path}")

        if output_format in ["markdown", "both"]:
            md_path = os.path.join(variant_dir, f"{variant}.md")
            _write_variant_markdown(
                method_stats, md_path, variant, variant_desc, run_count, n_scenes, metric_cols
            )
            print(f"Saved: {md_path}")

    # Save combined variant comparison
    if all_variant_summaries:
        combined = pd.concat(all_variant_summaries, ignore_index=True)

        if output_format in ["csv", "both"]:
            csv_path = os.path.join(output_dir, "variant_comparison.csv")
            combined.to_csv(csv_path, index=False, float_format="%.4f")
            print(f"Saved: {csv_path}")

        if output_format in ["markdown", "both"]:
            md_path = os.path.join(output_dir, "variant_comparison.md")
            _write_comparison_markdown(combined, md_path, metric_cols)
            print(f"Saved: {md_path}")


def _write_variant_markdown(
    df: pd.DataFrame, path: str, variant: str, variant_desc: str,
    run_count: int, n_scenes: int, metric_cols: List[str]
) -> None:
    """Write a markdown summary for a single variant."""
    import numpy as np

    lines = [
        f"# Variant Summary: {variant}",
        "",
        f"**Configuration:** {variant_desc}",
        f"**Total runs:** {run_count}",
        f"**Scenes:** {n_scenes}",
        "",
        "## Metrics Averaged Across All Scenes",
        "",
    ]

    # Build table for each metric type
    for metric_type in ["Final", "AUC", "MaxDelta"]:
        type_cols = [c for c in metric_cols if c.startswith(f"{metric_type}_")]
        if not type_cols:
            continue

        lines.append(f"### {metric_type} Values")
        lines.append("")

        # Extract metric names (PSNR, SSIM, LPIPS)
        metrics = sorted(set(c.replace(f"{metric_type}_", "").replace("_mean", "") for c in type_cols))

        # Header
        header = ["Method"] + [f"{m} (mean ± std)" for m in metrics]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        # Rows
        for _, row in df.iterrows():
            row_parts = [str(row["Method"])]
            for m in metrics:
                mean_col = f"{metric_type}_{m}_mean_mean"
                std_col = f"{metric_type}_{m}_mean_std"
                if mean_col in row and std_col in row:
                    mean_val = row[mean_col]
                    std_val = row[std_col]
                    if pd.notna(mean_val):
                        std_str = f" ± {std_val:.3f}" if pd.notna(std_val) else ""
                        row_parts.append(f"{mean_val:.4f}{std_str}")
                    else:
                        row_parts.append("-")
                else:
                    row_parts.append("-")
            lines.append("| " + " | ".join(row_parts) + " |")

        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_comparison_markdown(df: pd.DataFrame, path: str, metric_cols: List[str]) -> None:
    """Write a markdown comparison of all variants, organized by variant for easy method comparison."""
    import numpy as np

    lines = [
        "# Variant Comparison Summary",
        "",
        "Metrics averaged across all scenes. Each variant section shows all methods for easy comparison.",
        "",
    ]

    # Get unique variants and methods
    variants = sorted(df["Variant"].unique())
    methods = sorted(df["Method"].unique()) if "Method" in df.columns else []

    # Identify metrics
    metrics = sorted(set(
        c.replace("Final_", "").replace("AUC_", "").replace("MaxDelta_", "").replace("_mean", "")
        for c in metric_cols
    ))

    # For each variant, show all methods
    for variant in variants:
        variant_df = df[df["Variant"] == variant]
        if variant_df.empty:
            continue

        # Get run count and scene count
        run_count = variant_df["n_runs"].iloc[0] if "n_runs" in variant_df.columns else 0
        n_scenes = variant_df["n_scenes"].iloc[0] if "n_scenes" in variant_df.columns else 0

        # Parse variant for readable description
        parts = variant.split("__")
        desc_parts = []
        for part in parts:
            if part.startswith("kdtree_"):
                val = "True" if part == "kdtree_1" else "False"
                desc_parts.append(f"kdtree={val}")
            elif part.startswith("init_"):
                desc_parts.append(f"init_views={part.replace('init_', '')}")
            elif part.startswith("pts_"):
                val = "True" if part == "pts_1" else "False"
                desc_parts.append(f"3d_points={val}")
        variant_desc = ", ".join(desc_parts)

        lines.append(f"## {variant}")
        lines.append(f"**{variant_desc}** | Runs: {int(run_count)} | Scenes: {int(n_scenes)}")
        lines.append("")

        # Build a single table with all methods and all metric types
        # Columns: Method | Final PSNR | Final LPIPS | Final SSIM | AUC PSNR | AUC LPIPS | AUC SSIM | MaxDelta PSNR | ...
        header_parts = ["Method"]
        for metric_type in ["Final", "AUC", "MaxDelta"]:
            for m in metrics:
                header_parts.append(f"{metric_type} {m}")

        lines.append("| " + " | ".join(header_parts) + " |")
        lines.append("|" + "|".join(["---"] * len(header_parts)) + "|")

        for method in methods:
            method_row = variant_df[variant_df["Method"] == method]
            if method_row.empty:
                continue

            row = method_row.iloc[0]
            row_parts = [str(method)]

            for metric_type in ["Final", "AUC", "MaxDelta"]:
                for m in metrics:
                    mean_col = f"{metric_type}_{m}_mean_mean"
                    if mean_col in row and pd.notna(row[mean_col]):
                        row_parts.append(f"{row[mean_col]:.4f}")
                    else:
                        row_parts.append("-")

            lines.append("| " + " | ".join(row_parts) + " |")

        lines.append("")

    # Also add a compact cross-variant comparison table for quick overview
    lines.append("---")
    lines.append("")
    lines.append("# Quick Comparison Across Variants")
    lines.append("")
    lines.append("Shows key metrics (Final PSNR, AUC PSNR) for each method across all variants.")
    lines.append("")

    for method in methods:
        method_df = df[df["Method"] == method]
        if method_df.empty:
            continue

        lines.append(f"### {method}")
        lines.append("")

        header = ["Variant", "Runs", "Final PSNR", "Final LPIPS", "Final SSIM", "AUC PSNR", "AUC LPIPS", "AUC SSIM"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        for _, row in method_df.sort_values("Variant").iterrows():
            row_parts = [
                str(row["Variant"]),
                str(int(row["n_runs"])) if pd.notna(row["n_runs"]) else "-",
            ]
            for metric_type in ["Final", "AUC"]:
                for m in ["PSNR", "LPIPS", "SSIM"]:
                    mean_col = f"{metric_type}_{m}_mean_mean"
                    if mean_col in row and pd.notna(row[mean_col]):
                        row_parts.append(f"{row[mean_col]:.4f}")
                    else:
                        row_parts.append("-")
            lines.append("| " + " | ".join(row_parts) + " |")

        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_filtered_runs_report(
    excluded_df: pd.DataFrame, out_path: str, min_step: int
) -> None:
    """Save a report of filtered (excluded) runs."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Filtered Runs Report\n\n")
        f.write(f"Excluded runs with max step < {min_step}\n\n")
        f.write(f"Total excluded: {len(excluded_df)}\n\n")

        if not excluded_df.empty:
            f.write("| Run ID | Run Name | Scene | Max Step |\n")
            f.write("|--------|----------|-------|----------|\n")
            for _, row in excluded_df.iterrows():
                f.write(
                    f"| {row['run_id']} | {row.get('run_name', 'N/A')} | "
                    f"{row.get('scene', 'N/A')} | {int(row['max_step'])} |\n"
                )

    print(f"Saved: {out_path}")


def parse_bool(value: str) -> bool:
    """Parse boolean string."""
    return value.lower() in ("true", "1", "yes", "t")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze locally stored WandB curves with min-step filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - ingests all entities from ../results/
  python scripts/analyze_local_curves.py

  # Specific entity only
  python scripts/analyze_local_curves.py --entity navlab

  # Multiple specific entities
  python scripts/analyze_local_curves.py --entity navlab chengine-stanford-university

  # Explicit input file(s)
  python scripts/analyze_local_curves.py --input path/to/curves1.csv path/to/curves2.csv

  # Custom min step threshold
  python scripts/analyze_local_curves.py --min-step 25000

  # Filter by sweep params
  python scripts/analyze_local_curves.py --num-initial-views 1 10

  # Specific scenes only
  python scripts/analyze_local_curves.py --scenes caterpillar train

  # Group output by specific sweep parameters
  python scripts/analyze_local_curves.py --group-by kdtree_filter num_initial_views
        """,
    )

    # Input/output paths
    default_results_dir = "../results"
    default_output = "../results/combined_analysis"

    parser.add_argument(
        "--input",
        "-i",
        nargs="*",
        default=None,
        help="Path(s) to curves_with_params.csv file(s). If not specified, searches all entities in --results-dir",
    )
    parser.add_argument(
        "--results-dir",
        default=default_results_dir,
        help=f"Base results directory to search for entity data (default: {default_results_dir})",
    )
    parser.add_argument(
        "--entity",
        nargs="*",
        default=None,
        help="Specific entity/entities to include (default: all found in results-dir)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=default_output,
        help=f"Output directory for analysis results (default: {default_output})",
    )

    # Filtering options
    parser.add_argument(
        "--min-step",
        type=int,
        default=30000,
        help="Minimum max step threshold to include runs (default: 30000)",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Specific scenes to include (default: all)",
    )

    # Sweep parameter filters
    parser.add_argument(
        "--kdtree-filter",
        nargs="+",
        type=str,
        default=None,
        help="Filter by kdtree_filter values (true/false)",
    )
    parser.add_argument(
        "--num-initial-views",
        nargs="+",
        type=int,
        default=None,
        help="Filter by num_initial_views values",
    )
    parser.add_argument(
        "--load-3d-points",
        nargs="+",
        type=str,
        default=None,
        help="Filter by load_3d_points values (true/false)",
    )

    # Grouping/organization options
    parser.add_argument(
        "--group-by",
        nargs="+",
        choices=["kdtree_filter", "num_initial_views", "load_3d_points"],
        default=None,
        help="Group output by specific sweep parameters (generates separate outputs)",
    )
    parser.add_argument(
        "--split-by-variant",
        action="store_true",
        help="Split output into separate files per variant (default: combined file)",
    )

    # Output options
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=["csv", "markdown", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--auc-scale",
        type=float,
        default=1000.0,
        help="Scale factor for AUC values (default: 1000.0)",
    )

    args = parser.parse_args()

    # Resolve paths relative to current working directory (same as wandb_analysis_pipeline.py)
    results_dir = os.path.abspath(args.results_dir)
    output_dir = os.path.abspath(args.output_dir)

    # Find input files
    input_files = []
    if args.input:
        # Explicit input files provided
        input_files = [os.path.abspath(p) for p in args.input]
    else:
        # Search for curves_with_params.csv in all entity directories
        if os.path.exists(results_dir):
            for entity_dir in os.listdir(results_dir):
                entity_path = os.path.join(results_dir, entity_dir)
                if not os.path.isdir(entity_path):
                    continue
                # Filter by entity if specified
                if args.entity and entity_dir not in args.entity:
                    continue
                curves_file = os.path.join(entity_path, "wandb_exports", "curves_with_params.csv")
                if os.path.exists(curves_file):
                    input_files.append(curves_file)

    if not input_files:
        print(f"Error: No curves_with_params.csv files found in {results_dir}")
        print("Run wandb_analysis_pipeline.py first to download data.")
        return 1

    # Load and combine data from all input files
    print(f"Found {len(input_files)} data file(s):")
    all_dfs = []
    for input_path in input_files:
        # Extract entity name from path
        entity_name = os.path.basename(os.path.dirname(os.path.dirname(input_path)))
        print(f"  - {entity_name}: {input_path}")
        try:
            file_df = pd.read_csv(input_path)
            file_df["entity"] = entity_name  # Add entity column for tracking
            all_dfs.append(file_df)
            print(f"    Loaded {len(file_df)} rows, {file_df['run_id'].nunique()} runs")
        except Exception as e:
            print(f"    Warning: Failed to load {input_path}: {e}")

    if not all_dfs:
        print("Error: No data loaded from any files.")
        return 1

    df = pd.concat(all_dfs, ignore_index=True)
    total_runs = df["run_id"].nunique()
    total_entities = df["entity"].nunique() if "entity" in df.columns else 1
    print(f"\nCombined: {len(df)} rows from {total_runs} runs across {total_entities} entities")

    # Apply scene filter
    if args.scenes:
        df = apply_scene_filter(df, args.scenes)
        print(f"After scene filter ({', '.join(args.scenes)}): {df['run_id'].nunique()} runs")

    # Apply sweep parameter filters
    kdtree_filter = [parse_bool(v) for v in args.kdtree_filter] if args.kdtree_filter else None
    load_3d_points = [parse_bool(v) for v in args.load_3d_points] if args.load_3d_points else None

    df = apply_sweep_filters(
        df,
        kdtree_filter=kdtree_filter,
        num_initial_views=args.num_initial_views,
        load_3d_points=load_3d_points,
    )
    after_sweep_runs = df["run_id"].nunique()
    if kdtree_filter or args.num_initial_views or load_3d_points:
        print(f"After sweep filters: {after_sweep_runs} runs")

    # Filter by min step
    df, excluded_df = filter_by_min_step(df, args.min_step)
    filtered_runs = df["run_id"].nunique()
    excluded_count = len(excluded_df)

    print(f"Filtered out {excluded_count} runs with max step < {args.min_step}")
    print(f"Processing {filtered_runs} runs...")

    if df.empty:
        print("No runs remaining after filtering.")
        return 1

    # Print sweep parameter breakdown
    variant_run_counts = print_sweep_breakdown(df)

    # Save filtered runs report
    os.makedirs(output_dir, exist_ok=True)
    filtered_report_path = os.path.join(output_dir, "filtered_runs.txt")
    save_filtered_runs_report(excluded_df, filtered_report_path, args.min_step)

    # Create config for aggregation
    config = PipelineConfig(
        auc_scale=args.auc_scale,
        output_format=args.output_format,
    )

    # Aggregate metrics
    print("\nAggregating metrics...")
    aggregator = MetricsAggregator(config)
    summary_df = aggregator.aggregate(df)

    if summary_df.empty:
        print("No aggregated data.")
        return 1

    print(f"Aggregated {len(summary_df)} summary rows")

    # Generate tables
    print("\nGenerating tables...")
    generator = TableGenerator(config)

    # Determine grouping strategy
    if args.group_by:
        # Group by specified parameters
        group_cols = args.group_by
        print(f"Grouping output by: {', '.join(group_cols)}")

        # Get unique combinations from the curves data (not summary)
        group_combos = (
            df[["run_id"] + group_cols]
            .drop_duplicates()
            .drop(columns=["run_id"])
            .drop_duplicates()
        )

        for _, combo in group_combos.iterrows():
            # Build group key and filter
            group_key_parts = []
            mask = pd.Series([True] * len(summary_df), index=summary_df.index)

            for col in group_cols:
                val = combo[col]
                if col == "kdtree_filter":
                    group_key_parts.append(f"kdtree_{1 if val else 0}")
                elif col == "num_initial_views":
                    group_key_parts.append(f"init_{val}")
                elif col == "load_3d_points":
                    group_key_parts.append(f"pts_{1 if val else 0}")

                # Filter summary by Variant column which contains these params
                if "Variant" in summary_df.columns:
                    if col == "kdtree_filter":
                        pattern = f"kdtree_{1 if val else 0}"
                    elif col == "num_initial_views":
                        pattern = f"init_{val}"
                    elif col == "load_3d_points":
                        pattern = f"pts_{1 if val else 0}"
                    mask &= summary_df["Variant"].str.contains(pattern, na=False)

            group_key = "__".join(group_key_parts)
            group_df = summary_df[mask]

            if group_df.empty:
                continue

            # Count runs for this group
            group_run_count = sum(
                cnt for var, cnt in variant_run_counts.items()
                if all(part in var for part in group_key_parts)
            )

            print(f"  Generating output for {group_key} ({group_run_count} runs)...")

            group_dir = os.path.join(output_dir, "by_group", group_key)
            os.makedirs(group_dir, exist_ok=True)

            if args.output_format in ["csv", "both"]:
                generator.generate_csv(group_df, os.path.join(group_dir, "summary.csv"))

            if args.output_format in ["markdown", "both"]:
                generator.generate_markdown(
                    group_df,
                    os.path.join(group_dir, "summary.md"),
                    title=f"Summary - {group_key} ({group_run_count} runs)",
                )

    elif args.split_by_variant:
        # Split by full variant (original behavior)
        variants = summary_df["Variant"].unique() if "Variant" in summary_df.columns else ["default"]

        for variant in variants:
            variant_dir = os.path.join(output_dir, "by_variant", variant)
            os.makedirs(variant_dir, exist_ok=True)

            variant_df = (
                summary_df[summary_df["Variant"] == variant]
                if "Variant" in summary_df.columns
                else summary_df
            )

            run_count = variant_run_counts.get(variant, 0)
            print(f"  Generating output for {variant} ({run_count} runs)...")

            if args.output_format in ["csv", "both"]:
                generator.generate_csv(variant_df, os.path.join(variant_dir, "summary.csv"))

            if args.output_format in ["markdown", "both"]:
                generator.generate_markdown(
                    variant_df,
                    os.path.join(variant_dir, "summary.md"),
                    title=f"Summary - {variant} ({run_count} runs)",
                )

    # Always generate combined summary
    total_run_count = sum(variant_run_counts.values())
    print(f"  Generating combined summary ({total_run_count} total runs)...")

    if args.output_format in ["csv", "both"]:
        generator.generate_csv(summary_df, os.path.join(output_dir, "summary.csv"))

    if args.output_format in ["markdown", "both"]:
        generator.generate_markdown(
            summary_df,
            os.path.join(output_dir, "summary.md"),
            title=f"Summary - All Variants ({total_run_count} runs)",
        )

    # Generate variant-level summaries (averaged across scenes)
    print("\nGenerating variant-level summaries (averaged across scenes)...")
    generate_variant_summary(summary_df, variant_run_counts, output_dir, args.output_format)

    print("\nAnalysis complete!")
    print(f"Results saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
