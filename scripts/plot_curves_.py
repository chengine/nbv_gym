# plot_nbv_curves.py
# -----------------------------------------------------------
# Reads curves_long.csv exported from your W&B scraper and
# renders PSNR/LPIPS/SSIM vs gradient step for each scene.
# One figure per scene, with mean ± std across runs per method.
# -----------------------------------------------------------

import os
import math
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter

# ===================== CONFIG (edit me) =====================
CSV_PATH = "../results/wandb_exports/curves_long.csv"     # input CSV from the export script
OUTPUT_DIR = "../results/plots"                            # where to save figures
SCENES: Optional[List[str]] = [
    "caterpillar",
    "train",
    "ignatius",
    "shiny_statue_6pm",
    "space_laces_4pm",
    "chair_3pm",
]                  # e.g., ["shiny_statue_6pm", "chair_3pm"] or None = all
UNCERTAINTY_ALLOWLIST: List[str] = [
    "coverage",
    "fig",
    "view_fig",
    "fig_diag",
    "view_fig_diag",
    "fisher_info",
    "bayes-rays",
    "random",
]
INCLUDE_BASELINES_WITH_NO_SELECTOR = True           # keep methods where selector is NaN/None
METRICS_TO_PLOT = ["/psnr", "/lpips", "/ssim"]      # which metrics to draw (columns in 'metric')

# Optional smoothing (visual only): moving-average window (in steps)
SMOOTH_WINDOW_STEPS: Optional[int] = None           # e.g., 200; None disables smoothing

# Style / aesthetics
FIGSIZE = (14, 4.2)      # width, height per figure (one figure per scene)
DPI = 180
LINEWIDTH = 2.2
ALPHA_FILL = 0.20        # alpha for std band
MARK_EVERY = 10          # mark every N points (None to disable markers)
GRID_ALPHA = 0.25
TITLE_FONTSIZE = 15
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 11
LEGEND_FONTSIZE = 10
# ===========================================================

def _k_trunc_formatter(x, pos):
    # Truncate (not round): 6499 -> 6
    try:
        return f"{int(x // 1000)}"
    except Exception:
        return ""

def _nice_method_label(uncertainty: str, selector: Optional[str]) -> str:
    if selector and str(selector).strip().lower() not in {"nan", "none", ""}:
        return f"{uncertainty} • {selector}"
    return uncertainty


def _rolling_by_step(df: pd.DataFrame, window_steps: int) -> pd.DataFrame:
    """Apply a centered moving average over 'value' grouped by method+metric using step as x."""
    if df.empty or not window_steps:
        return df
    # we assume step spacing is uniform; compute window size in indices per group
    out = []
    for _, g in df.groupby(["scene", "uncertainty", "selector", "metric"], dropna=False):
        g = g.sort_values("step")
        if len(g) < 3:
            out.append(g)
            continue
        # rolling window in terms of indices (approx steps window by nearest size)
        win = max(1, min(len(g), int(round(window_steps / max(1, np.diff(g["step"].values).mean())))))
        g["value"] = g["value"].rolling(win, center=True, min_periods=max(1, win // 2)).mean()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _aggregate_curves(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average across runs per (scene, uncertainty, selector, metric, step),
    compute mean and std.
    """
    if df.empty:
        return df
    agg = (df.groupby(["scene", "uncertainty", "selector", "metric", "step"], dropna=False)["value"]
             .agg(["mean", "std", "count"])
             .reset_index())
    agg = agg.rename(columns={"mean": "y", "std": "y_std", "count": "n"})
    return agg


def _apply_allowlist(df: pd.DataFrame) -> pd.DataFrame:
    if UNCERTAINTY_ALLOWLIST:
        df = df[df["uncertainty"].isin(UNCERTAINTY_ALLOWLIST)]
    if not INCLUDE_BASELINES_WITH_NO_SELECTOR:
        df = df[~df["selector"].isna()]
    return df


def _prep_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # required columns
    needed = {"scene", "uncertainty", "selector", "metric", "step", "value"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Filter scenes
    if SCENES:
        df = df[df["scene"].isin(SCENES)]

    # Filter to target metrics
    df = df[df["metric"].isin(METRICS_TO_PLOT)]

    # Filter by uncertainty allowlist + baselines
    df = _apply_allowlist(df)

    # Clean types
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["step", "value"]).copy()
    df["selector"] = df["selector"].astype(object)  # allow None/NaN

    return df


def _set_matplotlib_style():
    # Clean, modern defaults
    mpl.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": TICK_FONTSIZE,
        "axes.titlesize": TITLE_FONTSIZE,
        "axes.labelsize": LABEL_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "axes.grid": True,
        "grid.alpha": GRID_ALPHA,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "lines.linewidth": LINEWIDTH,
        "axes.prop_cycle": mpl.cycler("color", plt.cm.tab10.colors + plt.cm.Set2.colors),
    })


def _draw_scene_figure(scene: str, agg: pd.DataFrame, out_dir: str):
    # make room on the right for the legend
    fig, axs = plt.subplots(1, len(METRICS_TO_PLOT), figsize=FIGSIZE, constrained_layout=False)
    if len(METRICS_TO_PLOT) == 1:
        axs = [axs]
    # reserve space for legend at right
    plt.subplots_adjust(right=0.80)  # ~20% of width for legend

    # collect legend entries (uncertainty only, dedup later)
    legend_handles = {}
    for ax, metric in zip(axs, METRICS_TO_PLOT):
        sub = agg[(agg["scene"] == scene) & (agg["metric"] == metric)]
        if sub.empty:
            # no axis title, per request
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, alpha=0.5)
            ax.set_xlabel("Step (K)")
            ax.xaxis.set_major_formatter(FuncFormatter(_k_trunc_formatter))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            # label with arrows in y-axis only
            if metric == "/psnr":
                ax.set_ylabel("PSNR (dB) ↑")
            elif metric == "/lpips":
                ax.set_ylabel("LPIPS ↓")
            elif metric == "/ssim":
                ax.set_ylabel("SSIM ↑")
            continue

        # plot each method; label ONLY by uncertainty (drop selector in legend)
        for (unc, sel), g in sub.groupby(["uncertainty", "selector"], dropna=False):
            g = g.sort_values("step")
            # draw line
            (line,) = ax.plot(
                g["step"], g["y"],
                marker="o" if MARK_EVERY else None,
                markevery=MARK_EVERY if MARK_EVERY else None,
                linewidth=LINEWIDTH,
            )
            # std band if available
            if not g["y_std"].isna().all() and (g["n"] > 1).any():
                ax.fill_between(g["step"], g["y"] - g["y_std"], g["y"] + g["y_std"], alpha=ALPHA_FILL)

            # capture a handle for this uncertainty label (first occurrence wins)
            label = str(unc)
            if label not in legend_handles:
                legend_handles[label] = line

        # axis cosmetics (no per-axis title)
        ax.set_xlabel("Step (K)")
        ax.xaxis.set_major_formatter(FuncFormatter(_k_trunc_formatter))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        if metric == "/psnr":
            ax.set_ylabel("PSNR (dB) ↑")   # add the arrow
        elif metric == "/lpips":
            ax.set_ylabel("LPIPS ↓")
        elif metric == "/ssim":
            ax.set_ylabel("SSIM ↑")

        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)

        if metric == "/psnr" and not sub["y"].empty:
            lo = float(np.floor(sub["y"].min() / 5) * 5)
            hi = float(np.ceil(sub["y"].max() / 5) * 5)
            if hi > lo:
                ax.set_ylim(lo, hi)

    # single legend to the RIGHT, outside the axes; no titles anywhere
    if legend_handles:
        fig.legend(
            legend_handles.values(), legend_handles.keys(),
            loc="center left", bbox_to_anchor=(0.82, 0.5),  # right side
            frameon=False, ncols=1,
        )

    # save
    os.makedirs(out_dir, exist_ok=True)
    safe = scene.replace("/", "_")
    fig.savefig(os.path.join(out_dir, f"{safe}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{safe}.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _set_matplotlib_style()

    df = _prep_data(CSV_PATH)

    # Aggregate mean/std across runs at same step
    agg = _aggregate_curves(df)

    # Optional smoothing (applied to the aggregated mean per method)
    if SMOOTH_WINDOW_STEPS:
        agg = agg.groupby(["scene", "uncertainty", "selector", "metric"], dropna=False, as_index=False)\
                 .apply(lambda g: _rolling_by_step(g, SMOOTH_WINDOW_STEPS))\
                 .reset_index(drop=True)

    scenes = sorted(agg["scene"].unique())
    if not scenes:
        print("No data to plot (after filters).")
        return

    out_dir = os.path.join(OUTPUT_DIR, "per_scene")
    for scene in scenes:
        _draw_scene_figure(scene, agg, out_dir)

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
