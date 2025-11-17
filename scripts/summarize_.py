# nbv_summaries_all_in_one_grouped.py
# -----------------------------------------------------------
# One CSV with scene-separated blocks + one HTML with one
# paper-ready table per scene (black grid, sleek styling).
# -----------------------------------------------------------

import os
import numpy as np
import pandas as pd
from io import StringIO

# ====================== CONFIG (edit me) ======================
# ENTITY = "chengine-stanford-university"
ENTITY = "navlab"
CSV_PATH = f"../results/{ENTITY}/wandb_exports/curves_long.csv"
OUT_DIR = f"../results/{ENTITY}/tables"

# Only these uncertainty methods appear in the output tables
METHODS_ALLOWLIST = ["coverage", "fisher_info", "bayes-rays", "random"]  # e.g. ["coverage","fig","view_fig"]

# Baseline used for AUC(method - baseline)
BASELINE = "random"
INCLUDE_BASELINE_FOR_AUC = True  # use baseline internally even if not in allowlist (baseline not shown)

# Metrics present in curves_long.csv
METRICS = ["/psnr", "/lpips", "/ssim"]
HIGHER_IS_BETTER = {"/psnr": True, "/lpips": False, "/ssim": True}

# Optional: restrict to scenes; None => all scenes
SCENES = [
    "caterpillar",
    "train",
    "ignatius",
    "shiny_statue_6pm",
    "space_laces_4pm",
    "chair_3pm",
]  # e.g., ["shiny_statue_6pm", "chair_3pm"]
# =============================================================

PRETTY = {"/psnr": "PSNR (↑)", "/lpips": "LPIPS (↓)", "/ssim": "SSIM (↑)"}
HILITE = {1: "#ffcccc", 2: "#ffdd99", 3: "#fff2b3"}  # red, orange, yellow (soft)

def _ensure_columns(df: pd.DataFrame, need):
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"CSV missing columns: {miss}")

def _final_value(curve: pd.DataFrame):
    if curve.empty:
        return None
    idx = curve["step"].idxmax()
    return float(curve.loc[idx, "y"])

def _auc_diff_vs_random(method_curve: pd.DataFrame, random_curve: pd.DataFrame):
    if method_curve.empty or random_curve.empty:
        return None
    x_m = method_curve["step"].to_numpy(dtype=float)
    y_m = method_curve["y"].to_numpy(dtype=float)
    x_r = random_curve["step"].to_numpy(dtype=float)
    y_r = random_curve["y"].to_numpy(dtype=float)
    if len(x_m) < 2 or len(x_r) < 2:
        return None
    y_r_interp = np.interp(x_m, x_r, y_r, left=y_r[0], right=y_r[-1])
    return float(np.trapz(y_m - y_r_interp, x_m))

def _rank(values: pd.Series, higher: bool) -> pd.Series:
    # 1 = best; NaNs sink
    return values.rank(ascending=not higher, method="min")

def _truncate2(x):
    """Truncate toward zero to 2 decimals (no rounding)."""
    if pd.isna(x):
        return x
    return np.trunc(float(x) * 100.0) / 100.0

def _scene_table(scene: str, df: pd.DataFrame, allowset: set | None):
    """Return (table_with_ranks, html_table_without_ranks) for a single scene."""
    sdf = df[df["scene"] == scene].copy()
    # Mean curve per (uncertainty, metric, step) aggregated over runs/selectors
    mean_curves = (
        sdf.groupby(["scene", "uncertainty", "metric", "step"], dropna=False)["value"]
           .mean()
           .reset_index()
           .rename(columns={"value": "y"})
    )

    # Allowed methods (+ baseline if needed for AUC calculation)
    if allowset is not None:
        allowed_for_curves = set(allowset)
        if INCLUDE_BASELINE_FOR_AUC:
            allowed_for_curves.add(BASELINE)
        mean_curves = mean_curves[mean_curves["uncertainty"].isin(allowed_for_curves)]

    # Methods to display (rows)
    if allowset is not None:
        methods = [m for m in sorted(mean_curves["uncertainty"].dropna().astype(str).unique())
                   if m in allowset]
    else:
        methods = sorted(mean_curves["uncertainty"].dropna().astype(str).unique())
    if not methods:
        return None, None

    # Curves dict
    curves_by_metric = {}
    for metric in METRICS:
        sub = mean_curves[mean_curves["metric"] == metric]
        curves_by_metric[metric] = {
            m: sub[sub["uncertainty"] == m][["step", "y"]]
                    .sort_values("step").reset_index(drop=True)
            for m in sub["uncertainty"].dropna().astype(str).unique()
        }

    # Assemble numeric table for this scene
    rows = []
    for m in methods:
        row = {"Scene": scene, "Method": m}
        for metric in METRICS:
            mc = curves_by_metric.get(metric, {}).get(m, pd.DataFrame(columns=["step", "y"]))
            row[f"Final {metric}"] = _final_value(mc)
            if m == BASELINE:
                row[f"AUC {metric}"] = 0.0
            else:
                rc = curves_by_metric.get(metric, {}).get(BASELINE, pd.DataFrame(columns=["step", "y"]))
                row[f"AUC {metric}"] = _auc_diff_vs_random(mc, rc)
        rows.append(row)

    table = pd.DataFrame(rows).set_index(["Scene", "Method"])

    # Truncate (no rounding)
    for metric in METRICS:
        for prefix in ("Final", "AUC"):
            col = f"{prefix} {metric}"
            table[col] = table[col].apply(_truncate2)

    # Ranks per scene (aligned to MultiIndex)
    for metric in METRICS:
        fcol = f"Final {metric}"
        acol = f"AUC {metric}"
        table[f"Rank Final {metric}"] = table[fcol].groupby(level="Scene").rank(
            ascending=not HIGHER_IS_BETTER[metric], method="min"
        )
        table[f"Rank AUC {metric}"] = table[acol].groupby(level="Scene").rank(
            ascending=not HIGHER_IS_BETTER[metric], method="min"
        )

    # Pretty headers
    rename = {}
    for metric in METRICS:
        pretty = PRETTY[metric]
        rename[f"Final {metric}"] = f"Final {pretty}"
        rename[f"AUC {metric}"]   = f"AUC {pretty}"
        rename[f"Rank Final {metric}"] = f"Rank Final {pretty}"
        rename[f"Rank AUC {metric}"]   = f"Rank AUC {pretty}"
    table = table.rename(columns=rename)

    # CSV view (with ranks)
    csv_cols = []
    for metric in METRICS:
        pretty = PRETTY[metric]
        csv_cols += [f"Final {pretty}", f"Rank Final {pretty}"]
    for metric in METRICS:
        pretty = PRETTY[metric]
        csv_cols += [f"AUC {pretty}", f"Rank AUC {pretty}"]
    csv_table = table[csv_cols]

    # HTML view (drop rank columns)
    html_cols = []
    for metric in METRICS:
        pretty = PRETTY[metric]
        html_cols += [f"Final {pretty}"]
    for metric in METRICS:
        pretty = PRETTY[metric]
        html_cols += [f"AUC {pretty}"]
    html_table = table[html_cols]
    return csv_table, html_table

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    _ensure_columns(df, ["scene", "uncertainty", "metric", "step", "value"])

    # Filter scenes/metrics and clean types
    if SCENES:
        df = df[df["scene"].isin(SCENES)]
    df = df[df["metric"].isin(METRICS)].copy()
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["step", "value"]).copy()

    scenes = sorted(df["scene"].unique())
    if not scenes:
        print("No scenes to process.")
        return

    allowset = set(METHODS_ALLOWLIST) if METHODS_ALLOWLIST else None

    # ---------- Build per-scene tables ----------
    per_scene_csv = []
    per_scene_html = []

    for scene in scenes:
        csv_table, html_table = _scene_table(scene, df, allowset)
        if csv_table is None:
            continue
        per_scene_csv.append((scene, csv_table.reset_index()))
        per_scene_html.append((scene, html_table.reset_index()))

    if not per_scene_csv:
        print("No data after filtering.")
        return

    # ---------- ONE CSV with scene-separated blocks ----------
    csv_path = os.path.join(OUT_DIR, "all_scenes_grouped.csv")
    with open(csv_path, "w") as f:
        for i, (scene, block) in enumerate(per_scene_csv):
            # Scene header row
            f.write(f"Scene,{scene}\n")
            # Column headers + data
            block.to_csv(f, index=False, float_format="%.2f")
            # Blank separator line (except after last)
            if i < len(per_scene_csv) - 1:
                f.write("\n")
    print(f"CSV saved: {csv_path}")

    # ---------- ONE HTML with one table per scene (sleek, paper-ready) ----------
    # Global CSS (black grid, tight typography, subtle hover)
    css = """
    <style>
      :root {
        --border: #000;
        --bg-header: #111;
        --fg-header: #fff;
        --bg-alt: #fafafa;
      }
      body {
        font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, 'Apple Color Emoji', 'Segoe UI Emoji';
        margin: 24px;
        color: #111;
        background: #fff;
      }
      h2.scene-title {
        margin: 28px 0 10px 0;
        font-size: 20px;
        letter-spacing: 0.2px;
        font-weight: 700;
      }
      table.paper {
        border-collapse: collapse;
        width: 100%;
        margin: 8px 0 28px 0;
        table-layout: fixed;
      }
      table.paper th, table.paper td {
        border: 1.5px solid var(--border);
        padding: 8px 10px;
        text-align: center;
        vertical-align: middle;
        font-size: 13.5px;
      }
      table.paper th {
        background: var(--bg-header);
        color: var(--fg-header);
        font-weight: 700;
      }
      table.paper tr:nth-child(even) td {
        background: var(--bg-alt);
      }
      table.paper tr:hover td {
        filter: brightness(0.98);
      }
      .cell-best  { background: #ffcccc !important; }  /* red */
      .cell-2nd   { background: #ffdd99 !important; }  /* orange */
      .cell-3rd   { background: #fff2b3 !important; }  /* yellow */
    </style>
    """

    # Build HTML per scene with highlights
    sections = []
    for scene, html_block in per_scene_html:
        # Compute ranks for this block (Final + AUC per metric)
        for metric in METRICS:
            pretty = PRETTY[metric]
            # finals
            html_block[f"_rank_final_{metric}"] = _rank(
                html_block[f"Final {pretty}"], HIGHER_IS_BETTER[metric]
            )
            # aucs
            html_block[f"_rank_auc_{metric}"] = _rank(
                html_block[f"AUC {pretty}"], HIGHER_IS_BETTER[metric]
            )

        # Render using pandas Styler to get HTML table, then inject classes for highlights
        # We’ll use a very simple Styler only to get the HTML scaffolding; borders are via CSS above
        styler = html_block.style.hide(axis="index")
        # 2-decimal display (values already truncated)
        styler = styler.format(precision=2)

        # Convert to HTML and then add highlight classes
        table_html = styler.to_html()
        # Inject table class "paper"
        table_html = table_html.replace('<table class="dataframe">', '<table class="paper">')

        # Now add highlight classes by string replacement per cell value position.
        # Simpler: rebuild the HTML using the DataFrame to mark td with classes.
        # We’ll reconstruct manually for reliability.

        headers = ['Method'] + [c for c in html_block.columns if not c.startswith("_rank_") and c != 'Method']
        # Ensure first column is Method (it is after reset_index, but guard)
        if 'Method' not in html_block.columns:
            # if not present (shouldn't happen), skip
            pass

        # Build rows with highlight classes
        buf = StringIO()
        buf.write('<table class="paper">\n<thead>\n<tr>')
        for h in headers:
            buf.write(f"<th>{h}</th>")
        buf.write("</tr>\n</thead>\n<tbody>\n")

        for i, row in html_block.iterrows():
            buf.write("<tr>")
            # Method cell
            buf.write(f"<td>{row.get('Method','')}</td>")
            # Metric cells with highlights
            for metric in METRICS:
                pretty = PRETTY[metric]
                # Final
                v_f = row[f"Final {pretty}"]
                r_f = row[f"_rank_final_{metric}"]
                cls_f = "cell-best" if r_f == 1 else "cell-2nd" if r_f == 2 else "cell-3rd" if r_f == 3 else ""
                buf.write(f'<td class="{cls_f}">{v_f:.2f}</td>')
            for metric in METRICS:
                pretty = PRETTY[metric]
                # AUC
                v_a = row[f"AUC {pretty}"]
                r_a = row[f"_rank_auc_{metric}"]
                cls_a = "cell-best" if r_a == 1 else "cell-2nd" if r_a == 2 else "cell-3rd" if r_a == 3 else ""
                buf.write(f'<td class="{cls_a}">{v_a:.2f}</td>')
            buf.write("</tr>\n")
        buf.write("</tbody>\n</table>\n")
        final_table_html = buf.getvalue()

        sections.append(f'<h2 class="scene-title">Scene: {scene}</h2>\n{final_table_html}')

    # Assemble full HTML
    html_path = os.path.join(OUT_DIR, "all_scenes_grouped.html")
    with open(html_path, "w") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'>")
        f.write(css)
        f.write("</head><body>\n")
        for sec in sections:
            f.write(sec)
        f.write("\n</body></html>")
    print(f"HTML saved: {html_path}")

    print(f"\nDone. Outputs in: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
