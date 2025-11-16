#!/usr/bin/env python3
# Build an HTML table (same aesthetic as build_html.py) from a per-dataset summary CSV.
# Adds: scene & method renames; explicit method ordering.

import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

upto = 10000

# ===== Paths =====
### Tanks and Temples ###
IN_CSV   = f"../results/collated/summary_tanks_and_temples_upto_{upto}.csv"   # <-- point to your per-dataset CSV
OUT_HTML = f"../results/collated/html/tnt_table_upto_{upto}.html"

# ===== Display control =====
# Rename maps (applied before whitelist and averaging)
SCENE_RENAMES: Dict[str, str] = {
    "ignatius": "Ignatius",
    "train": "Train",
    "caterpillar": "Caterpillar",
}

# Show only these scenes (after renaming) in the stacked table; averages still use ALL scenes in the CSV
SCENE_WHITELIST: Optional[List[str]] = ["Ignatius", "Train"]   # e.g., ["caterpillar", "ignatius"]

## ------------------ ###

## CAPTURES ###
# IN_CSV   = f"../results/collated/summary_captures_upto_{upto}.csv"   # <-- point to your per-dataset CSV
# OUT_HTML = f"../results/collated/html/captures_table_upto_{upto}.html"

# # ===== Display control =====
# # Rename maps (applied before whitelist and averaging)
# SCENE_RENAMES: Dict[str, str] = {
#     "shiny statue": "Shiny",
#     "space laces": "Space",
#     "chair": "Chair",
# }

# # Show only these scenes (after renaming) in the stacked table; averages still use ALL scenes in the CSV
# SCENE_WHITELIST: Optional[List[str]] = ["Shiny", "Space"]   # e.g., ["caterpillar", "ignatius"]
## ------------------ ###


### MIPNERF 360 ###
# IN_CSV   = f"../results/collated/summary_mipnerf360_upto_{upto}.csv"   # <-- point to your per-dataset CSV
# OUT_HTML = f"../results/collated/html/mipnerf360_table_upto_{upto}.html"

# # ===== Display control =====
# # Rename maps (applied before whitelist and averaging)
# SCENE_RENAMES: Dict[str, str] = {
#     "bicycle": "Bicycle",
#     "bonsai": "Bonsai",
#     "counter": "Counter",
#     "flowers": "Flowers",
#     "garden": "Garden",
#     "kitchen": "Kitchen",
#     "room": "Room",
#     "stump": "Stump",
#     "treehill": "Treehill",
# }

# # Show only these scenes (after renaming) in the stacked table; averages still use ALL scenes in the CSV
# SCENE_WHITELIST: Optional[List[str]] = ["Garden", "Counter"]   # e.g., ["caterpillar", "ignatius"]
### ------------------ ###

METHOD_RENAMES: Dict[str, str] = {
    "coverage": "Coverage (Ours)",
    "random": "Random",
    "fisher_info": "FisherRF",
    "bayes-rays": "Bayes' Rays",
}

# Explicit method ordering (after rename). Leave [] to keep CSV order.
# If STRICT_METHOD_ORDER=True, any method not listed here is dropped.
METHOD_ORDER: List[str] = [
    "Bayes' Rays",
    "FisherRF",
    "Random",
    "Coverage (Ours)",
]
STRICT_METHOD_ORDER: bool = True   # True => drop methods not in METHOD_ORDER; False => append the rest

OUR_METHODS = ["Coverage (Ours)"]  # e.g., ["Coverage+"] or ["OPTICS"]

# Target 3 metrics (order of columns) and header arrows
METRICS = ["PSNR", "SSIM", "LPIPS"]
ARROWS  = {"PSNR": "↑", "SSIM": "↑", "LPIPS": "↓"}

# Higher/lower direction (for subcell ranking)
HIGHER_IS_BETTER = {"PSNR": True, "SSIM": True, "LPIPS": False}

# Column name patterns in the CSV (means only)
FINAL_COL = "Final {m}_mean"
AUC_COL   = "AUC {m}_mean"

AUC_MULTIPLIER = upto / 1000

# ========== Helpers ==========

def _two_dec(val) -> str:
    try:
        x = float(val)
        if np.isnan(x): return "—"
        return f"{x:.2f}"
    except Exception:
        return "—"

def _rank_series(s: pd.Series, higher: bool) -> pd.Series:
    return s.rank(ascending=not higher, method="min")

def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _apply_renames_and_method_order(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Renames ---
    if "Scene" in df.columns and SCENE_RENAMES:
        df["Scene"] = df["Scene"].map(SCENE_RENAMES).fillna(df["Scene"])

    if "Method" in df.columns and METHOD_RENAMES:
        df["Method"] = df["Method"].map(METHOD_RENAMES).fillna(df["Method"])

    # --- Method ordering ---
    if "Method" in df.columns and METHOD_ORDER:
        # If strict: drop methods not in the list
        if STRICT_METHOD_ORDER:
            df = df[df["Method"].isin(METHOD_ORDER)].copy()

        # Build the final order (append any unlisted methods if not strict)
        if STRICT_METHOD_ORDER:
            final_order = METHOD_ORDER
        else:
            existing = [m for m in df["Method"].unique().tolist() if m not in METHOD_ORDER]
            final_order = METHOD_ORDER + sorted(existing)

        df["Method"] = pd.Categorical(df["Method"], categories=final_order, ordered=True)
        # We'll sort rows per scene by this categorical later (in render)
    return df

def _compute_scene_rows(scene: str, sdf: pd.DataFrame, scene_idx: int) -> List[str]:
    """
    Build HTML rows for a single scene block: one row per Method (respecting METHOD_ORDER if set).
    """
    # Ensure needed numeric columns
    needed = []
    for m in METRICS:
        needed += [FINAL_COL.format(m=m), AUC_COL.format(m=m)]
    sdf = _ensure_numeric(sdf.copy(), needed)

    for m in METRICS:
        auc_col = AUC_COL.format(m=m)
        if auc_col in sdf.columns and AUC_MULTIPLIER != 1:
            sdf[auc_col] = sdf[auc_col] / AUC_MULTIPLIER

    # Order methods if categorical was set
    if pd.api.types.is_categorical_dtype(sdf["Method"]):
        sdf = sdf.sort_values("Method")
    else:
        # else keep CSV order
        pass

    # Build ranking per metric, for Final and AUC separately
    ranks_final, ranks_auc = {}, {}
    for m in METRICS:
        higher = HIGHER_IS_BETTER[m]
        fvals = sdf[FINAL_COL.format(m=m)]
        avals = sdf[AUC_COL.format(m=m)]
        ranks_final[m] = _rank_series(fvals, higher)
        ranks_auc[m]   = _rank_series(avals, higher)

    scene_class = "scene-even" if (scene_idx % 2 == 1) else "scene-odd"
    rows: List[str] = []
    row_count = len(sdf)
    if row_count == 0:
        return rows

    for ridx, row in sdf.reset_index(drop=True).iterrows():
        tds = []
        # Left rotated scene rail (rowspan)
        if ridx == 0:
            tds.append(
                f"<td class='scene-vert {scene_class}' rowspan='{row_count}'>"
                f"<span class='scene-vert-text'>{scene}</span></td>"
            )
        # Method
        method = str(row["Method"])
        is_ours = method in OUR_METHODS
        method_cls = "cell-text our-method" if is_ours else "cell-text"
        tds.append(f"<td class='method {scene_class}'><span class='{method_cls}'>{method}</span></td>")

        # Metrics (Final | AUC)
        for m in METRICS:
            v_final = row.get(FINAL_COL.format(m=m), np.nan)
            v_auc   = row.get(AUC_COL.format(m=m),   np.nan)

            is_best_final = (ranks_final[m].iloc[ridx] == 1)
            is_best_auc   = (ranks_auc[m].iloc[ridx]   == 1)

            td_extra = " has-chip" if (is_best_final or is_best_auc) else ""
            left_cls  = "sub final" + (" sub-best" if is_best_final else "")
            right_cls = "sub auc"   + (" sub-best" if is_best_auc   else "")

            tds.append(
                f"<td class='metric-cell {scene_class}{td_extra}'>"
                f"<div class='{left_cls}'><span class='cell-text'>{_two_dec(v_final)}</span></div>"
                f"<div class='{right_cls}'><span class='cell-text'>{_two_dec(v_auc)}</span></div>"
                "</td>"
            )

        rows.append(f"<tr class='{scene_class}'>" + "".join(tds) + "</tr>")
    return rows

def _compute_average_block(df_all: pd.DataFrame) -> List[str]:
    """
    Average across scenes per Method (after renames).
    """
    needed = []
    for m in METRICS:
        needed += [FINAL_COL.format(m=m), AUC_COL.format(m=m)]
    df = _ensure_numeric(df_all.copy(), needed)

    for m in METRICS:
        auc_col = AUC_COL.format(m=m)
        if auc_col in df.columns and AUC_MULTIPLIER != 1:
            df[auc_col] = df[auc_col] / AUC_MULTIPLIER

    # If method order categorical, keep it
    if pd.api.types.is_categorical_dtype(df["Method"]):
        ordered_methods = [m for m in df["Method"].cat.categories if m in df["Method"].unique()]
        df["Method"] = pd.Categorical(df["Method"], categories=ordered_methods, ordered=True)

    agg = (
        df.groupby("Method", dropna=False)[[FINAL_COL.format(m=m) for m in METRICS] +
                                           [AUC_COL.format(m=m) for m in METRICS]]
          .mean(numeric_only=True)
          .reset_index()
    )

    # Sort by categorical if present
    if pd.api.types.is_categorical_dtype(agg["Method"]):
        agg = agg.sort_values("Method")

    # Ranks across methods (averages)
    ranks_final_avg, ranks_auc_avg = {}, {}
    for m in METRICS:
        higher = HIGHER_IS_BETTER[m]
        ranks_final_avg[m] = _rank_series(agg[FINAL_COL.format(m=m)], higher)
        ranks_auc_avg[m]   = _rank_series(agg[AUC_COL.format(m=m)],   higher)

    avg_rows: List[str] = []
    avg_scene = "Average"
    avg_row_count = len(agg)

    for idx, row in agg.iterrows():
        tds = []
        if idx == 0:
            tds.append(
                f"<td class='scene-vert scene-avg' rowspan='{avg_row_count}'>"
                f"<span class='scene-vert-text'>{avg_scene}</span></td>"
            )
        # method
        is_ours = str(row["Method"]) in OUR_METHODS
        method_cls = "cell-text our-method" if is_ours else "cell-text"
        tds.append(f"<td class='method avg-cell'><span class='{method_cls}'>{row['Method']}</span></td>")

        # metrics
        for m in METRICS:
            vf = row[FINAL_COL.format(m=m)]
            va = row[AUC_COL.format(m=m)]
            is_best_f = (ranks_final_avg[m].iloc[idx] == 1)
            is_best_a = (ranks_auc_avg[m].iloc[idx]   == 1)
            td_extra  = " has-chip" if (is_best_f or is_best_a) else ""
            left_cls  = "sub final" + (" sub-best" if is_best_f else "")
            right_cls = "sub auc"   + (" sub-best" if is_best_a else "")
            tds.append(
                f"<td class='metric-cell avg-cell{td_extra}'>"
                f"<div class='{left_cls}'><span class='cell-text'>{_two_dec(vf)}</span></div>"
                f"<div class='{right_cls}'><span class='cell-text'>{_two_dec(va)}</span></div>"
                "</td>"
            )

        avg_rows.append("<tr class='avg-row'>" + "".join(tds) + "</tr>")
    return avg_rows

def _render_table(show_df: pd.DataFrame, all_df: pd.DataFrame) -> str:
    headers = ["", "Method"] + [f"{m} {ARROWS.get(m, '')}" for m in METRICS]
    n_cols  = len(headers)
    colgroup = '<colgroup><col class="scene-col" />' + '<col />' * (n_cols - 1) + '</colgroup>'

    # scene display order
    scenes = list(dict.fromkeys(show_df["Scene"].tolist()))
    body_rows: List[str] = []

    for sidx, sc in enumerate(scenes):
        sdf = show_df[show_df["Scene"] == sc].copy()
        body_rows += _compute_scene_rows(sc, sdf, sidx)

    # append averages block over ALL scenes in CSV
    body_rows += _compute_average_block(all_df)

    thead = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>"
    tbody = "<tbody>\n" + "\n".join(body_rows) + "\n</tbody>"

    return f"""
    <section id="print-target" class="card">
      <div class="table-wrap">
        <div class="table-surface">
          <table class="paper">
            {colgroup}
            {thead}
            {tbody}
          </table>
        </div>
      </div>
    </section>
    """

def main():
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"Could not find: {IN_CSV}")

    df = pd.read_csv(IN_CSV)

    # minimal schema check
    for col in ["Scene", "Method"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column '{col}'")

    # keep only needed columns
    keep_cols = ["Scene", "Method"]
    for m in METRICS:
        keep_cols += [FINAL_COL.format(m=m), AUC_COL.format(m=m)]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # --- apply renames and method order ---
    df = _apply_renames_and_method_order(df)

    # --- scene whitelist (after rename) ---
    if SCENE_WHITELIST:
        show_df = df[df["Scene"].isin(SCENE_WHITELIST)].copy()
        if show_df.empty:
            raise RuntimeError("SCENE_WHITELIST excluded all rows; nothing to display.")
    else:
        show_df = df.copy()

    html_section = _render_table(show_df, df)

    # ---------- CSS (same look) ----------
    css = """
    <style>
      :root {
        --border: #000;
        --bg-header: #000;
        --fg-header: #fff;
        --bg-alt: #fafafa;
        --card-bg: #fff;
        --shadow-card: 0 18px 40px rgba(0,0,0,.10), 0 6px 20px rgba(0,0,0,.06);

        --fs-header: 24px;
        --fs-method: 18px;
        --fs-scene:  12px;
        --fs-number: 15px;

        --chip-radius-best: 8px;

        --gray-bg: #4a4a4a;
        --gray-fg: #ffffff;
        --avg-green: #e8f6ea;
      }
      * { box-sizing: border-box; }
      body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
             margin: 28px; color: #111; background: #fff; }
      .card { border-radius: 18px; background: var(--card-bg); box-shadow: var(--shadow-card);
              margin: 24px 0 32px 0; border: 1px solid rgba(0,0,0,0.06); overflow: hidden; }
      .table-wrap { padding: 14px 14px 18px 14px; }
      .table-surface { position: relative; border-radius: 14px; overflow: hidden; }
      .table-surface::before {
        content: ""; position: absolute; inset: 0; border-radius: 14px;
        box-shadow: 0 0 0 1.5px var(--border) inset; z-index: 4; pointer-events: none;
      }

      table.paper { width: 100%; border-collapse: separate; border-spacing: 0; table-layout: fixed; background: #fff; }
      table.paper col.scene-col { width: 18px; }

      table.paper th, table.paper td {
        border: 1.5px solid var(--border); padding: 8px 10px; text-align: center; vertical-align: middle;
        background-clip: padding-box; position: relative; z-index: 1; font-size: 13.5px;
      }
      table.paper thead th { background: var(--bg-header); color: var(--fg-header);
                             font-weight: 700; font-size: var(--fs-header); }

      table.paper th:first-child, table.paper td.scene-vert {
        width: 18px !important; max-width: 18px !important; padding-left: 0; padding-right: 0; overflow: hidden;
      }

      table.paper tr > *:first-child { border-left: 0; }
      table.paper tr > *:last-child  { border-right: 0; }
      table.paper thead tr:first-child > * { border-top: 0; }
      table.paper tbody tr:last-child  > * { border-bottom: 0; }

      /* Rotated scene rail */
      .scene-vert { background: #fff; border-left: 0; position: relative; padding: 0; }
      .scene-vert-text {
        position: absolute; top: 50%; left: 50%;
        transform: translate(-50%, -50%) rotate(-90deg);
        transform-origin: center; white-space: nowrap; font-weight: 700; line-height: 1;
        font-size: var(--fs-scene);
      }

      /* Method font */
      td.method .cell-text { font-size: var(--fs-method); line-height: 1.15; white-space: nowrap; }

      /* Numbers */
      table.paper td .cell-text { font-size: var(--fs-number); line-height: 1.1; white-space: nowrap; }

      /* Metric cell split (Final | AUC) */
      td.metric-cell { position: relative; padding: 0; overflow: hidden; }
      td.metric-cell .sub {
        position: absolute; top: 0; bottom: 0; display: flex; align-items: center; justify-content: center; padding: 6px 6px;
        background: transparent;
      }
      td.metric-cell .sub.final { left: 0; right: 50%; }
      td.metric-cell .sub.auc   { left: 50%; right: 0; }

      /* Crisp diagonal divider */
      td.metric-cell::before {
        content: ""; position: absolute; left: 50%; top: 50%; width: 160%; height: 0;
        border-top: 1px solid var(--border);
        transform: translate(-50%, -50%) rotate(-45deg);
        z-index: 1; opacity: 0.9;
      }
      td.metric-cell .cell-text { position: relative; z-index: 2; }

      /* Chip (best subcell only) */
      td.metric-cell .sub.sub-best::after {
        content: ""; position: absolute; inset: 8px; border-radius: var(--chip-radius-best);
        background: #ffcccc; z-index: 1; pointer-events: none;
      }
      td.metric-cell .sub.sub-best .cell-text { z-index: 2; font-weight: 700; color: #000 !important; }

      /* Even-scene gray for the whole block (chips render on top) */
      tr.scene-even td { background: var(--gray-bg) !important; color: var(--gray-fg) !important; }

      /* Averages base */
      tr.avg-row td.avg-cell { background: var(--avg-green) !important; color: #000 !important; }

        /* Bold highlight for our method name */
        td.method .our-method {
        font-weight: 800 !important;
        }

      /* Print tweaks (tight PDF) */
      @media print {
        @page { size: 320mm 240mm; margin: 4mm; }
        html, body { margin: 0 !important; padding: 0 !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
        body > *:not(#print-target) { display: none !important; }  #print-target { display: block !important; }
        .card, .table-surface { box-shadow: none !important; } .table-surface::before { box-shadow: none !important; }
        .table-wrap { padding: 0 !important; } table.paper th, table.paper td { border-width: 1px !important; }
        .card, .table-wrap, .table-surface, table.paper, table.paper tr, table.paper td, table.paper th {
          page-break-inside: avoid; break-inside: avoid;
        }
        body { zoom: 0.94; }
      }
    </style>
    """

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'>")
        f.write(css)
        f.write("</head><body>\n")
        f.write(_render_table(show_df, df))
        f.write("\n</body></html>")

    print(f"HTML written to: {OUT_HTML}")

if __name__ == "__main__":
    main()
