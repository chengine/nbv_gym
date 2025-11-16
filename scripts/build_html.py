# build_html.py
# -----------------------------------------------------------
# Single stacked table for selected scenes + "average" block appended.
#  - Even scene blocks: ALL non-chip cells turn gray with white text.
#  - Columns compressed to PSNR / SSIM / LPIPS.
#  - Each metric cell is split into two subcells: Final (left) and AUC (right),
#    separated by a diagonal. Best subcell only gets a red chip (smaller than subcell).
#  - Averages are computed over ALL scenes (even those not shown) and appended as
#    a final block with scene label "average" and pastel green base.
# -----------------------------------------------------------

import os
from io import StringIO
from typing import List, Tuple, Optional, Dict
import unicodedata
import numpy as np
import pandas as pd

# ===== Paths =====
ENTITY = "navlab"
IN_CSV = f"../results/{ENTITY}/tables/all_scenes_grouped.csv"
OUT_HTML = f"../results/{ENTITY}/tables/output.html"

# ===== Behavior =====
FINAL_PREFIX = "Final "
AUC_PREFIX   = "AUC "

# Which scenes to SHOW (averages still computed over ALL)
SCENE_WHITELIST = [
    "caterpillar",
    "ignatius",
]

# Directionality
HIGHER_BETTER_SUBSTRINGS = ["PSNR", "SSIM"]
LOWER_BETTER_SUBSTRINGS  = ["LPIPS"]

# Target 3 metrics (order of columns)
METRICS = ["PSNR", "SSIM", "LPIPS"]


# ========== CSV helpers ==========
def _split_grouped_csv_lines(path: str) -> List[Tuple[str, List[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines()]
    blocks: List[Tuple[str, List[str]]] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1; continue
        if not line.startswith("Scene,"):
            i += 1; continue
        scene = line.split(",", 1)[1] if "," in line else "UNKNOWN_SCENE"
        i += 1
        blk = []
        while i < n and lines[i].strip():
            blk.append(lines[i]); i += 1
        blocks.append((scene, blk))
    return blocks

def _norm_header(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).replace("\xa0", " ").strip()
    if s.lower().startswith("unnamed:"): s = s.split(":", 1)[-1]
    return " ".join(s.split()).lower()

def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    keys = [_norm_header(c) for c in cols]
    pretty: Dict[str, str] = {}
    for c, k in zip(cols, keys): pretty.setdefault(k, str(c))
    out = pd.DataFrame(index=df.index)
    from collections import defaultdict
    groups = defaultdict(list)
    for c, k in zip(cols, keys): groups[k].append(c)
    for k, cols_k in groups.items():
        out[pretty[k]] = df[cols_k].bfill(axis=1).iloc[:, 0] if len(cols_k) > 1 else df[cols_k[0]]
    return out

def _ensure_single_method_column(df: pd.DataFrame) -> pd.DataFrame:
    ms = [c for c in df.columns if _norm_header(c) == "method"]
    if not ms: raise ValueError("Block missing 'Method' column.")
    if len(ms) > 1:
        df["Method"] = df[ms].bfill(axis=1).iloc[:, 0]
        df = df.drop(columns=[c for c in ms if c != "Method"])
    elif ms[0] != "Method":
        df = df.rename(columns={ms[0]: "Method"})
    return df

def _to_scalar_method(val):
    if isinstance(val, pd.Series):
        val = val.dropna()
        val = val.iloc[0] if len(val) else None
    if val is None: return "—"
    try:
        if isinstance(val, float) and np.isnan(val): return "—"
    except Exception:
        pass
    return str(val)

def _two_dec_str(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    try: return f"{float(x):.2f}"
    except Exception: return "—"

def _detect_metric_direction(metric_name: str) -> bool:
    u = metric_name.upper()
    if any(k in u for k in HIGHER_BETTER_SUBSTRINGS): return True
    if any(k in u for k in LOWER_BETTER_SUBSTRINGS): return False
    return True

def _rank_series(s: pd.Series, higher: bool) -> pd.Series:
    return s.rank(ascending=not higher, method="min")


# Map a dataframe’s columns into dicts {metric: series} for Final/AUC
def _extract_metric_sets(df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
    finals_cols = [c for c in df.columns if str(c).startswith(FINAL_PREFIX)]
    aucs_cols   = [c for c in df.columns if str(c).startswith(AUC_PREFIX)]

    finals: Dict[str, pd.Series] = {}
    aucs:   Dict[str, pd.Series] = {}

    def pick(cols: List[str], metric: str) -> Optional[str]:
        # pick the first column whose name contains the metric token (case-insensitive)
        for c in cols:
            if metric.lower() in c.lower():
                return c
        return None

    for m in METRICS:
        fc = pick(finals_cols, m)
        ac = pick(aucs_cols, m)
        finals[m] = df[fc] if fc in df.columns else pd.Series([np.nan] * len(df), index=df.index)
        aucs[m]   = df[ac] if ac in df.columns else pd.Series([np.nan] * len(df), index=df.index)

    return finals, aucs


def _collect_blocks() -> List[Tuple[str, pd.DataFrame]]:
    """Parse all scene blocks; divide AUC by 1000; return (scene, cleaned_df)."""
    blocks = _split_grouped_csv_lines(IN_CSV)
    cleaned: List[Tuple[str, pd.DataFrame]] = []
    for scene, blk in blocks:
        df = pd.read_csv(StringIO("\n".join(blk)))
        df = _coalesce_duplicate_columns(df)
        df = _ensure_single_method_column(df)
        if "Scene" not in df.columns:
            df.insert(0, "Scene", scene)
        df = df[df["Scene"].astype(str) == scene]

        keep = ["Scene", "Method"] + [c for c in df.columns if not str(c).startswith("Rank ")]
        df = df[keep].copy()
        num_cols = [c for c in df.columns if c not in ("Scene", "Method")]
        for c in num_cols: df[c] = pd.to_numeric(df[c], errors="coerce")

        # scale AUCs
        for c in [c for c in df.columns if str(c).startswith(AUC_PREFIX)]:
            df[c] = df[c] / 1000.0

        cleaned.append((scene, df))
    return cleaned


# ========== Render ==========
def _render_all_with_avg(show_blocks: List[Tuple[str, pd.DataFrame]],
                         all_blocks: List[Tuple[str, pd.DataFrame]]) -> str:
    """One big table: selected scenes stacked + averages block appended."""

    # Use the first available block (shown or all) to infer existence; headers are fixed (3 metrics)
    template_df = (show_blocks[0][1] if show_blocks else all_blocks[0][1]).copy()
    ARROWS = {"PSNR": "↑", "SSIM": "↑", "LPIPS": "↓"}
    headers = ["", "Method"] + [f"{m} {ARROWS.get(m,'')}" for m in METRICS]

    # colgroup with fixed first column width
    n_cols = len(headers)
    colgroup = '<colgroup><col class="scene-col" />' + '<col />' * (n_cols - 1) + '</colgroup>'

    body_rows: List[str] = []

    # ---------- per-scene rows ----------
    for scene_idx, (scene, df) in enumerate(show_blocks):
        # ensure numeric and required columns exist
        finals_map, aucs_map = _extract_metric_sets(df)

        # per-metric subcell ranks (separately for Final and AUC), higher/lower depends on metric
        ranks_final: Dict[str, pd.Series] = {}
        ranks_auc:   Dict[str, pd.Series] = {}
        for m in METRICS:
            higher = _detect_metric_direction(m)
            ranks_final[m] = _rank_series(finals_map[m], higher)
            ranks_auc[m]   = _rank_series(aucs_map[m],   higher)

        row_count = len(df)
        if row_count == 0: continue
        first_index = df.index[0]
        scene_class = "scene-even" if (scene_idx % 2 == 1) else "scene-odd"

        for ridx, row in df.iterrows():
            tds = []
            # left rotated scene label (rowspan)
            if ridx == first_index:
                tds.append(
                    f"<td class='scene-vert {scene_class}' rowspan='{row_count}'>"
                    f"<span class='scene-vert-text'>{scene}</span></td>"
                )
            # method
            tds.append(f"<td class='method {scene_class}'><span class='cell-text'>{_to_scalar_method(row['Method'])}</span></td>")

            # 3 metric cells, each with two subcells
            for m in METRICS:
                v_final = row[finals_map[m].name] if finals_map[m].name in df.columns else np.nan
                v_auc   = row[aucs_map[m].name]   if aucs_map[m].name   in df.columns else np.nan

                # which subcell(s) get chips?
                is_best_final = (ranks_final[m].get(ridx, np.nan) == 1)
                is_best_auc   = (ranks_auc[m].get(ridx,   np.nan) == 1)

                # td gets .has-chip if ANY subcell has chip (so gray parity skips it)
                td_extra = " has-chip" if (is_best_final or is_best_auc) else ""

                # subcell classes
                left_cls  = "sub final" + (" sub-best" if is_best_final else "")
                right_cls = "sub auc"   + (" sub-best" if is_best_auc   else "")

                tds.append(
                    f"<td class='metric-cell {scene_class}{td_extra}'>"
                    f"<div class='{left_cls}'><span class='cell-text'>{_two_dec_str(v_final)}</span></div>"
                    f"<div class='{right_cls}'><span class='cell-text'>{_two_dec_str(v_auc)}</span></div>"
                    "</td>"
                )

            body_rows.append(f"<tr class='{scene_class}'>" + "".join(tds) + "</tr>")

    # ---------- averages block appended ----------
    # concat all scenes (not only shown) to compute averages
    all_df = pd.concat([df.assign(SceneName=sc) for sc, df in all_blocks], ignore_index=True)
    all_df = all_df.loc[:, ~all_df.columns.duplicated()]

    # coalesce "Method"
    def _nh(x: str) -> str:
        return " ".join(str(x).replace("\xa0", " ").strip().lower().split())
    method_like = [c for c in all_df.columns if _nh(c) == "method"]
    if len(method_like) > 1:
        all_df["Method"] = all_df[method_like].bfill(axis=1).iloc[:, 0]
        all_df.drop(columns=[c for c in method_like if c != "Method"], inplace=True)
    elif method_like and method_like[0] != "Method":
        all_df.rename(columns={method_like[0]: "Method"}, inplace=True)
    all_df["Method"] = all_df["Method"].astype(str)

    # get per-metric columns for averaging
    finals_map_all, aucs_map_all = _extract_metric_sets(all_df)
    # group by Method
    agg_cols = [finals_map_all[m].name for m in METRICS] + [aucs_map_all[m].name for m in METRICS]
    grp = (
        all_df.groupby("Method", dropna=False)[agg_cols]
        .mean(numeric_only=True)
        .reset_index()
    )

    # ranks across methods (averages) per subcell
    ranks_final_avg: Dict[str, pd.Series] = {}
    ranks_auc_avg:   Dict[str, pd.Series] = {}
    for m in METRICS:
        higher = _detect_metric_direction(m)
        ranks_final_avg[m] = _rank_series(grp[finals_map_all[m].name], higher)
        ranks_auc_avg[m]   = _rank_series(grp[aucs_map_all[m].name],   higher)

    avg_scene = "average"
    avg_row_count = len(grp)
    for idx, row in grp.iterrows():
        tds = []
        if idx == 0:
            tds.append(
                f"<td class='scene-vert scene-avg' rowspan='{avg_row_count}'>"
                f"<span class='scene-vert-text'>{avg_scene}</span></td>"
            )
        # method (base colored pastel green via CSS unless subcell chip overrides)
        tds.append(f"<td class='method avg-cell'><span class='cell-text'>{row['Method']}</span></td>")

        # 3 metric cells (Final/AUC subcells)
        for m in METRICS:
            vf = row[finals_map_all[m].name]
            va = row[aucs_map_all[m].name]
            is_best_f = (ranks_final_avg[m].get(idx, np.nan) == 1)
            is_best_a = (ranks_auc_avg[m].get(idx,   np.nan) == 1)
            td_extra  = " has-chip" if (is_best_f or is_best_a) else ""
            left_cls  = "sub final" + (" sub-best" if is_best_f else "")
            right_cls = "sub auc"   + (" sub-best" if is_best_a else "")
            tds.append(
                "<td class='metric-cell avg-cell" + td_extra + "'>"
                f"<div class='{left_cls}'><span class='cell-text'>{_two_dec_str(vf)}</span></div>"
                f"<div class='{right_cls}'><span class='cell-text'>{_two_dec_str(va)}</span></div>"
                "</td>"
            )

        body_rows.append("<tr class='avg-row'>" + "".join(tds) + "</tr>")

    # assemble table
    thead = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>"
    tbody = "<tbody>\n" + "\n".join(body_rows) + "\n</tbody>"

    # return f"""
    # <section class="card">
    #   <div class="table-wrap">
    #     <div class="table-surface">
    #       <table class="paper">
    #         {colgroup}
    #         {thead}
    #         {tbody}
    #       </table>
    #     </div>
    #   </div>
    # </section>
    # """

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

    blocks = _collect_blocks()

    # Filter for display (if whitelist provided)
    if SCENE_WHITELIST:
        show_blocks = [(sc, df) for sc, df in blocks if sc in SCENE_WHITELIST]
        if not show_blocks:
            raise RuntimeError("SCENE_WHITELIST excluded all scenes; nothing to display.")
    else:
        show_blocks = blocks

    html_section = _render_all_with_avg(show_blocks, blocks)

    css = """
    <style>
      :root {
        --border: #000;
        --bg-header: #000;  /* black header */
        --fg-header: #fff;
        --bg-alt: #fafafa;
        --card-bg: #fff;
        --shadow-card: 0 18px 40px rgba(0,0,0,.10), 0 6px 20px rgba(0,0,0,.06);

        /* font sizes (override as needed) */
        --fs-header: 24px;
        --fs-method: 18px;
        --fs-scene: 12px;
        --fs-number: 15px;

        /* chip sizing */
        --chip-inset-best-v: 3px;
        --chip-inset-best-h: 4px;
        --chip-radius-best: 8px;

        /* parity + avg colors */
        --gray-bg: #4a4a4a;
        --gray-fg: #ffffff;
        --avg-green: #e8f6ea;   /* pastel green */
      }
      * { box-sizing: border-box; }
      body {
        font-family: "Inter", system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
        margin: 28px; color: #111; background: #fff;
      }
      .card {
        border-radius: 18px; background: var(--card-bg); box-shadow: var(--shadow-card);
        margin: 24px 0 32px 0; border: 1px solid rgba(0,0,0,0.06); overflow: hidden;
      }
      .table-wrap { padding: 14px 14px 18px 14px; }
      .table-surface { position: relative; border-radius: 14px; overflow: hidden; }
      .table-surface::before {
        content: ""; position: absolute; inset: 0; border-radius: 14px;
        box-shadow: 0 0 0 1.5px var(--border) inset; z-index: 4; pointer-events: none;
      }

      table.paper {
        width: 100%; border-collapse: separate; border-spacing: 0; table-layout: fixed; background: #fff;
      }
      table.paper col.scene-col { width: 18px; } /* first narrow column */

      table.paper th, table.paper td {
        border: 1.5px solid var(--border); padding: 8px 10px; text-align: center; vertical-align: middle;
        background-clip: padding-box; position: relative; z-index: 1; font-size: 13.5px;
      }
      table.paper th { background: var(--bg-header); color: var(--fg-header); font-weight: 700; font-size: var(--fs-header); }

      /* enforce first column width */
      table.paper th:first-child,
      table.paper td.scene-vert {
        width: 18px !important; max-width: 18px !important; padding-left: 0; padding-right: 0; overflow: hidden;
      }

      /* outer border via wrapper */
      table.paper tr > *:first-child { border-left: 0; }
      table.paper tr > *:last-child  { border-right: 0; }
      table.paper thead tr:first-child > * { border-top: 0; }
      table.paper tbody tr:last-child  > * { border-bottom: 0; }

      table.paper tbody tr:nth-child(even) td { background: var(--bg-alt); }
      table.paper tr:hover td { filter: brightness(0.98); }

      /* Scene label (rotated) */
      .scene-vert { background: #fff; border-left: 0; position: relative; padding: 0; }
      .scene-vert-text {
        position: absolute; top: 50%; left: 50%;
        transform: translate(-50%, -50%) rotate(-90deg);
        transform-origin: center; white-space: nowrap; font-weight: 700; line-height: 1;
        font-size: var(--fs-scene);
      }

      /* Method font size */
      td.method .cell-text { font-size: var(--fs-method); line-height: 1.15; white-space: nowrap; }

      /* Numeric font size (subcells too) */
      table.paper td .cell-text { font-size: var(--fs-number); line-height: 1.1; white-space: nowrap; }

      /* ========== Metric cell split into two subcells (Final | AUC) ========== */
      td.metric-cell {
        position: relative; padding: 0; overflow: hidden;
      }
      /* Subcells fill halves */
      td.metric-cell .sub {
        position: absolute; top: 0; bottom: 0; display: flex; align-items: center; justify-content: center;
        padding: 6px 6px; /* inner breathing room */
      }
      td.metric-cell .sub.final { left: 0; right: 50%; }
      td.metric-cell .sub.auc   { left: 50%; right: 0; }

    /* Replace the old gradient divider: */
    td.metric-cell::before {
    content: "";
    position: absolute;
    left: 50%;
    top: 50%;
    width: 160%;
    height: 0;               /* just a line */
    border-top: 1px solid var(--border);
    transform: translate(-50%, -50%) rotate(-45deg); /* diagonal */
    z-index: 1;              /* under numbers, over cell bg */
    opacity: 0.9;            /* subtle */
    }

/* Keep numbers above the divider */
td.metric-cell .cell-text { position: relative; z-index: 2; }

      /* Raise text above diagonal */
      td.metric-cell .cell-text { position: relative; z-index: 2; }

      /* --------- Chip only for BEST subcells ---------- */
      td.metric-cell .sub.sub-best::after {
        content: "";
        position: absolute;
        /* chip smaller than subcell: inset within the half */
        inset: 10px;
        border-radius: var(--chip-radius-best);
        background: #ffcccc;      /* red chip */
        z-index: 1;               /* below text, above base */
        pointer-events: none;
      }
      td.metric-cell .sub.sub-best .cell-text {
        position: relative; z-index: 2; font-weight: 700;
        text-shadow: 0 1px 0 rgba(255,255,255,0.35);
      }

      /* ====== Even-scene parity styling (ALL non-chip cells go gray/white) ====== */
      /* Add .has-chip to td.metric-cell when any subcell is best so parity doesn't gray it */
      tr.scene-even td.method         { background: var(--gray-bg) !important; color: var(--gray-fg) !important; }
      tr.scene-even td.scene-vert     { background: var(--gray-bg) !important; color: var(--gray-fg) !important; }

      /* ====== Averages block base color ====== */
      tr.avg-row td.avg-cell { background: var(--avg-green) !important; color: #000 !important; }

    /* Even scene blocks: gray backdrop + white text for every data cell,
    including those that contain chips. Chips render on top. */
    tr.scene-even td {
    background: var(--gray-bg) !important;
    color: var(--gray-fg) !important;
    }

    /* Keep the averages row styling untouched */
    tr.avg-row td.avg-cell {
    background: var(--avg-green) !important;
    color: #000 !important;
    }

    /* Subcells stay transparent; only the td supplies the gray backdrop */
    td.metric-cell .sub { background: transparent; }

    /* Chip still renders on top of gray */
    td.metric-cell .sub.sub-best::after {
    /* (keep your existing styles) */
    z-index: 1;        /* under the number, above gray background */
    }
    td.metric-cell .cell-text { z-index: 2; }  /* already in your CSS */

/* Even scene blocks: gray base + white text on every cell (chips sit on top) */
tr.scene-even td {
  background: var(--gray-bg) !important;
  color: var(--gray-fg) !important;
}

/* Subcells are transparent so gray shows through */
td.metric-cell .sub { background: transparent; }

/* Chip still renders above gray */
td.metric-cell .sub.sub-best::after { z-index: 1; }
td.metric-cell .cell-text           { z-index: 2; }

/* Red chip text should always be black + bold, even in gray rows */
td.metric-cell .sub.sub-best .cell-text,
tr.scene-even td.metric-cell .sub.sub-best .cell-text {
  color: #000 !important;
  font-weight: 700;
  text-shadow: none;  /* optional */
}

/* === Font size knobs (independent) === */
:root{
  --fs-header: 24px;  /* header (with arrows) */
  --fs-scene:  12px;  /* rotated scene name (1st col) */
  --fs-method: 18px;  /* method names (2nd col) */
  --fs-value:  15px;/* numeric values in subcells */
}

/* Header */
table.paper thead th { 
  font-size: var(--fs-header) !important;
}

/* Rotated scene name (span inside the first column td) */
table.paper td.scene-vert > .scene-vert-text {
  font-size: var(--fs-scene) !important;
  line-height: 1 !important;
}

/* Method names (second column) */
table.paper td.method > .cell-text {
  font-size: var(--fs-method) !important;
  line-height: 1.15 !important;
  white-space: nowrap !important;
}

/* Numeric values (both Final and AUC subcells) */
table.paper td.metric-cell .sub > .cell-text {
  font-size: var(--fs-value) !important;
  line-height: 1.1 !important;
  white-space: nowrap !important;
}

/* Keep BEST chip text bold/black regardless of row color */
table.paper td.metric-cell .sub.sub-best > .cell-text {
  color: #000 !important;
  font-weight: 700 !important;
  text-shadow: none !important;
}

/* ====== PRINT-ONLY RULES FOR CHROME HEADLESS ====== */
@media print {
  /* 1) Page size & margins (tweak as needed). 
     Try A3 landscape first; if it still spills, bump to 320mm x 220mm. */
  @page {
    size: 300mm 220mm;   /* landscape-like; adjust width/height for your table */
    margin: 2mm;         /* near-zero margins to remove whitespace */
  }

  /* 2) Nuke default body margins & hide everything except the table card */
  html, body {
    margin: 0 !important;
    padding: 0 !important;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }

  /* If you can, give your main table wrapper an id="print-target". */
  body > *:not(#print-target) { display: none !important; }
  #print-target { display: block !important; }

  /* If you *don’t* have #print-target, fallback: keep only the first card */
  body > section.card ~ * { display: none !important; }

  /* 3) Remove shadows/outer borders so no ghost whitespace renders */
  .card, .table-surface { box-shadow: none !important; }
  .table-surface::before { box-shadow: none !important; }
  .table-wrap { padding: 0 !important; }

  /* 4) Tighten cell borders so they don't thicken in print */
  table.paper th, table.paper td { border-width: 1px !important; }

  /* 5) Prevent page breaks inside the table */
  .card, .table-wrap, .table-surface, table.paper, table.paper tr, table.paper td, table.paper th {
    page-break-inside: avoid;
    break-inside: avoid;
  }

  /* 6) Optional: final global scale to fit (0.85–1.0) */
  body { zoom: 0.94; }
}

    </style>
    """

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'>")
        f.write(css)
        f.write("</head><body>\n")
        f.write(html_section + "\n")
        f.write("</body></html>")
    print(f"HTML written to: {OUT_HTML}")

if __name__ == "__main__":
    main()
