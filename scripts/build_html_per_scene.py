# build_html_from_grouped_csv_vertical_scene_thin.py
# -----------------------------------------------------------
# Same functionality as before, but:
#  - thinner first column for scene name (rotated 90°)
#  - removes the per-table scene header bar
# -----------------------------------------------------------

import os
from io import StringIO
from typing import List, Tuple, Optional, Dict
import unicodedata
import numpy as np
import pandas as pd

# ENTITY = "chengine-stanford-university"
ENTITY = "navlab"
IN_CSV = f"../results/{ENTITY}/tables/all_scenes_grouped.csv"
OUT_HTML = f"../results/{ENTITY}/tables/output.html"

FINAL_PREFIX = "Final "
AUC_PREFIX   = "AUC "
HIGHER_BETTER_SUBSTRINGS = ["PSNR", "SSIM"]
LOWER_BETTER_SUBSTRINGS  = ["LPIPS"]
HILITE = {1: "#ffcccc", 2: "#ffdd99", 3: "#fff2b3"}

def _split_grouped_csv_lines(path: str) -> List[Tuple[str, List[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines()]
    blocks, i, n = [], 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if not line: i += 1; continue
        if not line.startswith("Scene,"): i += 1; continue
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

def _rank_series(s: pd.Series, higher: bool) -> pd.Series:
    return s.rank(ascending=not higher, method="min")

def _two_dec_str(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    try: return f"{float(x):.2f}"
    except Exception: return "—"

def _detect_metric_direction(col: str) -> bool:
    u = str(col).upper()
    if any(t in u for t in (t.upper() for t in HIGHER_BETTER_SUBSTRINGS)): return True
    if any(t in u for t in (t.upper() for t in LOWER_BETTER_SUBSTRINGS)): return False
    return True

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

def _render_scene_table(scene: str, df: pd.DataFrame) -> str:
    keep = ["Method"] + [c for c in df.columns if not str(c).startswith("Rank ") and c != "Scene"]
    df = df[keep].copy()
    for c in [c for c in df.columns if c != "Method"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    finals = [c for c in df.columns if str(c).startswith(FINAL_PREFIX)]
    aucs   = [c for c in df.columns if str(c).startswith(AUC_PREFIX)]
    headers = ["", "Method"] + finals + aucs

    # divide AUCs by 1000 for display
    for col in aucs:
        df[col] = df[col] / 1000.0

    n_cols = len(headers)
    colgroup = '<colgroup><col class="scene-col" />' + '<col />' * (n_cols - 1) + '</colgroup>'

    ranks = {}
    for col in finals + aucs:
        ranks[col] = _rank_series(df[col], _detect_metric_direction(col))

    row_count = len(df)
    rows_html = []
    for r_idx, row in df.iterrows():
        method_val = _to_scalar_method(row["Method"])
        tds = []
        if r_idx == df.index[0]:
            tds.append(
                f"<td class='scene-vert' rowspan='{row_count}'>"
                f"<span class='scene-vert-text'>{scene}</span></td>"
            )
        tds.append(f"<td class='method'><span class='cell-text'>{method_val}</span></td>")
        for col in finals:
            val = _two_dec_str(row[col]); rnk = ranks[col].get(r_idx, np.nan)
            cls = "cell-best" if rnk == 1 else "cell-2nd" if rnk == 2 else "cell-3rd" if rnk == 3 else ""
            tds.append(f"<td class='{cls}'><span class='cell-text'>{val}</span></td>")
        for col in aucs:
            val = _two_dec_str(row[col]); rnk = ranks[col].get(r_idx, np.nan)
            cls = "cell-best" if rnk == 1 else "cell-2nd" if rnk == 2 else "cell-3rd" if rnk == 3 else ""
            tds.append(f"<td class='{cls}'><span class='cell-text'>{val}</span></td>")
        rows_html.append("<tr>" + "".join(tds) + "</tr>")

    thead = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>"
    tbody = "<tbody>\n" + "\n".join(rows_html) + "\n</tbody>"

    # NOTE: removed the per-table header (no <div class="card-header">Scene: ...</div>)
    return f"""
    <section class="card">
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
    blocks = _split_grouped_csv_lines(IN_CSV)
    if not blocks: raise RuntimeError("No scene blocks found in the grouped CSV.")

    sections: List[str] = []
    for scene, blk in blocks:
        block = pd.read_csv(StringIO("\n".join(blk)))
        block = _coalesce_duplicate_columns(block)
        block = _ensure_single_method_column(block)
        if "Scene" not in block.columns:
            block.insert(0, "Scene", scene)
        block = block[block["Scene"].astype(str) == scene]
        sections.append(_render_scene_table(scene, block))

    css = """
    <style>
      :root {
        --border: #000;
        --bg-header: #000;
        --fg-header: #fff;
        --bg-alt: #fafafa;
        --card-bg: #fff;
        --shadow-card: 0 18px 40px rgba(0,0,0,.10), 0 6px 20px rgba(0,0,0,.06);
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
      table.paper th, table.paper td {
        border: 1.5px solid var(--border); padding: 8px 10px; text-align: center; vertical-align: middle;
        background-clip: padding-box; position: relative; z-index: 1; font-size: 13.5px;
      }
      table.paper th { background: var(--bg-header); color: var(--fg-header); font-weight: 700; }
      table.paper tr > *:first-child { border-left: 0; }
      table.paper tr > *:last-child  { border-right: 0; }
      table.paper thead tr:first-child > * { border-top: 0; }
      table.paper tbody tr:last-child  > * { border-bottom: 0; }
      table.paper tbody tr:nth-child(even) td { background: var(--bg-alt); }
      table.paper tr:hover td { filter: brightness(0.98); }

      td.method { font-weight: 600; }
      .cell-text { position: relative; z-index: 3; }

      /* Thinner first column with 90° rotated scene text */
      .scene-vert {
        width: 10px; min-width: 10px;   /* <-- make thinner here */
        background: #fff; border-left: 0;
        position: relative; padding: 0;
      }
      .scene-vert-text {
        position: absolute; top: 50%; left: 50%;
        transform: translate(-50%, -50%) rotate(-90deg);
        transform-origin: center;
        white-space: nowrap; font-weight: 700; letter-spacing: 0.4px; line-height: 1;
        -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
      }

      /* Chips for top-3 (same as before) */
      .cell-best, .cell-2nd, .cell-3rd { background: transparent !important; }
      .cell-best::after, .cell-2nd::after, .cell-3rd::after {
        content: ""; position: absolute; inset: 3px 4px; border-radius: 8px; pointer-events: none; z-index: 2;
      }
    /* 1st = biggest chip */
    .cell-best::after {
    inset: 4px 4px;          /* top/bot, left/right — least inset = largest chip */
    border-radius: 10px;     /* slightly rounder for the larger chip */
    background: #ffcccc;
    box-shadow:
        0 12px 28px rgba(0,0,0,0.30),
        0 6px 16px rgba(0,0,0,0.22),
        0 2px 6px rgba(0,0,0,0.18),
        inset 0 1px 0 rgba(255,255,255,0.55);
    }

    /* 2nd = medium chip */
    .cell-2nd::after {
    inset: 4px 16px;          /* a bit smaller */
    border-radius: 9px;
    background: #ffdd99;
    box-shadow:
        0 9px 20px rgba(0,0,0,0.24),
        0 4px 12px rgba(0,0,0,0.18),
        0 1px 4px rgba(0,0,0,0.14),
        inset 0 1px 0 rgba(255,255,255,0.50);
    }

    /* 3rd = smallest chip */
    .cell-3rd::after {
    inset: 4px 32px;          /* smallest */
    border-radius: 8px;
    background: #fff2b3;
    box-shadow:
        0 6px 14px rgba(0,0,0,0.18),
        0 3px 8px rgba(0,0,0,0.12),
        0 1px 3px rgba(0,0,0,0.10),
        inset 0 1px 0 rgba(255,255,255,0.45);
    }
      /* First column width is controlled by the colgroup, not the td */
        table.paper col.scene-col { width: 22px; }  /* <-- change to 10px, 22px, etc. */

        /* Make sure the header cell and body cells obey the fixed width */
        table.paper th:first-child,
        table.paper td.scene-vert {
        width: 22px !important;     /* keep in sync with col.scene-col */
        max-width: 22px !important;
        padding-left: 0;
        padding-right: 0;
        overflow: hidden;            /* prevent content from expanding the column */
        }

        /* Your rotated scene label stays absolutely positioned inside the fixed cell */
        .scene-vert { position: relative; padding: 0; }
        .scene-vert-text {
        position: absolute; top: 50%; left: 50%;
        transform: translate(-50%, -50%) rotate(-90deg);
        white-space: nowrap;
        }
    </style>
    """

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'>")
        f.write(css)
        f.write("</head><body>\n")
        for sec in sections:
            f.write(sec + "\n")
        f.write("</body></html>")
    print(f"HTML written to: {OUT_HTML}")

if __name__ == "__main__":
    main()
