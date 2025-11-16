# build_html_all_in_one_with_avg_inline.py
# -----------------------------------------------------------
# Single stacked table for all scenes + averages appended as a
# final "scene block" (scene name shown as 'average').
# - Even-numbered scene blocks: non-chip cells become gray with white text
# - Averages block: non-chip cells pastel green
# - Chips still render with rank-based colors/shadows on top
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

# ===== Column tags & behavior =====
FINAL_PREFIX = "Final "
AUC_PREFIX   = "AUC "
HIGHER_BETTER_SUBSTRINGS = ["PSNR", "SSIM"]
LOWER_BETTER_SUBSTRINGS  = ["LPIPS"]

SCENE_WHITELIST = [
    "caterpillar",
    "ignatius",
]

# ----- utilities -----
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

        aucs = [c for c in df.columns if str(c).startswith(AUC_PREFIX)]
        for c in aucs: df[c] = df[c] / 1000.0
        cleaned.append((scene, df))
    return cleaned

# ----- render -----
def _render_all_with_avg(show_blocks: list[tuple[str, pd.DataFrame]],
                         all_blocks: list[tuple[str, pd.DataFrame]]) -> str:
    # use show_blocks to draw the per-scene rows
    # use all_blocks to compute the averages block

    """Render one big table with all scenes stacked + an appended 'average' block."""
    # # From first block, derive column order
    # first_df = blocks[0][1]
    # finals = [c for c in first_df.columns if c.startswith(FINAL_PREFIX)]
    # aucs   = [c for c in first_df.columns if c.startswith(AUC_PREFIX)]
    # headers = ["", "Method"] + finals + aucs  # first header blank: rotated scene col

    # # colgroup with fixed first column width
    # n_cols = len(headers)
    # colgroup = '<colgroup><col class="scene-col" />' + '<col />' * (n_cols - 1) + '</colgroup>'

    # body_rows: List[str] = []

    # --- header and colgroup derived from the first SHOWN block ---
    first_df = show_blocks[0][1] if show_blocks else all_blocks[0][1]
    finals = [c for c in first_df.columns if c.startswith(FINAL_PREFIX)]
    aucs   = [c for c in first_df.columns if c.startswith(AUC_PREFIX)]
    headers = ["", "Method"] + finals + aucs
    n_cols = len(headers)
    colgroup = '<colgroup><col class="scene-col" />' + '<col />' * (n_cols - 1) + '</colgroup>'

    body_rows = []

    # render each scene block (apply scene-even class based on scene index for gray non-chip cells)
    for scene_idx, (scene, df) in enumerate(show_blocks):
        # ensure consistent columns across scenes
        for col in finals + aucs:
            if col not in df.columns:
                df[col] = np.nan
        df = df[["Scene", "Method"] + finals + aucs].copy()

        # ranks per metric within this scene
        ranks = {col: _rank_series(df[col], _detect_metric_direction(col)) for col in finals + aucs}

        row_count = len(df)
        if row_count == 0:
            continue
        first_index = df.index[0]

        scene_class = "scene-even" if (scene_idx % 2 == 1) else "scene-odd"
        for ridx, row in df.iterrows():
            tr_cls = scene_class
            tds = []
            if ridx == first_index:
                tds.append(
                    f"<td class='scene-vert' rowspan='{row_count}'>"
                    f"<span class='scene-vert-text'>{scene}</span></td>"
                )
            method_val = _to_scalar_method(row["Method"])
            tds.append(f"<td class='method'><span class='cell-text'>{method_val}</span></td>")
            # finals
            for col in finals:
                val = _two_dec_str(row[col]); rnk = ranks[col].get(ridx, np.nan)
                cls = "cell-best" if rnk == 1 else "cell-2nd" if rnk == 2 else "cell-3rd" if rnk == 3 else ""
                tds.append(f"<td class='{cls}'><span class='cell-text'>{val}</span></td>")
            # aucs
            for col in aucs:
                val = _two_dec_str(row[col]); rnk = ranks[col].get(ridx, np.nan)
                cls = "cell-best" if rnk == 1 else "cell-2nd" if rnk == 2 else "cell-3rd" if rnk == 3 else ""
                tds.append(f"<td class='{cls}'><span class='cell-text'>{val}</span></td>")

            body_rows.append(f"<tr class='{tr_cls}'>" + "".join(tds) + "</tr>")

    # ------- averages block appended at bottom -------
    # Build averages by method across scenes
    all_df = pd.concat([df.assign(SceneName=sc) for sc, df in all_blocks], ignore_index=True)
    # prevent duplicate-named columns from concat
    all_df = all_df.loc[:, ~all_df.columns.duplicated()]

    def _nh(x: str) -> str:
        return " ".join(str(x).replace("\xa0", " ").strip().lower().split())
    method_like = [c for c in all_df.columns if _nh(c) == "method"]
    if len(method_like) > 1:
        all_df["Method"] = all_df[method_like].bfill(axis=1).iloc[:, 0]
        all_df.drop(columns=[c for c in method_like if c != "Method"], inplace=True)
    elif method_like and method_like[0] != "Method":
        all_df.rename(columns={method_like[0]: "Method"}, inplace=True)
    all_df["Method"] = all_df["Method"].astype(str)

    finals = [c for c in all_df.columns if c.startswith(FINAL_PREFIX)]
    aucs   = [c for c in all_df.columns if c.startswith(AUC_PREFIX)]

    grp = (
        all_df.groupby("Method", dropna=False)[finals + aucs]
        .mean(numeric_only=True)
        .reset_index()
    )

    # rank across methods on the averages block
    rank_maps_avg = {col: _rank_series(grp[col], _detect_metric_direction(col)) for col in finals + aucs}

    # append averages as a final scene block; label scene as "average"
    avg_scene = "average"
    avg_row_count = len(grp)
    if avg_row_count > 0:
        # first averages row carries the scene-name cell with rowspan
        for idx, row in grp.iterrows():
            tds = []
            if idx == 0:
                tds.append(
                    f"<td class='scene-vert scene-avg' rowspan='{avg_row_count}'>"
                    f"<span class='scene-vert-text'>{avg_scene}</span></td>"
                )
            # method
            tds.append(f"<td class='method avg-cell'><span class='cell-text'>{row['Method']}</span></td>")
            # finals
            for col in finals:
                val = _two_dec_str(row[col]); rnk = rank_maps_avg[col].get(idx, np.nan)
                cls = "cell-best" if rnk == 1 else "cell-2nd" if rnk == 2 else "cell-3rd" if rnk == 3 else "avg-cell"
                tds.append(f"<td class='{cls}'><span class='cell-text'>{val}</span></td>")
            # aucs
            for col in aucs:
                val = _two_dec_str(row[col]); rnk = rank_maps_avg[col].get(idx, np.nan)
                cls = "cell-best" if rnk == 1 else "cell-2nd" if rnk == 2 else "cell-3rd" if rnk == 3 else "avg-cell"
                tds.append(f"<td class='{cls}'><span class='cell-text'>{val}</span></td>")

            # mark whole row as average-row for styling
            body_rows.append("<tr class='avg-row'>" + "".join(tds) + "</tr>")

    # assemble table
    thead = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>"
    tbody = "<tbody>\n" + "\n".join(body_rows) + "\n</tbody>"

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

    blocks = _collect_blocks()
    # Filter for display (if whitelist provided)
    if SCENE_WHITELIST:
        show_blocks = [(sc, df) for sc, df in blocks if sc in SCENE_WHITELIST]
        if not show_blocks:
            raise RuntimeError("SCENE_WHITELIST excluded all scenes; nothing to display.")
    else:
        show_blocks = blocks

    # Build single table with averages computed over ALL blocks
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

        /* rank-dependent chip sizing (optional tweak) */
        --chip-inset-best-v: 2px;  --chip-inset-best-h: 3px;
        --chip-inset-2nd-v:  4px;  --chip-inset-2nd-h:  6px;
        --chip-inset-3rd-v:  6px;  --chip-inset-3rd-h:  8px;
        --chip-radius-best: 10px;  --chip-radius-2nd: 9px;  --chip-radius-3rd: 8px;

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
      table.paper col.scene-col { width: 18px; } /* adjust scene column width here */

      table.paper th, table.paper td {
        border: 1.5px solid var(--border); padding: 8px 10px; text-align: center; vertical-align: middle;
        background-clip: padding-box; position: relative; z-index: 1; font-size: 13.5px;
      }
      table.paper th { background: var(--bg-header); color: var(--fg-header); font-weight: 700; }

      /* blank first header, enforce width on first column cells */
      table.paper th:first-child,
      table.paper td.scene-vert {
        width: 18px !important; max-width: 18px !important; padding-left: 0; padding-right: 0; overflow: hidden;
      }

      /* Remove outer borders (outer stroke drawn by wrapper) */
      table.paper tr > *:first-child { border-left: 0; }
      table.paper tr > *:last-child  { border-right: 0; }
      table.paper thead tr:first-child > * { border-top: 0; }
      table.paper tbody tr:last-child  > * { border-bottom: 0; }

      /* Default row zebra (will be overridden by scene-even rules where needed) */
      table.paper tbody tr:nth-child(even) td { background: var(--bg-alt); }
      table.paper tr:hover td { filter: brightness(0.98); }

      td.method { font-weight: 600; }
      .cell-text { position: relative; z-index: 3; }

      /* Rotated scene text */
      .scene-vert { background: #fff; border-left: 0; position: relative; padding: 0; }
      .scene-vert-text {
        position: absolute; top: 50%; left: 50%;
        transform: translate(-50%, -50%) rotate(-90deg);
        transform-origin: center; white-space: nowrap; font-weight: 700; letter-spacing: .4px; line-height: 1;
      }

      /* Chips for top-3: variable size by rank */
      .cell-best, .cell-2nd, .cell-3rd { background: transparent !important; }
      .cell-best::after, .cell-2nd::after, .cell-3rd::after {
        content: ""; position: absolute; pointer-events: none; z-index: 2;
      }
      .cell-best::after { inset: var(--chip-inset-best-v) var(--chip-inset-best-h); border-radius: var(--chip-radius-best); background: #ffcccc;
        box-shadow: 0 12px 28px rgba(0,0,0,0.30), 0 6px 16px rgba(0,0,0,0.22), 0 2px 6px rgba(0,0,0,0.18), inset 0 1px 0 rgba(255,255,255,0.55); }
      .cell-2nd::after  { inset: var(--chip-inset-2nd-v)  var(--chip-inset-2nd-h);  border-radius: var(--chip-radius-2nd);  background: #ffdd99;
        box-shadow: 0 9px 20px rgba(0,0,0,0.24), 0 4px 12px rgba(0,0,0,0.18), 0 1px 4px rgba(0,0,0,0.14), inset 0 1px 0 rgba(255,255,255,0.50); }
      .cell-3rd::after  { inset: var(--chip-inset-3rd-v)  var(--chip-inset-3rd-h);  border-radius: var(--chip-radius-3rd);  background: #fff2b3;
        box-shadow: 0 6px 14px rgba(0,0,0,0.18), 0 3px 8px rgba(0,0,0,0.12), 0 1px 3px rgba(0,0,0,0.10), inset 0 1px 0 rgba(255,255,255,0.45); }

      /* ---- Scene parity styling: for EVEN scenes only, make NON-CHIP cells gray w/ white text ---- */
      /* Rows receive .scene-even class per scene block */
      tr.scene-even td:not(.cell-best):not(.cell-2nd):not(.cell-3rd) {
        background: var(--gray-bg) !important;
        color: var(--gray-fg) !important;
      }
      /* Also darken the Method cell in even scenes */
      tr.scene-even td.method { background: var(--gray-bg) !important; color: var(--gray-fg) !important; }

      /* Keep the rotated scene label readable on even scenes */
      tr.scene-even td.scene-vert { background: #fff !important; color: #000 !important; }

      /* ---- Averages block styling ---- */
      /* Marked as .avg-row and cells get .avg-cell if not chips */
      tr.avg-row td.avg-cell {
        background: var(--avg-green) !important;
        color: #000 !important;
      }
      /* scene cell for averages (left rail) */
      td.scene-avg { background: #fff !important; }

    /* ------------ Font size controls (tweak these) ------------ */
    :root{
    --fs-header: 20px;   /* table header cells (black row) */
    --fs-method: 18px;   /* method names (second column) */
    --fs-scene: 12px;    /* rotated scene names (first column) */
    --fs-number: 18px; /* numeric values in metric cells */
    }

    /* Header row */
    table.paper th {
    font-size: var(--fs-header);
    }

    /* Method names (the 2nd column) */
    td.method .cell-text {
    font-size: var(--fs-method);
    line-height: 1.15;
    white-space: nowrap;
    }

    /* Rotated scene label (the 1st column) */
    .scene-vert-text {
    font-size: var(--fs-scene);
    line-height: 1;
    }

    /* Numeric values (all metric cells, with or without chips) */
    table.paper td:not(.method) .cell-text {
    font-size: var(--fs-number);
    line-height: 1.1;
    white-space: nowrap;
    }

    /* --- Keep ONLY the red chip for best; others become normal cells --- */

    /* Nuke any styling for 2nd/3rd: remove chip and reset background/text */
    .cell-2nd, .cell-3rd {
    background: transparent !important;
    }
    .cell-2nd::after, .cell-3rd::after {
    content: none !important;    /* kills the chip pseudo-element */
    box-shadow: none !important;
    }

    /* Ensure normal cell text styling for 2nd/3rd */
    .cell-2nd .cell-text, .cell-3rd .cell-text {
    font-weight: 400;            /* regular weight */
    }

    /* Keep/strengthen the BEST (red) chip */
    .cell-best {
    background: transparent !important;  /* chip is drawn by ::after */
    }
    .cell-best::after {
    /* your existing red chip styles can stay; these are safe defaults */
    content: "";
    position: absolute;
    inset: var(--chip-inset-best-v, 2px) var(--chip-inset-best-h, 3px);
    border-radius: var(--chip-radius-best, 10px);
    background: #ffcccc;
    pointer-events: none;
    z-index: 2;
    box-shadow:
        0 12px 28px rgba(0,0,0,0.30),
        0 6px 16px rgba(0,0,0,0.22),
        0 2px 6px rgba(0,0,0,0.18),
        inset 0 1px 0 rgba(255,255,255,0.55);
    }

    /* Make the BEST chip's value bold (and crisp over the highlight) */
    .cell-best .cell-text {
    position: relative;
    z-index: 3;     /* above the chip */
    font-weight: 700;
    /* optional tiny contrast lift */
    text-shadow: 0 1px 0 rgba(255,255,255,0.35);
    }
    /* --- Kill chip shadows entirely --- */
    .cell-best::after,
    .cell-2nd::after,
    .cell-3rd::after {
    box-shadow: none !important;
    }

    /* (Optional) also disable row hover dimming, if you had it */
    table.paper tr:hover td {
    filter: none !important;
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
