# build_html_from_grouped_csv_vertical_scene.py
# -----------------------------------------------------------
# Reads all_scenes_grouped.csv and renders one sleek HTML with
# a rounded "card" per scene. Adds a vertical first column with
# the scene name (rowspan). Robust header dedupe + top-3 chips.
# -----------------------------------------------------------

import os
from io import StringIO
from typing import List, Tuple, Optional, Dict
import unicodedata
import numpy as np
import pandas as pd

# ====================== CONFIG ======================
IN_CSV = "../results/chengine-stanford-university/tables/all_scenes_grouped.csv"
OUT_HTML = "../results/chengine-stanford-university/tables/output.html"

FINAL_PREFIX = "Final "
AUC_PREFIX   = "AUC "

HIGHER_BETTER_SUBSTRINGS = ["PSNR", "SSIM"]  # higher is better
LOWER_BETTER_SUBSTRINGS  = ["LPIPS"]         # lower is better

# Chip fill colors (best -> 1)
HILITE = {1: "#ffcccc", 2: "#ffdd99", 3: "#fff2b3"}  # red, orange, yellow
# ====================================================


def _split_grouped_csv_lines(path: str) -> List[Tuple[str, List[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines()]
    blocks: List[Tuple[str, List[str]]] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if not line.startswith("Scene,"):
            i += 1
            continue
        scene = line.split(",", 1)[1] if "," in line else "UNKNOWN_SCENE"
        i += 1
        block_lines: List[str] = []
        while i < n and lines[i].strip():
            block_lines.append(lines[i])
            i += 1
        blocks.append((scene, block_lines))
    return blocks


def _norm_header(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).replace("\xa0", " ").strip()
    if s.lower().startswith("unnamed:"):
        s = s.split(":", 1)[-1]
    return " ".join(s.split()).lower()


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate headers (case/space/nbsp insensitive) by taking first non-null per row."""
    cols = list(df.columns)
    keys = [_norm_header(c) for c in cols]
    pretty: Dict[str, str] = {}
    for c, k in zip(cols, keys):
        pretty.setdefault(k, str(c))
    out = pd.DataFrame(index=df.index)
    from collections import defaultdict
    groups = defaultdict(list)
    for c, k in zip(cols, keys):
        groups[k].append(c)
    for k, cols_k in groups.items():
        if len(cols_k) == 1:
            out[pretty[k]] = df[cols_k[0]]
        else:
            out[pretty[k]] = df[cols_k].bfill(axis=1).iloc[:, 0]
    return out


def _ensure_single_method_column(df: pd.DataFrame) -> pd.DataFrame:
    ms = [c for c in df.columns if _norm_header(c) == "method"]
    if not ms:
        raise ValueError("Block missing 'Method' column.")
    if len(ms) > 1:
        df["Method"] = df[ms].bfill(axis=1).iloc[:, 0]
        df = df.drop(columns=[c for c in ms if c != "Method"])
    elif ms[0] != "Method":
        df = df.rename(columns={ms[0]: "Method"})
    return df


def _rank_series(s: pd.Series, higher_is_better: bool) -> pd.Series:
    return s.rank(ascending=not higher_is_better, method="min")


def _two_dec_str(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "—"


def _detect_metric_direction(col_name: str) -> bool:
    name_upper = str(col_name).upper()
    if any(tok.upper() in name_upper for tok in HIGHER_BETTER_SUBSTRINGS):
        return True
    if any(tok.upper() in name_upper for tok in LOWER_BETTER_SUBSTRINGS):
        return False
    return True


def _to_scalar_method(val):
    if isinstance(val, pd.Series):
        val = val.dropna()
        val = val.iloc[0] if len(val) else None
    if val is None:
        return "—"
    try:
        if isinstance(val, float) and np.isnan(val):
            return "—"
    except Exception:
        pass    # non-float
    return str(val)


def _render_scene_table(scene: str, df: pd.DataFrame) -> str:
    # keep only non-rank columns
    keep_cols = ["Method"] + [c for c in df.columns if not str(c).startswith("Rank ") and c != "Scene"]
    df = df[keep_cols].copy()

    # coerce numeric metric columns
    num_cols = [c for c in df.columns if c != "Method"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    finals = [c for c in df.columns if str(c).startswith(FINAL_PREFIX)]
    aucs   = [c for c in df.columns if str(c).startswith(AUC_PREFIX)]
    headers = ["Scene", "Method"] + finals + aucs  # add first "Scene" column header

    # ranks for highlighting
    rank_maps = {}
    for col in finals + aucs:
        higher = _detect_metric_direction(col)
        rank_maps[col] = _rank_series(df[col], higher)

    row_count = len(df)

    # build rows with a single vertical scene cell spanning all rows (rowspan)
    rows_html = []
    for r_idx, row in df.iterrows():
        method_val = _to_scalar_method(row["Method"])
        tds = []
        if r_idx == df.index[0]:
            # first row: add the scene cell once with rowspan
            tds.append(
                f"<td class='scene-vert' rowspan='{row_count}'>"
                f"<span class='scene-vert-text'>{scene}</span></td>"
            )
        # method cell
        tds.append(f"<td class='method'><span class='cell-text'>{method_val}</span></td>")
        # metric cells
        for col in finals:
            val = _two_dec_str(row[col])
            rnk = rank_maps[col].get(r_idx, np.nan)
            cls = "cell-best" if rnk == 1 else "cell-2nd" if rnk == 2 else "cell-3rd" if rnk == 3 else ""
            tds.append(f"<td class='{cls}'><span class='cell-text'>{val}</span></td>")
        for col in aucs:
            val = _two_dec_str(row[col])
            rnk = rank_maps[col].get(r_idx, np.nan)
            cls = "cell-best" if rnk == 1 else "cell-2nd" if rnk == 2 else "cell-3rd" if rnk == 3 else ""
            tds.append(f"<td class='{cls}'><span class='cell-text'>{val}</span></td>")
        rows_html.append("<tr>" + "".join(tds) + "</tr>")

    thead = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>"
    tbody = "<tbody>\n" + "\n".join(rows_html) + "\n</tbody>"

    return f"""
    <section class="card">
      <div class="card-header">Scene: {scene}</div>
      <div class="table-wrap">
        <div class="table-surface">
          <table class="paper">
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
    if not blocks:
        raise RuntimeError("No scene blocks found in the grouped CSV.")

    sections: List[str] = []
    for scene, block_lines in blocks:
        block_csv = "\n".join(block_lines)
        block = pd.read_csv(StringIO(block_csv))
        block = _coalesce_duplicate_columns(block)
        block = _ensure_single_method_column(block)
        if "Scene" not in block.columns:
            block.insert(0, "Scene", scene)
        block = block[block["Scene"].astype(str) == scene]
        sections.append(_render_scene_table(scene, block))

    # CSS: same look + vertical scene column + chips + stronger red/orange shadows already in place
    css = """
    <style>
      :root {
        --border: #000;
        --bg-header: #000;  /* black header */
        --fg-header: #fff;
        --bg-alt: #fafafa;
        --card-bg: #fff;
        --shadow-card: 0 18px 40px rgba(0,0,0,.10), 0 6px 20px rgba(0,0,0,.06);
      }
      * { box-sizing: border-box; }
      body {
        font-family: "Inter", system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
        margin: 28px;
        color: #111;
        background: #fff;
      }
      .card {
        border-radius: 18px;
        background: var(--card-bg);
        box-shadow: var(--shadow-card);
        margin: 24px 0 32px 0;
        border: 1px solid rgba(0,0,0,0.06);
        overflow: hidden;
      }
      .card-header {
        font-weight: 700;
        font-size: 18px;
        padding: 14px 18px;
        background: linear-gradient(180deg, #f7f7f7 0%, #ededed 100%);
        border-bottom: 1px solid rgba(0,0,0,0.10);
      }
      .table-wrap { padding: 14px 14px 18px 14px; }

      .table-surface {
        position: relative;
        border-radius: 14px;
        overflow: hidden;
      }
      .table-surface::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 14px;
        box-shadow: 0 0 0 1.5px var(--border) inset; /* continuous rounded outer stroke */
        z-index: 4;
        pointer-events: none;
      }

      table.paper {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        table-layout: fixed;
        background: #fff;
      }

      table.paper th, table.paper td {
        border: 1.5px solid var(--border);
        padding: 8px 10px;
        text-align: center;
        vertical-align: middle;
        background-clip: padding-box;
        position: relative;
        z-index: 1;
        font-size: 13.5px;
      }
      table.paper th {
        background: var(--bg-header);  /* black */
        color: var(--fg-header);       /* white */
        font-weight: 700;
      }

      /* Remove outer borders; wrapper draws the rounded outline */
      table.paper tr > *:first-child { border-left: 0; }
      table.paper tr > *:last-child  { border-right: 0; }
      table.paper thead tr:first-child > * { border-top: 0; }
      table.paper tbody tr:last-child  > * { border-bottom: 0; }

      /* Zebra + hover */
      table.paper tbody tr:nth-child(even) td { background: var(--bg-alt); }
      table.paper tr:hover td { filter: brightness(0.98); }

      td.method { font-weight: 600; }
      .cell-text { position: relative; z-index: 3; }

      /* Scene vertical column */
      /* Narrow first column, acts as positioning container */
        .scene-vert {
        width: 36px;               /* adjust to taste */
        min-width: 36px;
        background: #fff;
        border-left: 0;            /* outer stroke drawn by wrapper */
        position: relative;        /* anchor for absolutely positioned text */
        padding: 0;                /* no padding; we’ll center via transform */
        }
        /* Rotate the scene name 90° and center it within the cell */
        .scene-vert-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%) rotate(-90deg);  /* <-- the rotation */
        transform-origin: center;
        white-space: nowrap;       /* keep name on one line */
        font-weight: 700;
        letter-spacing: 0.5px;
        line-height: 1;
        /* optional subtle anti-aliasing hint */
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        }

      /* Top-3 chips (smaller rounded rectangle inside cell) */
      .cell-best, .cell-2nd, .cell-3rd { background: transparent !important; }
      .cell-best::after,
      .cell-2nd::after,
      .cell-3rd::after {
        content: "";
        position: absolute;
        inset: 3px 4px;           /* adjust to make chips smaller/larger */
        border-radius: 8px;
        pointer-events: none;
        z-index: 2;               /* under text, above base cell */
      }
      .cell-best::after {
        background: #ffcccc;
        box-shadow:
          0 12px 28px rgba(0,0,0,0.30),
          0 6px 16px rgba(0,0,0,0.22),
          0 2px 6px rgba(0,0,0,0.18),
          inset 0 1px 0 rgba(255,255,255,0.55);
      }
      .cell-2nd::after {
        background: #ffdd99;
        box-shadow:
          0 9px 20px rgba(0,0,0,0.24),
          0 4px 12px rgba(0,0,0,0.18),
          0 1px 4px rgba(0,0,0,0.14),
          inset 0 1px 0 rgba(255,255,255,0.50);
      }
      .cell-3rd::after {
        background: #fff2b3;
        box-shadow:
          0 6px 14px rgba(0,0,0,0.18),
          0 3px 8px rgba(0,0,0,0.12),
          0 1px 3px rgba(0,0,0,0.10),
          inset 0 1px 0 rgba(255,255,255,0.45);
      }
    </style>
    """

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'>")
        f.write(css)
        f.write("</head><body>\n")
        for sec in sections:
            f.write(sec)
            f.write("\n")
        f.write("</body></html>")

    print(f"HTML written to: {OUT_HTML}")


if __name__ == "__main__":
    main()
