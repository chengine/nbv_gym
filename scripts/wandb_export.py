# wandb_nbv_export.py
import os
import re
import time
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import wandb

# ------------- CONFIG -------------
ENTITY = "chengine-stanford-university"    # <- set me
PROJECT_PREFIX = "next-best-view__"  # discover projects that start with this
# If you want to restrict to certain scenes, list them; else discover all
SCENES: Optional[List[str]] = [
    "caterpillar",
    "train",
    "ignatius",
    "shiny_statue_6pm",
    "space_laces_4pm",
    "chair_3pm",
]
# Desired metrics from "Eval Images Metrics Dict (all images)"
DESIRED = ["/psnr", "/lpips", "/ssim"]
# Throttle to be nice to the API
RATE_LIMIT_SLEEP = 0.25
# Output dir
OUT_DIR = "../results/wandb_exports"
# ----------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
IMG_DIR = os.path.join(OUT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

TAB_METRICS_HINT = "Eval Images Metrics Dict (all images)"
TAB_IMAGES_HINT = "Eval Images"

RUN_PATTERNS = [
    # scene__uncertainty__selector__YYYYMMDD-HHMM
    re.compile(r"^(?P<scene>[^_]+)__(?P<uncertainty>[^_]+)__(?P<selector>[^_]+)__(?P<ts>\d{8}-\d{4})$"),
    # scene__uncertainty__YYYYMMDD-HHMM
    re.compile(r"^(?P<scene>[^_]+)__(?P<uncertainty>[^_]+)__(?P<ts>\d{8}-\d{4})$"),
]

METRIC_KEY_MAP = {
    "/psnr": "Eval Images Metrics Dict (all images)/psnr",
    "/lpips": "Eval Images Metrics Dict (all images)/lpips",
    "/ssim": "Eval Images Metrics Dict (all images)/ssim",
}

def parse_run_name(name: str) -> Dict[str, Optional[str]]:
    """Parse run name into scene, uncertainty, selector, timestamp."""
    for pat in RUN_PATTERNS:
        m = pat.match(name or "")
        if m:
            d = m.groupdict()
            d.setdefault("selector", None)
            return d
    # fallback: try splitting
    parts = (name or "").split("__")
    scene = parts[0] if parts else None
    uncertainty = parts[1] if len(parts) > 1 else None
    selector = parts[2] if len(parts) > 3 else (parts[2] if len(parts) == 4 else None)
    ts = parts[-1] if parts else None
    return {"scene": scene, "uncertainty": uncertainty, "selector": selector, "ts": ts}

def discover_projects(api: wandb.Api, entity: str, prefix: str) -> List[str]:
    projects = []
    for p in api.projects(entity=entity):
        if p.name.startswith(prefix):
            projects.append(p.name)
    return sorted(projects)

def project_scene(name: str) -> str:
    """Extract scene name from project 'next-best-view__{scene}'."""
    if name.startswith(PROJECT_PREFIX):
        return name[len(PROJECT_PREFIX):]
    return name

def union_history_keys(run: wandb.apis.public.Run, sample_rows: int = 200) -> set:
    keys = set()
    i = 0
    for row in run.scan_history():
        keys.update(row.keys())
        i += 1
        if i >= sample_rows:
            break
    return keys

def find_metric_keys(all_keys: Iterable[str], desired_suffixes: List[str]) -> Dict[str, str]:
    """
    Find the fully-qualified metric key for each desired suffix (e.g., '/psnr')
    under the metrics tab. Returns mapping like {'/psnr': 'Eval Images Metrics Dict (all images)/psnr', ...}
    """
    found = {}
    for suf in desired_suffixes:
        # prefer keys that contain the metrics-tab hint and end with suffix
        candidates = [k for k in all_keys if k.endswith(suf) and TAB_METRICS_HINT.lower() in k.lower()]
        if not candidates:
            # fallback: any key that ends with the suffix
            candidates = [k for k in all_keys if k.endswith(suf)]
        # pick the shortest (usually the cleanest)
        if candidates:
            found[suf] = sorted(candidates, key=len)[0]
    return found

def find_image_key(all_keys: Iterable[str]) -> Optional[str]:
    """
    Find the fully-qualified image key for the images tab (…/img).
    """
    candidates = [k for k in all_keys if k.endswith("/img") and TAB_IMAGES_HINT.lower() in k.lower()]
    if not candidates:
        candidates = [k for k in all_keys if k.endswith("/img")]
    return sorted(candidates, key=len)[0] if candidates else None

def fetch_metrics_long(
    run: wandb.apis.public.Run,
    metric_key_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Stream rows via scan_history(keys=metric_keys) and emit long-format records directly.
    Works even when _timestamp/_step are missing.
    """
    if not metric_key_map:
        return pd.DataFrame()

    wanted = list(metric_key_map.values())
    rows = []
    for row in run.scan_history(keys=wanted + ["_step", "_timestamp"]):
        step = row.get("_step")
        ts = row.get("_timestamp")
        # normalize timestamp if numeric
        if isinstance(ts, (int, float)):
            import pandas as pd
            ts = pd.to_datetime(ts, unit="s", utc=True)
        for suf, fkey in metric_key_map.items():
            if fkey in row and row[fkey] is not None:
                rows.append({"step": step, "timestamp": ts, "metric": suf, "value": row[fkey]})

    import pandas as pd
    return pd.DataFrame(rows)

def download_eval_images(
    run: wandb.apis.public.Run,
    img_key: str,
    dest_dir: str
) -> pd.DataFrame:
    """
    Iterate history; whenever img_key is present, download images.
    Returns a manifest dataframe with columns: [step, timestamp, file_path, caption?].
    """
    rows = []
    for row in run.scan_history(keys=[img_key]):
        if img_key not in row or row[img_key] is None:
            continue
        payload = row[img_key]
        ts = row.get("_timestamp", None)
        step = row.get("_step", None)
        # W&B may store a list of images or a single image object
        imgs = payload if isinstance(payload, list) else [payload]
        for idx, img in enumerate(imgs):
            # Public API image dicts usually have "path" (e.g., "media/images/xxx.png")
            path = None
            if isinstance(img, dict):
                path = img.get("path") or img.get("file", {}).get("path")
                caption = img.get("caption")
            else:
                path = None
                caption = None
            if not path:
                continue
            # download
            try:
                fobj = run.file(path)
                # Organize per-run folder
                run_dir = os.path.join(dest_dir, f"{run.project}_{run.id}")
                os.makedirs(run_dir, exist_ok=True)
                local_path = os.path.join(run_dir, os.path.basename(path))
                fobj.download(root=run_dir, replace=True)
                # ensure full path recorded
                local_path = os.path.join(run_dir, os.path.basename(path))
                rows.append({
                    "step": step,
                    "timestamp": pd.to_datetime(ts, unit="s", utc=True) if isinstance(ts, (int, float)) else ts,
                    "local_path": local_path,
                    "wandb_path": path,
                    "caption": caption,
                })
            except Exception as e:
                rows.append({
                    "step": step, "timestamp": ts, "local_path": None,
                    "wandb_path": path, "caption": None, "error": str(e)
                })
    return pd.DataFrame(rows)

def main():
    api = wandb.Api(timeout=60)
    # Discover matching projects
    projects = discover_projects(api, ENTITY, PROJECT_PREFIX)
    if SCENES:
        projects = [p for p in projects if project_scene(p) in SCENES]
    if not projects:
        print("No projects found.")
        return

    all_curves = []
    all_imgs = []

    for proj in projects:
        runs = api.runs(f"{ENTITY}/{proj}")
        for run in runs:
            time.sleep(RATE_LIMIT_SLEEP)
            # Parse identifiers
            name_info = parse_run_name(run.name or "")
            scene = name_info.get("scene") or project_scene(proj)
            uncertainty = name_info.get("uncertainty")
            selector = name_info.get("selector")
            # Inspect keys to find the exact columns
            keys = union_history_keys(run, sample_rows=200)
            key_map = find_metric_keys(keys, DESIRED)
            img_key = find_image_key(keys)

            # Fetch curves
            curves = fetch_metrics_long(run, METRIC_KEY_MAP)
            if not curves.empty:
                curves.insert(0, "project", proj)
                curves.insert(1, "scene", scene)
                curves.insert(2, "run_id", run.id)
                curves.insert(3, "run_name", run.name)
                curves.insert(4, "uncertainty", uncertainty)
                curves.insert(5, "selector", selector)
                all_curves.append(curves)

            # Fetch images manifest (optional)
            if img_key:
                imgs = download_eval_images(run, img_key, IMG_DIR)
                if not imgs.empty:
                    imgs.insert(0, "project", proj)
                    imgs.insert(1, "scene", scene)
                    imgs.insert(2, "run_id", run.id)
                    imgs.insert(3, "run_name", run.name)
                    imgs.insert(4, "uncertainty", uncertainty)
                    imgs.insert(5, "selector", selector)
                    all_imgs.append(imgs)

    # Save outputs
    curves_df = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame(
        columns=["project","scene","run_id","run_name","uncertainty","selector","step","timestamp","value","metric"]
    )
    images_df = pd.concat(all_imgs, ignore_index=True) if all_imgs else pd.DataFrame(
        columns=["project","scene","run_id","run_name","uncertainty","selector","step","timestamp","local_path","wandb_path","caption"]
    )

    curves_csv = os.path.join(OUT_DIR, "curves_long.csv")
    images_csv = os.path.join(OUT_DIR, "images_manifest.csv")
    curves_df.to_csv(curves_csv, index=False)
    images_df.to_csv(images_csv, index=False)
    # Optional Parquet
    try:
        curves_df.to_parquet(os.path.join(OUT_DIR, "curves_long.parquet"), index=False)
    except Exception:
        pass

    print(f"\nSaved:\n- {curves_csv}\n- {images_csv}")
    if not curves_df.empty:
        print(curves_df.head())
    if not images_df.empty:
        print(images_df.head())

if __name__ == "__main__":
    main()
