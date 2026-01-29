# WandB Analysis Pipeline Usage

## Overview

`wandb_analysis_pipeline.py` is a unified script that combines:
- WandB data downloading
- Metrics aggregation (Final, AUC, MaxDelta)
- Table generation (CSV and Markdown)

Key feature: **Separates runs by sweep parameters** (KDTREE_FILTER, NUM_INITIAL_VIEWS, LOAD_3D_POINTS) extracted from WandB run configs.

## Output Structure

```
results/{entity}/
    wandb_exports/
        state.json                    # Incremental update tracking
        curves_with_params.csv        # Raw curves with sweep params
    analysis/
        by_variant/
            kdtree_0__init_1__pts_1/   # Variant-specific outputs
                summary.csv
                summary.md
        tables/
            summary_all.csv           # All variants comparison
            summary_all.md
```

## Usage Examples

### Full Pipeline
```bash
python scripts/wandb_analysis_pipeline.py
```

### Specify Project Prefixes
Project prefixes match `PROJECT_NAME` from `sweep_scenes.py`. The script will pull all WandB projects starting with these prefixes.

```bash
# Single prefix (matches projects like "next-best-view-rebuttal__caterpillar")
python scripts/wandb_analysis_pipeline.py --project-prefixes next-best-view-rebuttal__

# Multiple prefixes
python scripts/wandb_analysis_pipeline.py --project-prefixes next-best-view__ next-best-view-rebuttal__
```

### Download Only
```bash
python scripts/wandb_analysis_pipeline.py --download-only
```

### Aggregate Existing Data Only
```bash
python scripts/wandb_analysis_pipeline.py --aggregate-only
```

### Filter by Sweep Parameters
```bash
python scripts/wandb_analysis_pipeline.py \
    --kdtree-filter false \
    --num-initial-views 1 10 \
    --load-3d-points true
```

### Force Re-download All Runs
```bash
python scripts/wandb_analysis_pipeline.py --force-reload
```

### Show Individual Scenes (Not Just Averages)
```bash
python scripts/wandb_analysis_pipeline.py --show-individual-scenes
```

### Specific Scenes Only
```bash
python scripts/wandb_analysis_pipeline.py --scenes caterpillar train ignatius
```

### Output Format Options
```bash
# CSV only
python scripts/wandb_analysis_pipeline.py --format csv

# Markdown only
python scripts/wandb_analysis_pipeline.py --format markdown

# Both (default)
python scripts/wandb_analysis_pipeline.py --format both
```

### Custom Entity
```bash
python scripts/wandb_analysis_pipeline.py --entity my-wandb-entity
```

## All CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--entity` | WandB entity name | `chengine-stanford-university` |
| `--project-prefixes` | Project prefix(es) to filter (space-separated) | `next-best-view__` |
| `--scenes` | Specific scenes to process | All scenes |
| `--metrics` | Metrics to aggregate | `/psnr /lpips /ssim` |
| `--auc-scale` | Scale factor for AUC values | `1000.0` |
| `--download-only` | Only download, skip aggregation | False |
| `--aggregate-only` | Only aggregate existing data | False |
| `--force-reload` | Force re-download all runs | False |
| `--show-individual-scenes` | Show individual scene rows | False |
| `--format` | Output format (csv/markdown/both) | `both` |
| `--kdtree-filter` | Filter by kdtree_filter values (true/false) | None (all) |
| `--num-initial-views` | Filter by num_initial_views values | None (all) |
| `--load-3d-points` | Filter by load_3d_points values (true/false) | None (all) |

## Project Naming Convention

Projects in WandB are named using the pattern from `sweep_scenes.py`:
```
{PROJECT_NAME}__{scene_name}
```

For example, with `PROJECT_NAME = "next-best-view-rebuttal"`:
- `next-best-view-rebuttal__caterpillar`
- `next-best-view-rebuttal__train`
- `next-best-view-rebuttal__bicycle`

Use `--project-prefixes next-best-view-rebuttal__` to pull only those projects.
