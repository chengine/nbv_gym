# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NBV-Gym is a research framework for **progressive Next Best View selection in 3D Gaussian Splatting**, built as a Nerfstudio plugin. It trains 3D Gaussian Splat models while intelligently selecting which camera views to add during training, using coverage-based and Fisher information metrics.

## Environment

Always use the `nbv_gym` conda environment when running code for this project. Since `conda activate` doesn't work in non-interactive shells, prefix commands with:

```bash
conda run -n nbv_gym <command>
```

## Key Commands

```bash
# Install (editable mode + register with Nerfstudio)
pip install -e .
ns-install-cli

# Train (primary entry point)
ns-train nbv-splat --data <path-to-data> --output-dir <output-dir>

# Train with progressive view selection
ns-train nbv-splat --data <path> \
    --pipeline.view-selector basic \
    --pipeline.view-metric coverage \
    --pipeline.add-every-n-steps 200 \
    --pipeline.add-num-views 1 \
    --pipeline.start-num-views 5

# Evaluate a trained model
python scripts/eval.py

# Formatting (pre-commit hooks use black with 100 char line length)
black --line-length=100 <file>
```

## Architecture

The project extends Nerfstudio's plugin system. All custom components live in `nbv_gym/` and override the corresponding Nerfstudio base classes:

```
config.py  →  MethodSpecification "nbv-splat" (entry point registered in pyproject.toml)
    ↓
trainer.py  →  NBVTrainer (custom viewer integration)
    ↓
pipeline.py  →  ViewSelectionPipeline (extends VanillaPipeline)
    ├── Orchestrates progressive view expansion every N steps
    ├── Instantiates the view selector via create_view_selector()
    └── Calls datamanager.expand_active_set() on schedule
    ↓
datamanager.py  →  ViewSelectionDataManager
    ├── Maintains active_train_indices vs remaining views
    └── Delegates scoring to view_selector
    ↓
model.py  →  NBVSplatModel (extends SplatfactoModel) / FisherSplatModel
    ├── Core Gaussian Splatting model with view metric support
    ├── setup_view_metric() configures which metric to compute
    └── FisherSplatModel used only when view_metric="fisher_rf"
```

### View Selection System (`view_selector.py`)

Factory pattern via `create_view_selector(mode)`:
- **AllViewSelector** — uses all views from start (no progressive selection)
- **RandomViewSelector** — random selection, optional KD-tree spatial filtering
- **VanillaViewSelector** — metric-based selection using coverage/FIG/Fisher-RF scores

### Coverage & Metrics (`util/coverage.py`)

Scoring functions for candidate views: `coverage`, `fig`, `view_fig`, `fig_diag`, `view_fig_diag`, `fig_color_field`, `fisher_rf`. These operate on per-Gaussian statistics using spherical bins and Fisher information.

### Rendering (`util/more_rendering.py`)

Custom moment rasterization (`moment_rasterization()`) for first/second moment rendering used in uncertainty estimation. Integrates with gsplat 1.5.3 CUDA kernels.

## Dependencies

- **Python 3.10**, **Nerfstudio** (from source), **gsplat 1.5.3**
- gsplat version may need to override what Nerfstudio installs; numpy must be < 2.0
- Fisher-RF metric requires a [modified Gaussian rasterizer](https://github.com/JiangWenPL/modified-diff-gaussian-rasterization-w-depth)

## Code Style

- Black formatter, 100 char line length (enforced by pre-commit hooks)
- Pre-commit also runs: YAML check, trailing whitespace fix, nbstripout for notebooks
