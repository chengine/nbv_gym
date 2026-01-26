# NBV-Gym: Next Best View Selection for Gaussian Splatting

A framework for optimizing camera view selection in 3D Gaussian Splatting using coverage-based metrics.

---

## Dependencies

| Dependency | Version |
|------------|---------|
| Python | 3.10 |
| gsplat | 1.5.3 |
| Nerfstudio | From source (as of 11/21/2025) |

> **Note:** You may need to overwrite the `gsplat` library included in Nerfstudio with the required version. If you encounter issues, try downgrading `numpy` to below 2.0.

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd nbv-gym
```

### 2. Install as a Python package

```bash
pip install -e .
```

### 3. Register with Nerfstudio

```bash
ns-install-cli
```

---

## Usage

Run NBV-Gym using the standard Nerfstudio training command:

```bash
ns-train nbv-splat --data <path-to-data> --output-dir <output-directory>
```

### Configuration Options

#### View Selector

Controls how new views are selected during progressive training.

```bash
--pipeline.view-selector <mode>
```

| Mode | Description |
|------|-------------|
| `all` | Use all views from the start (no progressive selection) |
| `random` | Randomly select views to add |
| `basic` | Use view metrics to intelligently select the most informative views |

#### View Metrics

When using `--pipeline.view-selector basic`, specify the metric for scoring candidate views:

```bash
--pipeline.view-metric <metric>
```

| Metric | Description |
|--------|-------------|
| `coverage` | Basic coverage metric |
| `fig` | Fisher Information Gain |
| `view_fig` | View-weighted Fisher Information Gain |
| `fig_diag` | Diagonal approximation of FIG |
| `view_fig_diag` | View-weighted diagonal FIG |
| `fig_color_field` | FIG with color field consideration |

#### KD-Tree Filtering

Enable spatial filtering to reduce the candidate pool during view selection:

```bash
--pipeline.view-selection-use-kdtree-filter True
--pipeline.view-selection-num-nearest-neighbors 5
```

#### Progressive Training Parameters

```bash
--pipeline.add-every-n-steps 200      # Steps between adding new views
--pipeline.add-num-views 1            # Number of views to add each time
--pipeline.start-num-views 10         # Initial number of views
```

### Example Commands

**Basic training with all views:**
```bash
ns-train nbv-splat --data ./data/scene \
    --output-dir ./outputs \
    --pipeline.view-selector all
```

**Progressive training with coverage-based selection:**
```bash
ns-train nbv-splat --data ./data/scene \
    --output-dir ./outputs \
    --pipeline.view-selector basic \
    --pipeline.view-metric coverage \
    --pipeline.add-every-n-steps 200 \
    --pipeline.add-num-views 1 \
    --pipeline.start-num-views 5
```

**Progressive training with KD-tree filtering:**
```bash
ns-train nbv-splat --data ./data/scene \
    --output-dir ./outputs \
    --pipeline.view-selector basic \
    --pipeline.view-metric fig \
    --pipeline.view-selection-use-kdtree-filter True \
    --pipeline.view-selection-num-nearest-neighbors 10
```

---

## Baselines

### Fisher-RF

> **Coming Soon:** Fisher-RF baseline will be implemented in a future release.

---

## License

See [LICENSE](LICENSE) for details.
