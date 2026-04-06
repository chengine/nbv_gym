# NBV-Gym: Next Best View Selection for Gaussian Splatting

A framework for optimizing camera view selection in 3D Gaussian Splatting using coverage-based metrics.

---

## Dependencies

| Dependency | Version |
|------------|---------|
| Python | 3.10 |
| Nerfstudio | From source (as of 11/21/2025) |
| PyTorch | 2.5.1+cu124 |
| gsplat | 1.5.3 |
| Pillow | < 11 |
| numpy | < 2.0 |

---

## Installation

### 1. Create a conda environment

```bash
conda create -n nbv_gym python=3.10 -y
conda activate nbv_gym
```

### 2. Install Nerfstudio from source

```bash
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
pip install -e .
cd ..
```

### 3. Clone and install NBV-Gym

```bash
git clone <repository-url>
cd nbv-gym
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` fixes dependency versions that Nerfstudio installs incorrectly:

- **PyTorch**: Downgraded to 2.5.1+cu124 (Nerfstudio's default ships CUDA 13.0 which requires a very recent NVIDIA driver)
- **gsplat**: Upgraded to 1.5.3 (Nerfstudio pins 1.4.0)
- **Pillow**: Pinned below 11 (11+ breaks nerfstudio's `pil_to_numpy`)
- **numpy**: Pinned below 2.0

> **Note:** If your NVIDIA driver supports a different CUDA version than 12.4, edit `requirements.txt` to use the appropriate PyTorch index URL (see [PyTorch installation](https://pytorch.org/get-started/locally/)).

### 5. Register with Nerfstudio

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
| `fisher_rf` | Fisher-RF uncertainty (requires modified rasterizer) |

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

Fisher-RF uncertainty is available as a view metric (`--pipeline.view-metric fisher_rf`). This requires installing the modified Gaussian rasterizer:

```bash
# Clone and install the modified rasterizer (use gcc/g++ 11)
git clone --recursive https://github.com/JiangWenPL/modified-diff-gaussian-rasterization-w-depth
cd modified-diff-gaussian-rasterization-w-depth
pip install -e . --no-build-isolation
```

> **Note:** A standalone Fisher-RF model with full training support will be added in a future release.

---

## License

See [LICENSE](LICENSE) for details.
