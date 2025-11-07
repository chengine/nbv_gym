# Shadow Splat Progressive View Selection

## Overview
Iterative training framework that starts with a small active view set and progressively expands it using coverage-based or random selection strategies.

## Configuration
**Pipeline:** `shadow_splat/pipeline.py:114-137` (`ViewSelectionPipelineConfig`)
- `add_every_n_steps: int = 1000` — Schedule for expanding active set (training steps)
- `add_num_views: int = 1` — Views added per expansion iteration
- `view_selector: str` — Selection mode: `"random"`, `"all"`, `"optics"`, or custom class path

**Optics Selector Options:**
- `optics_intrinsics_scale: float = 1.0` — Resolution downscaling for faster scoring (< 1.0 speeds up)
- `optics_use_kdtree_filter: bool = False` — Spatial filtering to limit candidates
- `optics_num_nearest_neighbors: int = 5` — Neighbors to consider when filtering

**DataManager:** `shadow_splat/datamanager.py:136-138` (`ViewSelectionDataManagerConfig`)
- `start_num_views: int = 1` — Initial active set size

## Training Loop
**Pipeline:** `shadow_splat/pipeline.py:176-196` (`ViewSelectionPipeline.get_train_loss_dict()`)
```
1. Every add_every_n_steps iterations:
   → Call datamanager.expand_active_set(k=add_num_views, step=step, model=model, pipeline=pipeline)
2. Sample training batches only from active_train_indices
3. Update model and loss
```

## View Selection Mechanism

### 1. Active Set Management
**DataManager:** `shadow_splat/datamanager.py:141-279` (`ViewSelectionDataManager`)
- `active_train_indices: List[int]` — Current training indices (starts with `start_num_views`)
- `all_train_indices: List[int]` — All available training indices
- `expand_active_set(k)` — Adds top-k remaining views (lines 181-222)

### 2. Selection Strategies
**Factory:** `shadow_splat/view_selector.py:155-201` (`create_view_selector()`)

#### Random Selection
**Class:** `shadow_splat/view_selector.py:33-42` (`RandomViewSelector`)
- Uniformly samples k views from remaining indices

#### Optics-Based Selection (Primary)
**Class:** `shadow_splat/view_selector.py:54-152` (`OpticsViewSelector`)

**Algorithm:**
1. **Candidate Pool** (lines 114-131):
   - Extract candidate cameras from `remaining_indices`
   - Optionally filter to k-nearest neighbors of last added view using KD-tree on camera origins

2. **Coverage Scoring Loop** (lines 134-145):
   - For each candidate camera:
     - Call `model.coverage_score_for_camera(camera, intrinsics_scale)`
     - Returns: `sum(coverage_pixels)` — total coverage score
   - Accumulate scores per camera

3. **Selection** (lines 147-150):
   - Sort by score (descending)
   - Select top-k indices

#### All Selection
**Class:** `shadow_splat/view_selector.py:45-51` (`AllViewSelector`)
- Selects all remaining views at once

### 3. Coverage Scoring
**Model Method:** `shadow_splat/model.py:586-642` (`ShadowSplatModel.coverage_score_for_camera()`)

1. Switch to eval mode
2. Optionally scale camera intrinsics: `intrinsics_scale * (fx, fy, cx, cy, width, height)`
3. Render: `self.get_outputs(camera, light=None)` — outputs include `"coverage"` channel
4. Return: `coverage.sum()` — sum of all coverage pixel values

### 4. Coverage Metric Computation
**Utilities:** `shadow_splat/util/coverage.py`

#### Per-Gaussian Coverage Tracking (during training)
- Maintains: `coverage_counts[N_gaussians, G_bins]` where G=128 (fibonacci sphere bins)
- Updated by: `update_view_coverage_for_frustum()` (lines 34-79)
  - Projects each Gaussian to current camera frustum
  - Computes bin direction for each visible Gaussian
  - Increments `coverage_counts[gaussian_id, bin_idx]`

#### Coverage Metric During Rendering
**Function:** `compute_coverage_per_gaussian()` (lines 83-127)

For each visible Gaussian in current view:
1. Compute cosines with all bin directions: `cos_all = inference_dirs @ bin_dirs^T` [N, G]
2. Compute per-Gaussian median coverage count: `med = coverage_counts.median(dim=1)` [N]
3. Mask "well-covered" bins: `invalid = (coverage_counts < med) | (coverage_counts == 0)` [N, G]
4. **Coverage score per Gaussian:** `max((cos_all + 1) / 2) * (1 - invalid)` [N]
   - Normalized cosine ∈ [0,1]
   - Multiplied by binary mask: 1 if bin is under-covered, 0 otherwise
5. Output: per-pixel coverage values (rendered as coverage channel)

**Fibonacci Sphere Binning:** `fibonacci_sphere()` (lines 10-23)
- Generates 128 evenly-spaced unit direction vectors on sphere
- Uses golden ratio for even distribution

## Data Flow
```
ViewSelectionPipeline.get_train_loss_dict(step)
  ↓ (every add_every_n_steps)
ViewSelectionDataManager.expand_active_set(k)
  ↓
ViewSelector.select_views(active_indices, remaining_indices, k)
  ↓ (if OpticsViewSelector)
For each candidate camera:
  ShadowSplatModel.coverage_score_for_camera(camera)
    ↓
  model.get_outputs(camera) → outputs["coverage"]
    ↓
  compute_coverage_per_gaussian() → per-pixel novelty scores
    ↓
  sum(coverage) → scalar score
  ↓
Select top-k by score
  ↓
Add to active_train_indices
Log metrics: Views Added, Total Active Views
```

## Key Files
- **Pipeline orchestration:** `shadow_splat/pipeline.py:139-223`
- **View selection logic:** `shadow_splat/view_selector.py`
- **Active set management:** `shadow_splat/datamanager.py:141-279`
- **Coverage scoring:** `shadow_splat/model.py:586-642`
- **Coverage utilities:** `shadow_splat/util/coverage.py`
- **Configuration:** `shadow_splat/config.py` (example config with ViewSelectionPipelineConfig)

## Metrics & Logging
`shadow_splat/datamanager.py:208-218` — Logs to wandb/tensorboard:
- `"View Selection/Views Added"` — Number of views added in this expansion
- `"View Selection/Total Active Views"` — Cumulative active view count
