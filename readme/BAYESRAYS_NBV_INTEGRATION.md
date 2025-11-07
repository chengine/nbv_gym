# BayesRays Next-Best-View Integration

## Overview

This document describes the integration of BayesRays uncertainty-based next-best-view (NBV) selection into shadow_splat's online progressive training framework. The integration enables NeRF models to be trained with adaptive view selection, where new views are added based on model uncertainty rather than just coverage metrics.

**Key Design Principle:** Non-destructive integration. All existing functionality is preserved; BayesRays is added as an optional view selection strategy alongside `random`, `optics`, and `all`.

## Architecture

### High-Level Flow

```
ViewSelectionPipeline.get_train_loss_dict(step)
  ↓
  Every add_every_n_steps:
    IF bayes selector:
      HessianComputer.compute_hessian_from_datamanager()
        ↓ (loops through active training views)
        ↓ (accumulates Hessian gradients)
        ↓ Returns: Hessian tensor
      ↓
      ViewSelectionDataManager.expand_active_set(hessian=hessian)
        ↓
        BayesRaysViewSelector.select_views(hessian=hessian)
          ↓ (for each candidate camera)
          ShadowSplatModel.uncertainty_score_for_camera(camera, hessian)
            ↓
            Inject Hessian into model
            ↓
            Render with uncertainty computation
            ↓
            Extract & aggregate uncertainty
            ↓
          ↓
          Select top-k by max uncertainty
      ↓
      Add selected views to training
    ELSE:
      [existing optics/random flow]
  ↓
  Continue training with expanded dataset
```

## Components

### 1. HessianComputer (`shadow_splat/bayesrays_utils.py`)

**Class:** `HessianComputer`

**Purpose:** Computes Hessian tensor in-memory from active training data without pausing training.

**Key Methods:**

- `__init__(lod: int, device)`: Initialize with grid resolution (log2) and device
  - Creates HashEncoding deformation field (matches BayesRays)
  - Sets up Hessian accumulator

- `compute_hessian_from_datamanager(model, datamanager, max_batches)`: Main entry point
  - Loops through active training data (like BayesRays' `main()`)
  - Calls `_get_unc_nerfacto()` to get ray samples and offsets
  - Accumulates Hessian via `find_uncertainty()`
  - Returns accumulated Hessian tensor

- `find_uncertainty(points, deform_points, rgb, spatial_distortion)`: Compute Hessian contribution
  - Mirrors BayesRays' `ComputeUncertainty.find_uncertainty()` exactly
  - Computes per-channel gradients and bins into spatial grid
  - Returns Hessian contribution for one batch

**Configuration Parameters:**
- `lod`: Hessian grid resolution. Default 8 → 256³ grid
- `max_batches`: Limit training data used. None = use all active views

### 2. BayesRaysViewSelector (`shadow_splat/view_selector.py`)

**Class:** `BayesRaysViewSelector` (extends `ViewSelector`)

**Purpose:** Selects views by maximizing model uncertainty.

**Key Methods:**

- `__init__(reduce_mode: str = "mean", lod: int = 8)`: Initialize selector
  - `reduce_mode`: How to aggregate per-pixel uncertainty ("mean" or "sum")
  - `lod`: Must match HessianComputer's lod

- `select_views(active_indices, remaining_indices, num_to_select, **kwargs)`: Main selection
  - Expects `hessian` in kwargs (passed from pipeline)
  - For each candidate camera:
    - Calls `model.uncertainty_score_for_camera(camera, hessian, reduce_mode, lod)`
    - Accumulates scores
  - Selects top-k by max uncertainty (uncertainty maximization)

**Factory Integration:**
```python
create_view_selector(
    mode="bayes",  # or "bayesrays"
    bayes_reduce_mode="mean",
    bayes_lod=8,
)
```

### 3. Model Method: `uncertainty_score_for_camera()` (`shadow_splat/model.py`)

**Purpose:** Evaluate uncertainty for a single candidate camera.

**Implementation:**
1. Switch to eval mode
2. Inject Hessian into model attributes:
   - `self.hessian`: Hessian tensor
   - `self.lod`: Grid resolution
   - `self.get_uncertainty`: Bound method for uncertainty computation
   - `self.N`: Approximate ray dataset size (4096×1000)
   - `self.filter_out`, `self.filter_thresh`, `self.white_bg`, `self.black_bg`: Uncertainty filtering options

3. Dynamically patch `self.get_outputs` with uncertainty-enabled version
   - Uses `bayesrays.scripts.output_uncertainty.get_output_fn()` to get appropriate method
   - For Nerfacto: uses `get_output_nerfacto_new()` which includes uncertainty channel

4. Render: `outputs = self.get_outputs(camera)` → includes `"uncertainty"` key

5. Aggregate:
   - `reduce_mode == "mean"`: `uncertainty.mean()`
   - `reduce_mode == "sum"`: `uncertainty.sum()`

6. Return scalar uncertainty score

### 4. Pipeline Integration (`shadow_splat/pipeline.py`)

**Extended Config:** `ViewSelectionPipelineConfig`

New parameters:
```python
bayes_reduce_mode: str = "mean"          # Aggregation: "mean" or "sum"
bayes_lod: int = 8                       # Hessian grid resolution (log2)
bayes_max_hessian_batches: Optional[int] = None  # Max training batches for Hessian
```

**Pipeline Initialization:**
- If `view_selector == "bayes"`:
  - Creates `BayesRaysViewSelector` with config params
  - Initializes `HessianComputer` with lod and device
  - Sets up caching for Hessian (computed once per `add_every_n_steps`)

**Training Loop (`get_train_loss_dict()`):**
1. Every `add_every_n_steps` iterations:
2. If using BayesRays selector:
   - Compute Hessian: `hessian_computer.compute_hessian_from_datamanager(...)`
   - Cache it: `self._cached_hessian = hessian`
3. Pass Hessian to expand_active_set: `expand_active_set(..., hessian=hessian)`
4. Selector uses it to score candidates

## Usage

### Configuration File

```yaml
pipeline:
  _target_: shadow_splat.pipeline.ViewSelectionPipeline
  view_selector: "bayes"          # Enable BayesRays selection
  add_every_n_steps: 1000         # Expand every 1000 training steps
  add_num_views: 1                # Add 1 view per expansion

  # BayesRays specific
  bayes_reduce_mode: "mean"       # or "sum"
  bayes_lod: 8                    # Hessian grid: 256³
  bayes_max_hessian_batches: null # Use all active views for Hessian

datamanager:
  _target_: shadow_splat.datamanager.ViewSelectionDataManager
  start_num_views: 1              # Start with 1 view
```

### Command Line

If using argument parser:

```bash
ns-train <method> \
  --pipeline.view_selector "bayes" \
  --pipeline.add_every_n_steps 1000 \
  --pipeline.add_num_views 1 \
  --pipeline.bayes_reduce_mode "mean" \
  --pipeline.bayes_lod 8
```

## Backward Compatibility

**Preserved Functionality:**
- `view_selector="random"`: Works as before (no changes)
- `view_selector="optics"`: Works as before (no changes)
- `view_selector="all"`: Works as before (no changes)
- `view_selector=None`: Defaults to random (no changes)
- Standard (non-view-selection) pipelines: Unaffected

**No Breaking Changes:**
- All new code is additive
- Existing methods unchanged
- Optional imports (only loads bayesrays_utils if bayes selector used)
- Graceful fallback if Hessian computation fails

## Technical Details

### Hessian Computation

The `HessianComputer` closely mirrors BayesRays' `ComputeUncertainty.main()`:

1. **Setup:**
   - Initializes Hessian accumulator: `[((2^lod)+1)³]` grid
   - Creates HashEncoding deformation field matching BayesRays spec

2. **Per-Batch:**
   - Gets ray samples via forward pass
   - Computes gradients w.r.t. deformation field offsets
   - Bins gradients into spatial grid via trilinear interpolation
   - Accumulates sum of squared gradients per bin

3. **Differences from CLI:**
   - No checkpoint I/O (stays in memory)
   - Loops over active training views (not all views)
   - Integrates seamlessly into training loop

### Uncertainty Rendering

Uses BayesRays' output injection approach:

1. **Hessian → Uncertainty:**
   - Pre-computed Hessian (H) stored in model
   - Per-point uncertainty: `u = 1/H` (inverse Hessian)
   - Applied with regularization: `H_reg = H/N + λ`

2. **Rendering:**
   - Per-sample uncertainty computed via interpolation
   - Alpha-blended with RGB during volume rendering
   - Clipped to reasonable range and normalized

3. **Aggregation:**
   - Per-pixel uncertainty → scalar via mean or sum
   - Higher uncertainty indicates more informative view

## Performance Considerations

### Computational Cost

**Hessian Computation:**
- Per expansion: O(N_active) ray samples processed
- Gradient computation: ~1 forward + 3 backward passes per batch
- Grid binning: O(N_samples) with trilinear interpolation
- Cost: ~1-5 minutes per expansion (for typical NeRF)

**Per-Candidate Scoring:**
- One full render per candidate camera
- No downscaling available (unlike optics selector)
- Cost scales with #candidates and image resolution

### Optimization Tips

1. **Reduce Candidates:** Use `optics_use_kdtree_filter=True` to limit candidates spatially
2. **Limit Hessian Batches:** Set `bayes_max_hessian_batches=100` to speed up Hessian computation
3. **Skip Early Expansion:** Increase `add_every_n_steps` (e.g., 5000) for less frequent expansion
4. **Batch Size:** Use smaller `add_num_views` (e.g., 1) to reduce per-expansion cost

## Dependencies

Required:
- `bayesrays` (BayesRays module)
- `nerfstudio`
- `torch`

Optional:
- `scipy` (for KD-tree filtering)

## Troubleshooting

### Error: "BayesRays selector requires bayesrays_utils module"
- Ensure `bayesrays_utils.py` is in the shadow_splat directory
- Check that BayesRays dependencies are installed

### Error: "Unknown view selector mode 'bayes'"
- Ensure `view_selector` config is exactly `"bayes"` or `"bayesrays"`
- Check YAML syntax

### Hessian computation very slow
- Reduce `bayes_max_hessian_batches` (default: all active views)
- Increase `add_every_n_steps` (fewer expansions)
- Use GPU with more memory

### Uncertainty values all zero
- Check that Hessian was computed (should print "Computing Hessian" progress)
- Verify model is in eval mode during scoring
- Check that BayesRays output functions are properly injected

## File Locations

| Component | File | Lines |
|-----------|------|-------|
| Hessian Computer | `shadow_splat/bayesrays_utils.py` | 1-250 |
| View Selector | `shadow_splat/view_selector.py` | 155-234 |
| Factory Update | `shadow_splat/view_selector.py` | 237-292 |
| Config Extension | `shadow_splat/pipeline.py` | 114-144 |
| Pipeline Init | `shadow_splat/pipeline.py` | 147-204 |
| Pipeline Training Loop | `shadow_splat/pipeline.py` | 206-242 |
| Model Method | `shadow_splat/model.py` | 644-713 |

## Future Enhancements

- [ ] Adaptive Hessian computation (only update every N steps)
- [ ] Parallel candidate scoring (batched camera rendering)
- [ ] Uncertainty-weighted coverage (combine metrics)
- [ ] Spatial diversity constraints (don't select too-close views)
- [ ] Support for other NeRF architectures (InstantNGP, etc.)
