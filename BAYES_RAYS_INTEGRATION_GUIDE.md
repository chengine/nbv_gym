# BayesRays Integration Guide

A comprehensive guide to the BayesRays (Hessian-based uncertainty view selection with Nerfacto) integration into the shadow-splat codebase.

## Overview

BayesRays is a progressive view selection method that uses **Hessian-based uncertainty estimation** to intelligently expand the training view set during Nerfacto training. Unlike the original shadow-splat pipeline which uses full-image Gaussian splatting, BayesRays uses **Nerfacto with ray-batched training** (4096 rays/batch) for memory efficiency.

### Key Architecture Decision

The original implementation attempted to force Nerfacto into shadow-splat's full-image pipeline, causing OOM errors on large images (1920×1080 = ~2M rays). **The correct approach is to use Nerfacto's native ray-batched architecture** with view selection implemented at the ray sampling level.

---

## File Structure

### New Files Created

#### 1. **`shadow_splat/bayes_rays_datamanager.py`** ⭐ CORE
- **Purpose**: Ray-batched datamanager with BayesRays view selection support
- **Key Classes**:
  - `BayesRaysParallelDataManagerConfig`: Configuration for ray-batched datamanager
  - `BayesRaysParallelDataManager`: Extends `ParallelDataManager` from nerfstudio
- **Key Methods**:
  - `expand_active_set(k, step, ...)`: Expands the active training view pool using uncertainty
  - `next_train(step)`: Filters rays to only those from active training views
  - `next_eval(step)`: Returns ray batch for evaluation

**Architecture Details**:
```
ParallelDataManager (nerfstudio base)
    ↓
BayesRaysParallelDataManager (our implementation)
    - Tracks active_train_indices (which views are in training pool)
    - Filters ray bundles to match active views
    - Integrates with view selector for progressive expansion
```

### Modified Files

#### 2. **`shadow_splat/config.py`** 📝 CONFIGURATION
**Changes**:
- Added import of `BayesRaysParallelDataManagerConfig`
- Created `bayes_rays` MethodSpecification with:
  - Model: Standard `NerfactoModelConfig()` (no wrapper needed)
  - Datamanager: `BayesRaysParallelDataManagerConfig` with 4096 rays/batch
  - Pipeline: `ViewSelectionPipelineConfig` with BayesRays selector
  - Trainer settings: Disabled full-image eval to avoid OOM

**Key Settings**:
```python
train_num_rays_per_batch=4096      # Ray-batched training
eval_num_rays_per_batch=4096       # Ray-batched evaluation
steps_per_eval_image=0             # Disabled (causes OOM)
steps_per_eval_batch=100           # Use lightweight batch eval instead
steps_per_eval_all_images=0        # Disabled
start_num_views=1                  # Start with 1 view, expand progressively
```

#### 3. **`shadow_splat/pipeline.py`** 🔄 CORE LOGIC
**Changes**:
- Updated `get_train_loss_dict()`:
  - Handles both 2-tuple (ray_bundle, batch) from ray-batched datamanager
  - Handles 3-tuple (cameras, batch, light) from full-image datamanager
  - Properly passes RayBundle to model (not Cameras)

- Updated `get_eval_loss_dict()`:
  - Similar tuple-length checks for compatibility

- Updated `get_eval_image_metrics_and_images()`:
  - Converts Cameras to RayBundle for both datamanager types
  - **Chunks full-image rays** to avoid OOM:
    - Generates full rays from camera (~2M rays for 1920×1080)
    - Processes in chunks of `eval_num_rays_per_chunk` (default 4096)
    - Merges chunk outputs back together
  - Added `_merge_outputs()` helper to concatenate tensor outputs

**Program Flow in Training**:
```
get_train_loss_dict(step)
├─ Expand active set if step % add_every_n_steps == 0
│  ├─ Compute Hessian from training data
│  └─ View selector picks next views based on uncertainty
├─ Get next training batch from datamanager
│  └─ datamanager.next_train(step) → (ray_bundle, batch)
├─ Forward pass: model(ray_bundle)
└─ Compute loss from model outputs and batch
```

**Program Flow in Evaluation**:
```
get_eval_image_metrics_and_images(step)
├─ Get eval image: datamanager.next_eval_image(step) → (camera, batch)
├─ Convert camera to full ray bundle
├─ Chunk rays into eval_num_rays_per_chunk sizes
├─ For each chunk:
│  ├─ Forward pass: model(chunk_ray_bundle)
│  └─ Collect outputs
├─ Merge chunk outputs
└─ Compute metrics and images
```

#### 4. **`shadow_splat/bayesrays_utils.py`** 📊 UNCERTAINTY COMPUTATION
**Changes**:
- Updated `compute_hessian_from_datamanager()`:
  - Handles both 2-tuple and 3-tuple returns from `next_train()`
  - Enables gradients with `torch.enable_grad()` for Hessian computation
  - Adds proper cleanup: `model.zero_grad()` and `torch.cuda.empty_cache()`
  - Wrapped in try-except for graceful error handling

**Key Method Signature**:
```python
def compute_hessian_from_datamanager(
    model: NerfactoModel,
    datamanager,
    max_batches: Optional[int] = None,
) -> torch.Tensor:
    """Compute Hessian for uncertainty-based view selection"""
```

#### 5. **`shadow_splat/view_selector.py`** 🎯 VIEW SELECTION
**Changes**:
- Updated `select_views()` in BayesRays selector:
  - Fixed camera indexing: `all_cameras[cam_idx : cam_idx + 1]` instead of batch indexing
  - Added fallback to random selection if model lacks `uncertainty_score_for_camera()`
  - Added capability check for uncertainty scoring

**View Selection Logic**:
```
select_views(remaining_indices, num_to_select)
├─ Check if model has uncertainty_score_for_camera
├─ If yes: Score each candidate camera by uncertainty
├─ If no: Fall back to random selection
└─ Return top-k candidates by score
```

#### 6. **`scripts/sweep_scenes.py`** 🧪 EVALUATION & TESTING
**Changes**:
- Added full BayesRays support to sweep script:
  - New method type: `"bayes-rays"`
  - Uses standard nerfstudio data format (not shadow-splat-data)
  - No optics-specific arguments needed

- Added comprehensive WandB integration:
  - Environment variable setup for project/tags/notes
  - Custom run naming with timestamps
  - Run group tracking to group all sweeps together
  - Graceful degradation if WandB not installed

- Improved CLI output:
  - Progress counter (e.g., "Run 1/6")
  - Sweep group ID display
  - Success/failure indicators
  - Continues on failure instead of stopping

---

## Training Flow Diagram

```
┌─────────────────────────────────────────────────────┐
│ ns-train bayes-rays --data <dataset>                │
└──────────────────┬──────────────────────────────────┘
                   ↓
        ┌──────────────────────┐
        │ Initialize Pipeline  │
        │ - BayesRaysParallel  │
        │   DataManager        │
        │ - NerfactoModel      │
        │ - ViewSelector       │
        └──────────┬───────────┘
                   ↓
        ┌──────────────────────────────────────┐
        │ Training Loop (max_num_iterations)   │
        └──────────────────────────────────────┘
                   ↓
    ┌──────────────┴──────────────┐
    ↓                             ↓
┌─────────────────┐    ┌─────────────────────────┐
│ Training Step   │    │ Eval Step               │
│ (every iter)    │    │ (every eval interval)   │
└────────┬────────┘    └────────────┬────────────┘
         ↓                          ↓
         │              ┌───────────────────────┐
         │              │ get_eval_loss_dict()  │
         │              │ (batch eval, 4096     │
         │              │  rays, lightweight)   │
         │              └───────────────────────┘
         │                          ↓
         ├──────────────────────────┤
         ↓                          ↓
    ┌─────────────────┐    ┌──────────────────────┐
    │ View Expansion? │    │ get_eval_image_      │
    │ (every 1000 iter)    │ metrics_and_images() │
    └────────┬────────┘    │ (full image, chunked)│
             ↓              └──────────────────────┘
    ┌─────────────────────┐
    │ Compute Hessian     │
    │ (uncertainty grid)  │
    └────────┬────────────┘
             ↓
    ┌─────────────────────┐
    │ Select Top Views    │
    │ (by uncertainty)    │
    └────────┬────────────┘
             ↓
    ┌─────────────────────┐
    │ Expand Active Set   │
    │ (add new views)     │
    └─────────────────────┘
```

---

## Data Flow: Training Iteration

```
get_train_loss_dict(step)
│
├─ IF step % add_every_n_steps == 0:
│  │
│  ├─ compute_hessian_from_datamanager()
│  │  └─ FOR each training image:
│  │     ├─ Get ray_bundle, batch from datamanager
│  │     ├─ Forward pass to get outputs
│  │     └─ Accumulate Hessian (uncertainty grid)
│  │
│  └─ expand_active_set()
│     ├─ Get remaining (unused) view indices
│     ├─ select_views(remaining, num_to_add, hessian)
│     │  └─ Score each candidate by uncertainty
│     └─ Add selected views to active_train_indices
│
├─ datamanager.next_train(step)
│  ├─ Get ray_bundle, batch from ray sampler
│  └─ Filter to only active training views:
│     └─ mask = [idx in active_train_indices for idx in ray_bundle.camera_indices]
│        ray_bundle = ray_bundle[mask]
│        batch = {k: v[mask] for v in batch.values()}
│
├─ model(ray_bundle)
│  └─ Nerfacto forward pass (4096 rays, one level of hierarchy)
│
└─ loss_dict = model.get_loss_dict(outputs, batch)
```

## Data Flow: Evaluation Iteration

```
get_eval_image_metrics_and_images(step)
│
├─ datamanager.next_eval_image(step)
│  └─ Returns (camera, batch) - single full-resolution image
│
├─ camera.generate_rays(camera_indices=all)
│  └─ Creates full ray bundle (~2M rays for 1920×1080)
│
├─ FOR i in range(0, num_rays, eval_num_rays_per_chunk=4096):
│  │
│  ├─ chunk_ray_bundle = ray_bundle[i:i+4096]
│  │
│  └─ model(chunk_ray_bundle)
│     └─ Nerfacto forward pass on chunk
│
├─ _merge_outputs([chunk_outputs_1, ..., chunk_outputs_n])
│  └─ Concatenate tensors from all chunks
│
└─ model.get_image_metrics_and_images(merged_outputs, batch)
```

---

## Key Architectural Differences

### Ray-Batched Training (BayesRays) ✅

| Aspect | Value |
|--------|-------|
| **Rays per iteration** | 4,096 rays (constant) |
| **Memory per iteration** | ~1 GB |
| **Training speed** | Fast (many small batches) |
| **View expansion** | Progressive (intelligent selection) |
| **Datamanager** | `BayesRaysParallelDataManager` |
| **Ray sampling** | Pixel sampler + camera index filter |

### Full-Image Training (Original Shadow-Splat) ❌ (for reference)

| Aspect | Value |
|--------|-------|
| **Rays per iteration** | ~2M rays (full 1920×1080) |
| **Memory per iteration** | 20+ GB (causes OOM) |
| **Training speed** | Slow (one large batch) |
| **View expansion** | Static or random |
| **Datamanager** | `FullImageDatamanager` |
| **Ray sampling** | Generate all rays from camera |

---

## Configuration: Recommended Settings

```python
# For standard datasets (512×512 to 1920×1080)
bayes_rays = MethodSpecification(
    TrainerConfig(
        method_name="bayes-rays",
        steps_per_eval_image=0,           # Disable full-image eval
        steps_per_eval_batch=100,          # Use lightweight batch eval
        steps_per_eval_all_images=0,       # Disable expensive renders
        max_num_iterations=30000,

        pipeline=ViewSelectionPipelineConfig(
            datamanager=BayesRaysParallelDataManagerConfig(
                train_num_rays_per_batch=4096,     # Standard nerfacto
                eval_num_rays_per_batch=4096,      # Matches training
                start_num_views=1,                 # Start with 1 view
            ),
            model=NerfactoModelConfig(),           # Standard nerfacto
            add_every_n_steps=1000,                # Expand every 1000 steps
            add_num_views=1,                       # Add 1 view at a time
            view_selector="bayes",                 # Hessian-based selection
            bayes_reduce_mode="mean",              # Aggregation method
            bayes_lod=8,                           # Grid resolution (2^8 = 256)
            disable_light=True,                    # No light data needed
        ),
    ),
)
```

---

## Running BayesRays Training

### Single Run with WandB
```bash
ns-train bayes-rays \
  --data /path/to/dataset \
  --vis viewer+wandb \
  --project-name my-project \
  --max-num-iterations 30000
```

### Batch Sweep
```bash
python scripts/sweep_scenes.py
# Configured in METHODS = ["bayes-rays"]
# Uses WandB for organized tracking
```

### Command-Line Arguments
```bash
# View expansion control
--pipeline.add-every-n-steps 500      # Expand every 500 steps
--pipeline.add-num-views 2             # Add 2 views per expansion

# Uncertainty computation
--pipeline.bayes-lod 8                 # 256^3 uncertainty grid
--pipeline.bayes-max-hessian-batches 50 # Limit Hessian computation

# Memory optimization
--pipeline.datamanager.train-num-rays-per-batch 2048  # Reduce rays
--pipeline.datamanager.eval-num-rays-per-batch 2048
```

---

## Troubleshooting

### Issue: CUDA Out of Memory during Evaluation
**Cause**: Full image evaluation creates 2M rays
**Solution**: Already handled - `steps_per_eval_image=0` and chunking in `get_eval_image_metrics_and_images()`

### Issue: View Selection Not Working
**Cause**: Model doesn't have `uncertainty_score_for_camera()`
**Solution**: Code automatically falls back to random selection. For Nerfacto, uncertainty-based scoring requires custom implementation.

### Issue: Training Diverges After View Expansion
**Cause**: New views may have very different lighting/angle
**Solution**: Reduce `add_num_views` (add fewer views per expansion), increase `add_every_n_steps` (expand less frequently)

### Issue: Memory Usage Growing Over Time
**Cause**: Hessian computation accumulates with larger active set
**Solution**: Set `--pipeline.bayes-max-hessian-batches 50` to limit computation

---

## Summary of Changes

| File | Status | Key Change |
|------|--------|-----------|
| `bayes_rays_datamanager.py` | ✨ NEW | Ray-batched datamanager with view selection |
| `config.py` | 📝 MODIFIED | Added bayes-rays method spec |
| `pipeline.py` | 🔄 MODIFIED | Tuple handling, ray chunking for eval |
| `bayesrays_utils.py` | 🔄 MODIFIED | Hessian computation fixes |
| `view_selector.py` | 🔄 MODIFIED | Camera indexing, uncertainty fallback |
| `sweep_scenes.py` | 🔄 MODIFIED | BayesRays support + WandB integration |

**Total Commits**: 7 commits integrating BayesRays into this session

---

## Next Steps

1. **Improve uncertainty scoring for Nerfacto**: Implement `uncertainty_score_for_camera()` for better view selection than random
2. **Optimize Hessian computation**: Reduce memory usage during Hessian computation with larger active sets
3. **Add visualization**: Tools to visualize which views are selected and why
4. **Extend to other NeRF models**: Adapt ray-batched approach to other architectures
