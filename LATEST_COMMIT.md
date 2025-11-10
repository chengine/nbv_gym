# Latest Git Commit

**Commit Hash:** 049dce7da38c18cf3653984dfd555aa9eff7d9b3

**Branch:** light_view_selection

**Author:** madang6 <madang@stanford.edu>

**Author Date:** Sun Nov 9 20:07:20 2025 -0800

**Commit Date:** Sun Nov 9 20:07:23 2025 -0800

## Commit Message

Add error handling and cleanup to pipeline and Hessian computation

- Wrap view expansion and Hessian computation in try-except blocks
- Add proper cleanup after Hessian computation (zero_grad, cuda.empty_cache)
- Gracefully degrade if Hessian or view expansion fails
- This prevents NaN loss propagation from failed operations

## Files Changed

### Modified Files

1. `.gitignore` - Added `nerfstudio_source/`
2. `README.md` - Changed `ns-train shadow_splat` to `ns-train shadow-splat`
3. `pyproject.toml` - Added `bayes-rays` entry point
4. `shadow_splat/bayesrays_utils.py` - Added cleanup code (zero_grad, cuda.empty_cache)
5. `shadow_splat/pipeline.py` - Wrapped Hessian computation and view expansion in try-except blocks

## Key Changes

- Error handling for Hessian computation that fails
- Error handling for view expansion that fails
- Proper GPU memory cleanup after operations
- Graceful degradation when operations fail
