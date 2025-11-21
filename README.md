# Coverage Optimization for Camera View Selection

## Dependencies
### This repository uses `python=3.10`, `gsplat=1.5.3`, and [Nerfstudio](https://docs.nerf.studio/quickstart/installation.html) from source (as of 11/21/2025). Remember to overwrite the `gsplat` library included in [Nerfstudio] with the required version. Sometimes, you may also have to downgrade `numpy` to below 2.0.

## Installation Instructions

### 1. Clone this repo.
`redacted`

### 2. Install `shadow_splat` as a python package once you `cd` into the `shadow_splat` repository.
`python -m pip install -e .`

### 3. Register `shadow_splat` with Nerfstudio.
`ns-install-cli`

## Usage
### Now, you can run `shadow_splat` like other models in Nerfstudio using the `ns-train shadow_splat` command.
### For example:
```python 
ns-train shadow-splat --data <path to the data> \
    --output-dir <path to the output directory> \
    --pipeline.view-selector optics \
    --pipeline.optics-coverage-metric coverage
```
### The `--pipeline.view-selector` argument can be `optics`, `random`, or `all`. 
### The `--pipeline.optics-coverage-metric` can be `coverage`, `fig`, `view_fig`, `fig_diag`, or `view_fig_diag`. 

## Baselines
### To install the Fisher-RF baseline,
`pip install -e . --no-build-isolation` at the following repo:
`https://github.com/JiangWenPL/modified-diff-gaussian-rasterization-w-depth` (use `git clone --recursive` and gcc/g++ 11). Afterwards, you can run a similar command:
```python 
ns-train fisher-splat --data <path to the data> \
    --output-dir <path to the output directory> \
    --pipeline.view-selector optics \
    --pipeline.optics-coverage-metric fisher_info
```
