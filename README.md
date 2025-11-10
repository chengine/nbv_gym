# Shadow-Splat: Relighting and Learning Gaussians with Physics

## Installation Instructions

## The installation instructions assumes you have installed [Nerfstudio](https://docs.nerf.studio/quickstart/installation.html).

### 1. Clone this repo.
`git clone git@github.com:chengine/shadow_splat.git`

### 2. Install `shadow_splat` as a python package once you `cd` into the `shadow_splat` repository.
`python -m pip install -e .`

## 3. Register `shadow_splat` with Nerfstudio.
`ns-install -cli`

### Now, you can run `shadow_splat` like other models in Nerfstudio using the `ns-train shadow_splat` command.
### For example:
```python 
ns-train shadow-splat --data <path to the data> \
    --output-dir <path to the output directory> \
    --pipeline.model.camera-optimizer.mode SO3xR3 \
    --pipeline.model.rasterize-mode antialiased
```

