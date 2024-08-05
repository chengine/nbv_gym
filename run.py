#%%
import numpy as np 
import torch
import time
import open3d as o3d 
from splat.utils import *
from gsplat.cuda._torch_impl import _rasterize_to_pixels
from gsplat.cuda._wrapper import rasterize_to_indices_in_range

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config_path = Path('outputs/splato/splatfacto/2024-08-04_235629/config.yml')

splat = NeRF(config_path)

#%%
# Recovers dataset poses
poses = splat.get_poses()

# Renders splat from a pose. We call this to get the intermediate variables from the rasterization function.
tnow = time.time()
output = splat.render(poses[0])
print("Elapsed: ", time.time() - tnow)
# %%
# Parses render output for the intermediate rasterization variables
info = output["info"]
#%%
# color, alpha = _rasterize_to_pixels(
#     info["means2d"],
#     info["conics"],
#     torch.rand(info["conics"].shape, device=device),
#     info["opacities"],
#     info["width"],
#     info["height"],
#     info["tile_size"],
#     info["isect_offsets"],
#     info["flatten_ids"],
#     None, 
#     100
# )

tnow = time.time()
# pixel_ids/gaussian ids are sorted in order of depth of gaussians
gs_ids, pixel_ids, camera_ids = rasterize_to_indices_in_range(
    0,
    1000,
    torch.ones(1, info["height"], info["width"], device=device),
    info["means2d"],
    info["conics"],
    info["opacities"],
    info["width"],
    info["height"],
    info["tile_size"],
    info["isect_offsets"],
    info["flatten_ids"],
)
print('Elapsed: ', time.time() - tnow)
#%%

tnow = time.time()
_, counts = torch.unique(pixel_ids, return_counts=True)
print('Elapsed: ', time.time() - tnow)
print(counts.max())
print(counts.min())
# %%
