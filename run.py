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
poses = splat.get_poses()

output = splat.render(poses[0])
# %%
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

gs_ids, pixel_ids, camera_ids = rasterize_to_indices_in_range(
    0,
    100000000,
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

#%%