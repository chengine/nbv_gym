#%%
import numpy as np 
import torch
import time
import open3d as o3d 
from splat.utils import *
from gsplat.cuda._torch_impl import _rasterize_to_pixels
from gsplat.cuda._wrapper import rasterize_to_indices_in_range
import matplotlib.pyplot as plt

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

og_image = output['rgb'].cpu().numpy()
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

means2d = info["means2d"]
conics = info["conics"]
opacities = info["opacities"]
image_width = info["width"]
image_height = info["height"]
tile_size = info["tile_size"]
isect_offsets = info["isect_offsets"]
flatten_ids = info["flatten_ids"]
#%%

# tnow = time.time()
# _, counts = torch.unique(pixel_ids, return_counts=True)
# print('Elapsed: ', time.time() - tnow)
# print(counts.max())
# print(counts.min())
# %%

# split ordered list

# tnow = time.time()
# for i in range(pixel_ids.max()):

#     intersect_ids = (pixel_ids == i)

#     g_ids = gs_ids[intersect_ids]
# print('Elapsed: ', time.time() - tnow)

from nerfacc import accumulate_along_rays, render_weight_from_alpha

C, N = means2d.shape[:2]

pixel_ids_x = pixel_ids % image_width
pixel_ids_y = pixel_ids // image_width
pixel_coords = torch.stack([pixel_ids_x, pixel_ids_y], dim=-1) + 0.5  # [M, 2]
deltas = pixel_coords - means2d[camera_ids, gs_ids]  # [M, 2]
c = conics[camera_ids, gs_ids]  # [M, 3]
sigmas = (
    0.5 * (c[:, 0] * deltas[:, 0] ** 2 + c[:, 2] * deltas[:, 1] ** 2)
    + c[:, 1] * deltas[:, 0] * deltas[:, 1]
)  # [M]
alphas = torch.clamp_max(
    opacities[camera_ids, gs_ids] * torch.exp(-sigmas), 0.999
)

indices = camera_ids * image_height * image_width + pixel_ids
total_pixels = C * image_height * image_width

weights, trans = render_weight_from_alpha(
    alphas, ray_indices=indices, n_rays=total_pixels
)

# src = weights[..., None]
# outputs = torch.zeros(
#             (total_pixels, src.shape[-1]), device=src.device, dtype=src.dtype
#             )
# outputs.index_add_(0, indices, src)
# %%

# # Change colors of splats
# colors = splat.pipeline.model.gauss_params["features_dc"]

# # colors.requires_grad_(False)

# new_color = colors[gs_ids]
# new_color = weights[..., None] * new_color

# colors[gs_ids] = new_color

def shadow_fn(input):
    new_weights = torch.ones(input.shape[0], device=input.device)
    new_weights[gs_ids] = trans

    if input.dim() == 3:
        # new_color = new_weights[..., None, None] * torch.sigmoid(input)
        # new_color = torch.logit(new_color)
        new_color = input
    else:
        new_color = new_weights[..., None] * torch.sigmoid(input)
        new_color = torch.logit(new_color)

        # print('weights', (torch.log(new_weights[..., None])+1).min())
        # print('inp max', input.max())
        # print('inp min', input.min())
        # print('out max', new_color.max())
        # print('out min', new_color.min())
    return new_color

# %%

for i in range(10):
    tnow = time.time()
    output = splat.render(poses[i], shadow_fn=shadow_fn)
    og_output = splat.render(poses[i], shadow_fn=None)
    print("Elapsed: ", time.time() - tnow)

    new_image = output["rgb"].cpu().numpy()
    og_image = og_output["rgb"].cpu().numpy()

    fig, ax = plt.subplots(2)
    ax[0].imshow(og_image)
    ax[1].imshow(new_image)
    plt.show()

#%%

renders = accumulate_along_rays(
        weights,
        splat.pipeline.model.colors,
        ray_indices=indices,
        n_rays=total_pixels,
    ).reshape(C, image_height, image_width, 1)

new_image = renders.cpu().numpy().squeeze()

fig, ax = plt.subplots(2)
ax[0].imshow(og_image)
ax[1].imshow(new_image)
plt.show()
# %%
