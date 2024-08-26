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

def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

config_path = Path('outputs/splato/splatfacto/2024-08-04_235629/config.yml')

splat = NeRF(config_path, dataset_mode='train')

#%%
# Recovers dataset poses
poses = splat.get_poses()

for i in range(len(poses)):

    # Renders splat from a pose. We call this to get the intermediate variables from the rasterization function.
    tnow = time.time()
    output = splat.render(poses[i])
    print("Elapsed: ", time.time() - tnow)

    og_image = output['rgb'].cpu().numpy()

    # Parses render output for the intermediate rasterization variables
    info = output["info"]

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

    sigmas = torch.zeros_like(sigmas)

    alphas = torch.clamp_max(
        opacities[camera_ids, gs_ids] * torch.exp(-sigmas), 0.999
    )

    indices = camera_ids * image_height * image_width + pixel_ids
    total_pixels = C * image_height * image_width

    weights, trans = render_weight_from_alpha(
        alphas, ray_indices=indices, n_rays=total_pixels
    )

    sorted_list, sorted_ind = torch.sort(trans, descending=False)
    # img_plane_gs = gs_ids[sorted_ind[: image_height * image_width]]
    img_plane_gs = gs_ids[sorted_ind]

    def shadow_fn(input):
        new_weights = torch.zeros(input.shape[0], device=input.device)
        new_weights[img_plane_gs] = sorted_list**(1/2)

        if input.dim() == 3:
            new_color = input
        else:
            new_color = new_weights[..., None] * SH2RGB(input)
            new_color = RGB2SH(new_color)

        return new_color

    tnow = time.time()
    output = splat.render(poses[i], shadow_fn=shadow_fn)
    og_output = splat.render(poses[i], shadow_fn=None)

    pov_output = splat.render(poses[0], shadow_fn=shadow_fn)
    print("Elapsed: ", time.time() - tnow)

    new_image = output["rgb"].cpu().numpy()
    og_image = og_output["rgb"].cpu().numpy()
    pov_image = pov_output["rgb"].cpu().numpy()

    fig, ax = plt.subplots(3)
    ax[0].imshow(og_image)
    ax[1].imshow(new_image)
    ax[2].imshow(pov_image)
    # plt.show()

    plt.savefig(f'renders_pov/r_{i}.png')

#%%