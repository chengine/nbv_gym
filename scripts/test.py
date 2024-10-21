#%%
import numpy as np 
import torch
import time
import open3d as o3d 
from splatloader import GaussianSplat
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def look_at(location, target, up):
    z = (location - target)
    z /= torch.norm(z)
    x = torch.cross(up, z)
    x /= torch.norm(x)
    y = torch.cross(z, x)
    y /= torch.norm(y)

    R = torch.stack([x, y, z], dim=1)
    return R

config_path = Path('outputs/moon_spiral_2/shadow-splat/2024-10-18_135934/config.yml')

splat = GaussianSplat(config_path, dataset_mode='train', device=device, res_factor=0.5)

#%%
# Recovers dataset poses
poses = splat.get_poses()
H, W, K = splat.get_camera_intrinsics()
pixel_coordinates = torch.meshgrid([torch.arange(H.item()), torch.arange(W.item())])
pixel_coordinates = torch.stack(pixel_coordinates, dim=-1).to(device)

center = torch.tensor([H, W], device=device) / 2

# Creates a spiral light source trajectory
t = 10*torch.linspace(0, 2*np.pi, len(poses))
t_z = torch.linspace(0.05, 1., len(poses))
light_source_poses = torch.stack([torch.cos(t), torch.sin(t), t_z], dim=-1).to(device)

light_source_poses[:, :2] = light_source_poses[:, :2] * (1- torch.linspace(0, 1, len(poses), device=device)[:, None])

for i in range(len(poses)):

    # Renders splat from a pose. We call this to get the intermediate variables from the rasterization function.
    tnow = time.time()
    torch.cuda.synchronize()

    # Update mask to oscillate
    diff = pixel_coordinates - center[None, None, :]
    dist = torch.norm(diff, dim=-1)

    # r is the light source conical radius
    r = 100*( np.abs(np.sin(i / 10)) + .1)
    mask = torch.exp(- dist / r)
    mask = mask / mask.max()

    mask_radius = (dist < 200).to(torch.float32)
    mask = mask * mask_radius

    # Implement additive color mixing
    #base_color = torch.tensor([0.9, 0.2, 0.1], device=device)       # closish to red
    base_color = torch.tensor([0.1, 0.2, 0.9], device=device)       # closish to red
    mask = mask[:, :, None] * base_color[None, None, :]

    # Update light source
    # Point the light source to the center of the scene
    light_source_pose = torch.eye(4, device=device)
    light_source_pose[:3, 3] = light_source_poses[i]
    light_source_pose[:3, :3] = look_at(light_source_poses[i], torch.tensor([0., 0., 0.], device=device), torch.tensor([0., 0., 1.], device=device)+1e-6*torch.randn(3, device=device))
    splat.update_light_source(light_source_pose, mask=mask)
    torch.cuda.synchronize()
    print("Time to update light source: ", time.time() - tnow)

    tnow = time.time()
    torch.cuda.synchronize()
    pov_output = splat.render(poses[0])
    torch.cuda.synchronize()
    print("Time to render: ", time.time() - tnow)

    pov_image = pov_output["rgb"].cpu().numpy()

    cv2.imwrite(f'renders/r_{i}.png', cv2.cvtColor(pov_image * 255, cv2.COLOR_BGR2RGB))

#%%
