#%%
import numpy as np 
import torch
import time
import open3d as o3d 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import cv2
import os

from nerfstudio.cameras.cameras import Cameras, CameraType

from shadow_splat.splatloader import GaussianSplat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reset peak memory stats at the start
torch.cuda.reset_peak_memory_stats()

def look_at(location, target, up):
    z = (location - target)
    z /= torch.norm(z)
    x = torch.cross(up, z)
    x /= torch.norm(x)
    y = torch.cross(z, x)
    y /= torch.norm(y)

    R = torch.stack([x, y, z], dim=1)
    return R

# config_path = Path('outputs/moon_spiral_2_masked/shadow-splat/2024-10-25_133148/config.yml')
# config_path = Path('outputs/poster/shadow-splat/2024-10-25_153801/config.yml')
config_path = Path('outputs/rains_chair/shadow-splat/2024-11-08_143929/config.yml')

splat = GaussianSplat(config_path, dataset_mode='train', device=device, res_factor=0.5)

script_dir = os.path.dirname(os.path.abspath(__file__))
render_output_path = os.path.abspath(os.path.join(script_dir, '..', 'renders'))
print(f"Render output path: {render_output_path}")

#%%
# Recovers dataset poses
poses = splat.get_poses()
H, W, K = splat.get_camera_intrinsics()
pixel_coordinates = torch.meshgrid([torch.arange(H.item()), torch.arange(W.item())])
pixel_coordinates = torch.stack(pixel_coordinates, dim=-1).to(device)

# Render camera
render_camera = Cameras(
    camera_to_worlds=poses[0],
    fx=1650.0,
    fy=1650.0,
    cx=1000.0,
    cy=1000.0,
    width=2000,
    height=2000,
    camera_type=CameraType.PERSPECTIVE,
)
light = Cameras(
    camera_to_worlds=poses[0],
    fx=1650.0,
    fy=1650.0,
    cx=1000.0,
    cy=1000.0,
    width=2000,
    height=2000,
    # camera_type=CameraType.PERSPECTIVE,
    camera_type=CameraType.ORTHOPHOTO,
)

center = torch.tensor([H, W], device=device) / 2

# Arcing light source trajectory
N = 10  # number of light source poses
t = torch.linspace(0, np.pi, N)
light_source_positions = torch.stack([torch.cos(t), torch.zeros(N), torch.sin(t)], dim=-1).to(device)
# fig = go.Figure(data=[go.Scatter3d(x=light_source_poses[:, 0].cpu().numpy(), 
#                                    y=light_source_poses[:, 1].cpu().numpy(), 
#                                    z=light_source_poses[:, 2].cpu().numpy(), mode='markers')])
# fig.show()

# light_source_poses[:, :2] = light_source_poses[:, :2] * (1-torch.linspace(0, 1, N, device=device)[:, None])

for i in range(N):

    # Renders splat from a pose. We call this to get the intermediate variables from the rasterization function.
    tnow = time.time()
    torch.cuda.synchronize()

    # Update light source
    # Point the light source to the center of the scene
    light_source_pose = torch.eye(4, device=device)
    light_source_pose[:3, 3] = light_source_positions[i]
    light_source_pose[:3, :3] = look_at(light_source_positions[i], torch.tensor([0., 0., 0.], device=device), torch.tensor([0., 0., 1.], device=device)+1e-6*torch.randn(3, device=device))
    light_source_pose = poses[i]
    light.camera_to_worlds = light_source_pose[None,:3, ...]
    splat.update_light_source(light)
    torch.cuda.synchronize()
    print("Time to update light source: ", time.time() - tnow)

    tnow = time.time()
    torch.cuda.synchronize()
    pov_output = splat.render(poses[0], camera=render_camera)
    torch.cuda.synchronize()
    print("Time to render: ", time.time() - tnow)

    pov_image = pov_output["rgb"].cpu().numpy()

    cv2.imwrite(f'{render_output_path}/r_{i}.png', cv2.cvtColor(pov_image * 255, cv2.COLOR_BGR2RGB))

peak_memory = torch.cuda.max_memory_allocated()
print(f"Peak memory usage: {peak_memory / (1024 ** 2):.2f} MB")
