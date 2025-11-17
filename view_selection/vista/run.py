#%%
import os
import numpy as np
import torch
import open3d as o3d
import time
import matplotlib
import matplotlib.pyplot as plt
import imageio

from voxel_traversal.traversal_utils import VoxelGrid
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

obj_path = 'data/stanford-bunny.obj'
# obj_path = 'bond/source/Bond_Test.glb'

mesh = o3d.io.read_triangle_mesh(obj_path)
mesh.compute_vertex_normals()
# mesh.scale(0.1, center=mesh.get_center())
mesh.rotate(mesh.get_rotation_matrix_from_xyz((np.pi/2, 0, 0)), center=np.zeros(3))
mesh.translate(-mesh.get_center())
points = np.asarray(mesh.vertices)

bounding_box = mesh.get_axis_aligned_bounding_box()
cmap = matplotlib.cm.get_cmap('turbo')

param_dict = {
    'discretizations': torch.tensor([100, 100, 100], device=device),
    'lower_bound': torch.tensor(bounding_box.get_min_bound(), device=device),
    'upper_bound': torch.tensor(bounding_box.get_max_bound(), device=device),
    'voxel_grid_values': None,
    'voxel_grid_binary': None,
}

colors = cmap( (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min()))[:, :3]
# colors = np.asarray(mesh.vertex_colors)

class OurVoxelGrid(VoxelGrid):
    def __init__(self, param_dict, ndim, device):
        super().__init__(param_dict, ndim, device)

    # NOTE: We are going to assume all indices are in bounds
    @torch.compile
    def termination_fn(self, indices, directions):
        mask = self.voxel_grid_binary[indices[:, 0], indices[:, 1], indices[:, 2]]
        values = self.voxel_grid_values[indices[:, 0], indices[:, 1], indices[:, 2]]

        return mask, values

vgrid = OurVoxelGrid(param_dict, 3, device)
vgrid.populate_voxel_grid_from_points(torch.tensor(points, device=device), torch.tensor(colors, device=device, dtype=torch.float32))

# scene = vgrid.create_mesh_from_points(torch.tensor(points, device=device), torch.tensor(colors, device=device, dtype=torch.float32))

# o3d.visualization.draw_geometries([scene])

def look_at(location, target, up):
    z = (location - target)
    z /= torch.norm(z)
    x = torch.cross(up, z, dim=-1)
    x /= torch.norm(x)
    y = torch.cross(z, x, dim=-1)
    y /= torch.norm(y)
    R = torch.stack([x, y, z], dim=-1)
    return R

K = torch.tensor([
    [100., 0., 75.],
    [0., 100., 75.],
    [0., 0., 1.]
], device=device)


tnows = []
num_cameras = [10, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
for i in num_cameras:
    N_cameras = i
    t = torch.linspace(0., 2*np.pi, N_cameras, device=device)
    far_clip = 1.

    depth_images = []
    directions = []
    origins = []

    c2w = torch.eye(4, device=device)[None].expand(N_cameras, -1, -1).clone()
    c2w[:, :3, 3] = torch.stack([0.25*torch.cos(t), 0.25*torch.sin(t), torch.zeros_like(t)], dim=-1)

    target = torch.zeros(3, device=device)[None].expand(N_cameras, -1)
    up = torch.tensor([0., 0., 1.], device=device)[None].expand(N_cameras, -1)
    c2w[:, :3, :3] = look_at(c2w[:, :3, 3], target, up)

    elapsed = []
    for j in range(1):
        tnow = time.time()
        torch.cuda.synchronize()
        image, depth, output = vgrid.camera_voxel_intersection(K, c2w, far_clip)
        torch.cuda.synchronize()
        elapsed.append(time.time() - tnow)
        print(f"Time taken ({i}): ", time.time() - tnow)
    tnows.append(np.array(elapsed)[1:].mean())

fig, ax = plt.subplots()
ax.plot(num_cameras[1:], tnows[1:])
ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='black', linestyle='dashed')
ax.set_xlabel('Number of cameras')
ax.set_ylabel('Time (s)')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim([10, 5000])
ax.set_ylim([np.array(tnows).min(), np.array(tnows).max()])

plt.savefig('time_taken.png')

# combined image
cmap = matplotlib.cm.get_cmap('turbo')

###NOTE: Uncomment to save images
# for i, (img, dep) in enumerate(zip(image, depth)):
#     dep = dep / far_clip
#     depth_mask = (dep == 0).squeeze()

#     #combined_image = torch.concatenate([img[..., 0], dep], axis=1)
#     #combined_image = cmap(combined_image.cpu().numpy())[..., :3]
#     depth = cmap(dep.cpu().numpy())[..., :3]
#     depth[depth_mask.cpu().numpy()] = 0
#     combined_image = np.concatenate([img.cpu().numpy(), depth], axis=1)

#     alpha = (combined_image > 0).any(axis=-1)
#     combined_image = np.concatenate([combined_image, alpha[..., None]], axis=-1)
#     imageio.imwrite(f'images/{i}.png', (combined_image * 255).astype(np.uint8))

# %%
