#%%
import numpy as np
import torch
import open3d as o3d
import time
import matplotlib
import imageio
import matplotlib.pyplot as plt

from voxel_traversal.traversal_utils import VoxelGrid, OptimizedVoxelGrid


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

def termination_fn(next_points, directions, next_indices, voxel_grid_binary, voxel_grid_values, termination_value_type=None):

    # Possible for indices to be out of bounds
    discretization = torch.tensor(voxel_grid_binary.shape, device=device)
    out_of_bounds = torch.any( torch.logical_or(next_indices < 0, next_indices - discretization.unsqueeze(0) > -1) , dim=-1)

    out_values = torch.zeros_like(next_points).to(torch.float32)

    # Mask out of bounds points
    next_points = next_points[~out_of_bounds]
    directions = directions[~out_of_bounds]
    next_indices = next_indices[~out_of_bounds]

    mask = voxel_grid_binary[next_indices[:, 0], next_indices[:, 1], next_indices[:, 2]]
    values = voxel_grid_values[next_indices[:, 0], next_indices[:, 1], next_indices[:, 2]]

    out_mask = ~out_of_bounds
    out_mask[~out_of_bounds] = mask

    out_values[~out_of_bounds] = values

    return out_mask, out_values

param_dict = {
    'discretizations': torch.tensor([100, 100, 100], device=device),
    'lower_bound': torch.tensor(bounding_box.get_min_bound(), device=device),
    'upper_bound': torch.tensor(bounding_box.get_max_bound(), device=device),
    'voxel_grid_values': None,
    'voxel_grid_binary': None,
    'termination_fn': termination_fn,
    'termination_value_type': None
}

colors = cmap( (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min()))[:, :3]
# colors = np.asarray(mesh.vertex_colors)

vgrid = VoxelGrid(param_dict, 3, device)
vgrid.populate_voxel_grid_from_points(torch.tensor(points, device=device), torch.tensor(colors, device=device, dtype=torch.float32))

vgrid_opt = OptimizedVoxelGrid(param_dict, 3, device)
vgrid_opt.populate_voxel_grid_from_points(torch.tensor(points, device=device), torch.tensor(colors, device=device, dtype=torch.float32))