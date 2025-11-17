#%%
import torch
import time
# Example tensors
N = 6000000
A = torch.zeros(N, 3)  # Tensor A of size N x 3
B = torch.rand(N, 3)  # Tensor B of size N x 3

# Index tensor of size N (which column of B to add to A)
index = torch.randint(0, 3, (N,))

tnow = time.time()
torch.cuda.synchronize()
# Create a tensor to hold updates for scatter_add_
updates = torch.gather(B, 1, index.unsqueeze(1))  # Gather the selected column for each row based on `index`

# Expand index for scatter_add_
index_expanded = index.unsqueeze(1)

# Perform scatter_add_
A[:100].scatter_add_(1, index_expanded[:100], updates[:100])
torch.cuda.synchronize()
print('Elapsed:', time.time() - tnow)

print(A)

#%%
points = torch.rand(N, 3, device='cuda')
indices = torch.randint(0, 100, (N, 3), device='cuda')
directions = torch.rand(N, 3, device='cuda')
sign_directions = torch.randint(0, 2, (N, 3), device='cuda') * 2 - 1
step_xyz = torch.rand(3, device='cuda')

voxel_grid_centers = torch.rand(100, 100, 100, 3, device='cuda')

@torch.compile
def one_step_voxel_ray_intersection1(points, indices, directions, sign_directions, step_xyz):
    # NOTE: !!! IMPORTANT !!! directions is the vector from the termination point to the point defined by points

    B = len(points)
    arange = torch.arange(B, device='cuda')

    xyz_max = voxel_grid_centers[indices[:, 0], indices[:, 1], indices[:, 2]] + step_xyz/2

    t = (xyz_max - points) / directions

    min_t, min_idx = torch.min(t, dim=-1)       # N, idx tells us which voxel index dimension to move in (+- 1)

    # Update next voxel intersection point and voxel index
    points = points + min_t.unsqueeze(-1) * directions
    new_indices = indices.clone()
    new_indices[arange, min_idx] += sign_directions[arange, min_idx]

    return points, new_indices, min_t

@torch.compile
def one_step_voxel_ray_intersection2(frac_indices, indices, indices_directions, sign_directions, step_xyz):
    max_indices = indices + step_xyz / 2
    t = (max_indices - frac_indices) / indices_directions
    min_t, min_idx = torch.min(t, dim=-1)
    min_idx = min_idx.unsqueeze(-1)
    min_t = min_t.unsqueeze(-1)

    updates = torch.gather(sign_directions, 1, min_idx)  # Gather the selected column for each row based on `index`
    # Perform scatter_add_ adding the updates to the indices and frac indices. NOTE: In place operation
    indices.scatter_add_(1, min_idx, updates)
    
    # Update the frac indices
    frac_indices = frac_indices + min_t * indices_directions

    return frac_indices, indices, min_t

#%%
# tnow = time.time()
# points, indices, min_t = one_step_voxel_ray_intersection1(points, indices, directions, sign_directions, step_xyz)
# print('Elapsed:', time.time() - tnow)

tnow = time.time()
frac_indices, indices, min_t = one_step_voxel_ray_intersection2(points, indices, directions, sign_directions, step_xyz)
torch.cuda.synchronize()
print('Elapsed:', time.time() - tnow)
# %%
