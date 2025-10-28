import numpy as np
import torch
import open3d as o3d
import time
from tensordict import TensorDict
from .camera_utils import get_rays, get_rays_batch

# All these functions are minor modifications to each other to make them fast for their use case
# They all hinge on the core function: one-step voxel ray intersection

# @torch.compile
# def one_step_voxel_ray_intersection(frac_indices, indices, indices_directions, sign_directions, half_sign_directions):

#     steps = (indices + half_sign_directions - frac_indices) / (indices_directions)

#     # Replace infs with a large number
#     #steps[torch.isinf(steps)] = 1e3

#     min_t, min_idx = torch.min(steps, dim=-1)
#     min_t_unsq = min_t.unsqueeze(-1)
#     min_idx = min_idx.unsqueeze(-1)

#     # Perform scatter_add_ adding the updates to the indices and frac indices. NOTE: In place operation
#     indices.scatter_add_(1, min_idx, torch.gather(sign_directions, 1, min_idx))
    
#     # Update the frac indices
#     frac_indices.addcmul_(min_t_unsq, indices_directions)  # In-place update

#     short_factor = (1. - min_t).unsqueeze(-1)
#     indices_directions.mul_(short_factor)

#     return frac_indices, indices, indices_directions, min_t, min_idx

# class VoxelGrid:
#     def __init__(self, param_dict, ndim, device):
#         self.discretizations = param_dict['discretizations']
#         self.ndim = ndim

#         if ndim is None:
#             self.ndim = len(self.discretizations)

#         assert self.ndim == 2 or self.ndim == 3, "Only 2D and 3D voxel grids are supported"
#         assert len(self.discretizations) == self.ndim, "Discretizations must be of length equal to the number of dimensions"

#         ### IMPORTANT !!! ###
#         # Lower bound must be strictly less than upper bound #
#         self.lower_bound = param_dict['lower_bound']
#         self.upper_bound = param_dict['upper_bound']

#         assert torch.all(self.lower_bound - self.upper_bound <= 0.), "Lower bound must be less than upper bound"
#         assert len(self.lower_bound) == self.ndim, "Lower bound must be of length equal to the number of dimensions"
#         assert len(self.upper_bound) == self.ndim, "Upper bound must be of length equal to the number of dimensions"

#         ### IMPORTANT !!! ###

#         self.voxel_grid_binary = param_dict['voxel_grid_binary']
#         self.voxel_grid_values = param_dict['voxel_grid_values']
#         self.device = device
#         self.val_ndim = param_dict['val_ndim']

#         self.initialize_voxel_grid()

#         self.discretizations_unsq = self.discretizations.unsqueeze(0)

#     #TODO: May want to take in some pre-made voxel grid values
#     def initialize_voxel_grid(self):
#         if self.voxel_grid_binary is None:
#             self.voxel_grid_binary = torch.zeros(tuple(self.discretizations), device=self.device, dtype=torch.bool)
#             self.voxel_grid_values = torch.zeros(tuple(self.discretizations) + (self.val_ndim,), device=self.device, dtype=torch.float32)
        
#         self.cell_sizes = (self.upper_bound - self.lower_bound) / self.discretizations

#         grid_pts = torch.meshgrid([torch.linspace(self.lower_bound[i], self.upper_bound[i], self.discretizations[i]+1, device=self.device) for i in range(self.ndim)])
#         self.grid_vertices = torch.stack(grid_pts, dim=-1)      # n1 x n2 x n3 x 3

#         if self.ndim == 2:
#             self.voxel_grid_centers = 0.5 * (self.grid_vertices[:-1, :-1] + self.grid_vertices[1:, 1:])
#             self.lower_grid_center = self.voxel_grid_centers[0, 0]

#         elif self.ndim == 3:
#             self.voxel_grid_centers = 0.5 * (self.grid_vertices[:-1, :-1, :-1] + self.grid_vertices[1:, 1:, 1:])
#             self.lower_grid_center = self.voxel_grid_centers[0, 0, 0]
        
#     def compute_voxel_index(self, points, debug=False):
#         # IMPORTANT !!! #
#         # This function does NOT handle points outside the voxel grid. It is the user's responsibility to ensure that the points are within the voxel grid, because
#         # there are many ways in which one could project outside points into the grid. 

#         ### IMPORTANT !!! ###
#         # We are going with the "round" convention for indexing.
#         voxel_index = torch.round( (points - self.lower_grid_center) / self.cell_sizes ).to(torch.int32)

#         # Sanity check
#         is_in_bounds = []
#         for i in range(self.ndim):
#             # if debug:
#             #     assert torch.all(voxel_index[:, i] >= 0) and torch.all(voxel_index[:, i] < self.discretizations[i]), "Points must be within the voxel grid"
     
#             is_in_bounds.append( torch.logical_and( (voxel_index[:, i] >= 0), (voxel_index[:, i] < self.discretizations[i])) )

#         is_in_bounds = torch.all( torch.stack(is_in_bounds, dim=-1), dim = -1 )

#         return voxel_index, is_in_bounds
    
#     def project_points_into_voxel_grid(self, points, directions):
#         # Reference: https://tavianator.com/2022/ray_box_boundary.html#:~:text=Geometry&text=The%20slab%20method%20tests%20for,intersects%20the%20box%2C%20if%20any.&text=t%20%3D%20x%20%E2%88%92%20x%200%20x%20d%20.
#         ### IMPORTANT !!! ###
#         # The directions you pass DO NOT need to be normalized. They will be treated as line segments!

#         # If the point is in the voxel grid, we don't do anything.
#         # However, if the point is outside the voxel grid, we want to use 
#         # directions such that the voxel we return is the first voxel that the ray intersects.
#         voxel_index, in_bounds = self.compute_voxel_index(points)

#         # indices are within the voxel grid
#         in_bound_voxel_index = voxel_index[in_bounds]

#         # indices are outside the voxel grid
#         out_bound_directions = directions[~in_bounds] + torch.randn_like(directions[~in_bounds]) * 1e-6       # Add some noise to reduce degeneracy
#         out_bound_points = points[~in_bounds]

#         # Of these out of bound points, there are rays that ultimately intersect the voxel, and others that miss the voxel grid entirely.
#         # The general test:

#         t1 = (self.lower_bound[None] - out_bound_points) / out_bound_directions     # N x 3
#         t2 = (self.upper_bound[None] - out_bound_points) / out_bound_directions     # N x 3

#         t = torch.stack([t1, t2], dim=-1)       # N x 3 x 2

#         # TODO: We can skip this first min/max step by looking at the signs of the directions directly!
#         tmin = torch.min(t, dim=-1).values      # N x 3
#         tmax = torch.max(t, dim=-1).values      # N x 3

#         tmin = torch.max(tmin, dim=-1).values     # N
#         tmax = torch.min(tmax, dim=-1).values     # N

#         # If tmin > tmax for any dimension, the line misses the voxel grid
#         misses = (tmin - tmax) > 0.     # N

#         # TODO: CHECK THIS PROCEDURE!!!

#         # Of the rays that we think intersect the voxel grid, we need to make sure that the line segment actually intersects the voxel grid (t < 1)
#         # AND we need to make sure the t's are also positive!!!
#         short = tmin > 1.     # N
#         reversed = tmin < 0.  # N

#         # # These points will NEVER intersect the voxel grid
#         misses = torch.logical_or(misses, short)
#         misses = torch.logical_or(misses, reversed)

#         # These are the points that do intersect the voxel grid but are outside the voxel grid
#         out_bound_intersect_dirs = out_bound_directions[~misses]
#         out_bound_intersect_pts = out_bound_points[~misses]

#         # and the progress along the line in order to hit the voxel grid
#         t_progress = tmin[~misses]
#         remaining_progress = 1. - t_progress

#         # and the point where the ray intersects the voxel grid
#         intersect_pts = out_bound_intersect_pts + out_bound_intersect_dirs * (t_progress[:, None])     # Add some noise to avoid numerical instability
#         intersect_directions = out_bound_intersect_dirs * remaining_progress[:, None]
#         intersect_voxel_index, _ = self.compute_voxel_index(intersect_pts)

#         in_bound_voxel_index = torch.clamp(in_bound_voxel_index, torch.zeros_like(self.discretizations).unsqueeze(0), self.discretizations.unsqueeze(0) - 1)
#         intersect_voxel_index = torch.clamp(intersect_voxel_index, torch.zeros_like(self.discretizations).unsqueeze(0), self.discretizations.unsqueeze(0) - 1)

#         # This mask never intersects
#         not_intersecting = torch.zeros_like(in_bounds, dtype=torch.bool)
#         not_intersecting[~in_bounds] = misses

#         output = {
#             'in_bounds': in_bounds,
#             'not_intersecting': not_intersecting,       # mask of not intersecting data points
#             'intersecting_out_bounds': torch.logical_and(~not_intersecting, ~in_bounds), # mask of intersecting data points  (not clearly in bounds but intersecting gives you the points outside the grid that intersect the grid)
#             'in_bounds_voxel_index': in_bound_voxel_index,
#             'in_bounds_points': points[in_bounds],               # directions remain unchanged since we are already in the voxel grid
#             'in_bounds_directions': directions[in_bounds],
#             'out_bounds_points': out_bound_intersect_pts,           
#             'out_bounds_directions': out_bound_intersect_dirs,
#             'out_bounds_intersect_points': intersect_pts,           # project points outside of voxel grid that do intersect the grid into the edge of the grid
#             'out_bounds_intersect_direction': intersect_directions, # remaining vector after intersecting the grid
#             'out_bounds_intersect_voxel_index': intersect_voxel_index,  # voxel index of the intersecting point with the voxel grid
#         }
     
#         # Returns the voxel index, the point, the truncated direction for points in the grid, and the indices of points that do intersect the grid
#         # Also returns the indices of points that don't ever intersect the voxel
#         return output

#     @torch.compile
#     def compute_voxel_ray_intersection(self, points, directions):
#         # We assume directions is such that points + t * directions, where t = 1, is the termination point

#         # Project the points into the voxel grid
#         output = self.project_points_into_voxel_grid(points, directions)

#         ray_indices = torch.arange(len(points), device=self.device)
#         not_intersecting_ray_indices = ray_indices[output['not_intersecting']]
#         intersecting_ray_indices = ray_indices[~output['not_intersecting']]

#         # Segment out the rays that intersect the voxel grid
#         in_bounds_points = output['in_bounds_points']
#         in_bounds_directions = output['in_bounds_directions']
#         in_bounds_voxel_index = output['in_bounds_voxel_index']

#         # These are the Out-of-bounds rays, truncated so that they intersect the voxel grid
#         out_bounds_intersect_points = output['out_bounds_intersect_points']
#         out_bounds_intersect_directions = output['out_bounds_intersect_direction']
#         out_bounds_intersect_voxel_index = output['out_bounds_intersect_voxel_index']

#         # Concatenate the in-bounds and out-bounds intersecting rays
#         points = torch.cat([in_bounds_points, out_bounds_intersect_points], dim=0)
#         directions = torch.cat([in_bounds_directions, out_bounds_intersect_directions], dim=0)
#         voxel_index = torch.cat([in_bounds_voxel_index, out_bounds_intersect_voxel_index], dim=0)

#         frac_indices = ( (points - self.lower_grid_center) / self.cell_sizes ).to(torch.float16)

#         # NOTE: It shouldn't be possible to get -1 or self.discretizations[None] in the voxel index
#         voxel_index = torch.clamp(voxel_index, torch.zeros_like(self.discretizations).unsqueeze(0), self.discretizations.unsqueeze(0) - 1).to(torch.int32)

#         # exiting_ray_indices = []
#         # out_of_length_ray_indices = []

#         terminated_voxel_index = torch.zeros((len(intersecting_ray_indices), 3), device=self.device, dtype=torch.int32)
#         terminated_ray_index = torch.zeros(len(intersecting_ray_indices), device=self.device, dtype=torch.int32)
#         terminated_ray_directions = torch.zeros((len(intersecting_ray_indices), 3), device=self.device, dtype=torch.float32)

#         # NOTE: This could be computed just once outside of this function!
#         indices_directions = (directions / self.cell_sizes.unsqueeze(0)).to(torch.float16)
#         sign_directions = torch.sign(directions + torch.randn(directions.shape, device=self.device)).to(torch.int32)
#         half_sign_directions = (sign_directions / 2).to(torch.float16)

#         counter = 0
#         num_terminated = 0
#         remaining_rays = len(intersecting_ray_indices)

#         traversed_voxels = torch.zeros_like(self.voxel_grid_binary, dtype=torch.bool)

#         # DATA TENSOR
#         data = TensorDict({
#             'intersecting_ray_indices': intersecting_ray_indices,
#             'frac_indices': frac_indices,
#             'voxel_index': voxel_index,
#             'indices_directions': indices_directions,
#             'sign_directions': sign_directions,
#             'half_sign_directions': half_sign_directions
#         }, batch_size=(len(intersecting_ray_indices),))

#         while remaining_rays > 0:

#             # Break out components of data tensor. NOTE: In-place!
#             intersecting_ray_indices_ = data['intersecting_ray_indices']
#             frac_indices_ = data['frac_indices']
#             voxel_index_ = data['voxel_index']
#             indices_directions_ = data['indices_directions']
#             sign_directions_ = data['sign_directions']
#             half_sign_directions_ = data['half_sign_directions']

#             # Begin the incremental traversal procedure
#             terminated = self.is_terminated(voxel_index_)

#             traversed_voxels[voxel_index_[:, 0], voxel_index_[:, 1], voxel_index_[:, 2]] = True

#             # Store the terminated rays
#             num_terminated_now = num_terminated + torch.sum(terminated)
#             terminated_ray_index[ num_terminated:num_terminated_now ] = intersecting_ray_indices_[terminated]
#             terminated_voxel_index[ num_terminated:num_terminated_now ] = voxel_index_[terminated]
#             terminated_ray_directions[ num_terminated:num_terminated_now ] = indices_directions_[terminated]
#             num_terminated = num_terminated_now

#             #print(voxel_index)
#             frac_indices_, voxel_index_, indices_directions_, min_t, min_idx = one_step_voxel_ray_intersection(frac_indices_, voxel_index_, indices_directions_, sign_directions_, half_sign_directions_)
      
#             # If the min_t is greater than 1, we have reached the end of the ray
#             out_of_length = (min_t >= 1.)

#             # NOTE: Indices cannot exceed 256!
#             voxel_index_ = voxel_index_.to(torch.uint8)
 
#             # We also want to terminate if the ray leaves the voxel grid (NOTE: !!! IMPORTANT !!! This is true only for a single voxel grid!!! We
#             # just need to truncate and store the ray if we have multiple voxel grids)
#             # out_of_bounds = (voxel_index_ - self.discretizations_unsq <= -1).all(dim=1)     #N
#             out_of_bounds = ( torch.gather(voxel_index_, 1, min_idx) - self.discretizations[min_idx] <= -1 ).squeeze()

#             not_keep = out_of_length | ~out_of_bounds | terminated

#             # NOTE! It is possible to have rays that are both out of bounds and out of length! We will treat them as out of bounds.
#             # exiting_ray_indices.append(intersecting_ray_indices[out_of_bounds])
#             # out_of_length_ray_indices.append(intersecting_ray_indices[out_of_length & ~out_of_bounds])

#             # If the min_t is less than 1, we have not reached the end of the ray
#             keep = ~not_keep

#             # Cache indexing results for reuse
#             keep_indices = keep.nonzero(as_tuple=True)[0]  # Avoid redundant computation

#             data = data[keep_indices]
  
#             remaining_rays = len(keep_indices)

#             if counter % 50 == 0:
#                 print('Ray Tracing Step: ', counter)
#                 print("Number of rays remaining: ", remaining_rays)

#             counter += 1

#         terminated_ray_index = terminated_ray_index[:num_terminated]
#         terminated_voxel_index = terminated_voxel_index[:num_terminated]
#         terminated_ray_directions = terminated_ray_directions[:num_terminated]

#         terminated_voxel_values = self.compute_termination_values(terminated_voxel_index, terminated_ray_directions)
        
#         # We want to store
#         output = {
#             # For rays that terminate due to the termination fn, we store the voxel index, 
#             # the ray index, and the value of the grid at termination
#             'terminated_voxel_index': terminated_voxel_index.to(torch.int32),
#             'terminated_ray_index': terminated_ray_index.to(torch.int32),
#             'terminated_voxel_values': terminated_voxel_values,
#             'terminated_ray_directions': terminated_ray_directions,
#             'traversed_voxels': traversed_voxels,   # N x N x N voxel grid of booleans

#             # For rays that (1) don't hit the voxel grid at all, (2) exit the voxel grid, or (3) reach the end of the ray length
#             # are stored by their ray index. Although functionally, these three categories are treated the same (i.e. they don't render to anything),
#             # the designations could be used for downstream tasks.
#             # 'not_intersecting_ray_index': not_intersecting_ray_indices,
#             # 'exiting_ray_index': torch.cat(exiting_ray_indices, dim=0) if len(exiting_ray_indices) > 0 else None,
#             # 'out_of_length_ray_index': torch.cat(out_of_length_ray_indices, dim=0) if len(out_of_length_ray_indices) > 0 else None,
#         }
#         #print('Time taken for intersections: ', tdiff)
#         return output
    
#     def is_terminated(self, indices):
#         return self.voxel_grid_binary[indices[:, 0], indices[:, 1], indices[:, 2]]
    
#     def compute_termination_values(self, indices, directions):
#         #terminated_values = torch.zeros((len(indices), self.val_ndim + 1), device=self.device)
#         voxel_values = self.voxel_grid_values[ indices[:, 0], indices[:, 1], indices[:, 2] ]
#         #terminated_values[:, :voxel_values.shape[-1]] = voxel_values
#         return voxel_values

#     # Renders the voxel grid, given the voxel grid, camera extrinsics, and intrinsics.
#     # Makes use of the nerfacc library for fast rendering.
#     # We can play with some of the approximations, or make it exact if need be.

#     # TODO: Might implement a near clip by just shifting the camera origin along some near_clip distance along ray direction.
#     def camera_voxel_intersection(self, K, c2w, far_clip):
#         W, H = int(K[0, 2] * 2), int(K[1, 2] * 2)

#         if c2w.dim() == 2:
#             origins, directions = get_rays(H, W, K, c2w, self.device)

#         else:
#             B = c2w.shape[0]    # batch number of poses
#             origins, directions = get_rays_batch(H, W, K, c2w, self.device)
#         # Flatten origins and directions
#         origins = origins.reshape(-1, 3)
#         directions = directions.reshape(-1, 3)

#         #normalize directions
#         directions = directions / torch.norm(directions, dim=-1, keepdim=True)

#         # Extend directions to far_clip
#         directions = directions * far_clip

#         # Compute the voxel intersections
#         output = self.compute_voxel_ray_intersection(origins, directions)

#         n_channels = output['terminated_voxel_values'].shape[-1]

#         if c2w.dim() == 2:
#             # render one image

#             # Create image and depth
#             image = torch.zeros((H, W, n_channels), device=self.device)      # N for the color channels, 1 for an uncertainty metric if available (otherwise 1.)
#             depth = torch.zeros((H, W), device=self.device)

#             # Compute the depth and image
#             image = image.view(H*W, -1)
#             image[output['terminated_ray_index']] = output['terminated_voxel_values'].to(torch.float32)
#             image = image.view(H, W, -1)

#             depth = depth.view(-1)
#             depth[output['terminated_ray_index']] = torch.linalg.norm(self.voxel_grid_centers[output['terminated_voxel_index'][:, 0], 
#                                                                             output['terminated_voxel_index'][:, 1], 
#                                                                             output['terminated_voxel_index'][:, 2]] - c2w[:3, -1].unsqueeze(0), dim=-1)
#             depth = depth.view(H, W, 1) 

#         else:
#             # render batch of images
#             # Create image and depth
#             image = torch.zeros((B, H, W, n_channels), device=self.device)
#             depth = torch.zeros((B, H, W), device=self.device)

#             # Compute the depth and image
#             image = image.reshape(B*H*W, -1)
#             image[output['terminated_ray_index']] = output['terminated_voxel_values'].to(torch.float32)
#             image = image.reshape(B, H, W, -1)

#             depth = depth.reshape(-1)

#             depth[output['terminated_ray_index']] = torch.linalg.norm(self.voxel_grid_centers[output['terminated_voxel_index'][:, 0], 
#                                                                             output['terminated_voxel_index'][:, 1], 
#                                                                             output['terminated_voxel_index'][:, 2]] - origins[output['terminated_ray_index']], dim=-1)
#             depth = depth.reshape(B, H, W)

#         output['rays_o'] = origins
#         output['rays_d'] = directions

#         return image, depth, output

#     def populate_voxel_grid_from_points(self, points, values):
#         voxel_index, is_in_grid = self.compute_voxel_index(points)

#         # Only store the points that are in the grid
#         voxel_index = voxel_index[is_in_grid]

#         # Populate the voxel grid with the values
#         # NOTE: THIS WILL OVERWRITE THE VALUES! WE MIGHT WANT TO DO A SCATTER ADD OR SOMETHING MORE SOPHISTICATED.
#         self.voxel_grid_values[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]] = values[is_in_grid].to(torch.float32)
#         self.voxel_grid_binary[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]] = True
    
#     def create_mesh_from_points(self, points, colors=None):
#         # colors needs to be thhe same length as points

#         # Create a mesh from the voxel grid
#         voxel_index, in_bounds = self.compute_voxel_index(points)

#         #assert torch.all(in_bounds), "All points must be within the voxel grid"

#         colors = colors[in_bounds]
#         voxel_index = voxel_index[in_bounds]

#         voxel_index_flatten = torch.tensor(np.ravel_multi_index(voxel_index.T.cpu().numpy(), tuple(self.discretizations)), device=self.device)

#         # Store the number of counts for each voxel
#         voxel_counts = torch.zeros(tuple(self.discretizations), device=self.device, dtype=voxel_index_flatten.dtype).flatten()
#         voxel_counts.scatter_add_(0, voxel_index_flatten, torch.ones_like(voxel_index_flatten))

#         voxel_counts = voxel_counts.reshape(tuple(self.discretizations))

#         # Store the colors for each voxel
#         voxel_colors = torch.zeros(tuple(self.discretizations) + (3,), device=self.device, dtype=colors.dtype).reshape(-1, 3)
#         voxel_colors.scatter_add_(0, voxel_index_flatten.unsqueeze(1).expand(-1, 3), colors)

#         voxel_colors = voxel_colors.reshape(tuple(self.discretizations) + (3,))

#         # Create the mesh
#         mask = voxel_counts > 0

#         grid_centers = self.voxel_grid_centers[mask]
#         grid_centers = grid_centers.view(-1, 3).cpu().numpy()

#         grid_colors = voxel_colors[mask]
#         grid_colors = grid_colors.view(-1, 3).cpu().numpy()

#         grid_colors = grid_colors / voxel_counts[mask].unsqueeze(1).cpu().numpy()

#         scene = o3d.geometry.TriangleMesh()
#         for cell_color, cell_center in zip(grid_colors, grid_centers):
#             box = o3d.geometry.TriangleMesh.create_box(width=self.cell_sizes[0].cpu().numpy(), 
#                                                         height=self.cell_sizes[1].cpu().numpy(), 
#                                                         depth=self.cell_sizes[2].cpu().numpy())
#             box = box.translate(cell_center, relative=False)
#             box.paint_uniform_color(cell_color)
#             scene += box

#         return scene

#@torch.compile
def one_step_voxel_ray_intersection(frac_indices, indices, indices_directions, sign_directions, half_sign_directions):

    step = (indices + half_sign_directions - frac_indices) / (indices_directions)

    step[step.isinf()] = 1e3

    min_t, min_idx = torch.min(step, dim=-1)
    min_t_unsq = min_t.unsqueeze(-1)
    min_idx = min_idx.unsqueeze(-1)

    # Perform scatter_add_ adding the updates to the indices and frac indices. NOTE: In place operation
    indices.scatter_add_(1, min_idx, torch.gather(sign_directions, 1, min_idx))
    
    # Update the frac indices
    frac_indices.addcmul_(min_t_unsq, indices_directions)  # In-place update

    short_factor = (1. - min_t).unsqueeze(-1)
    indices_directions.mul_(short_factor)

    return frac_indices, indices, indices_directions, min_t, min_idx

class VoxelGrid:
    def __init__(self, param_dict, ndim, device):
        self.discretizations = param_dict['discretizations']
        self.ndim = ndim

        if ndim is None:
            self.ndim = len(self.discretizations)

        assert self.ndim == 2 or self.ndim == 3, "Only 2D and 3D voxel grids are supported"
        assert len(self.discretizations) == self.ndim, "Discretizations must be of length equal to the number of dimensions"

        ### IMPORTANT !!! ###
        # Lower bound must be strictly less than upper bound #
        self.lower_bound = param_dict['lower_bound']
        self.upper_bound = param_dict['upper_bound']

        assert torch.all(self.lower_bound - self.upper_bound <= 0.), "Lower bound must be less than upper bound"
        assert len(self.lower_bound) == self.ndim, "Lower bound must be of length equal to the number of dimensions"
        assert len(self.upper_bound) == self.ndim, "Upper bound must be of length equal to the number of dimensions"

        ### IMPORTANT !!! ###

        self.voxel_grid_binary = param_dict['voxel_grid_binary']
        self.voxel_grid_values = param_dict['voxel_grid_values']
        self.device = device
        self.val_ndim = param_dict['val_ndim']

        self.initialize_voxel_grid()

        self.discretizations_unsq = self.discretizations.unsqueeze(0)

    #TODO: May want to take in some pre-made voxel grid values
    def initialize_voxel_grid(self):
        if self.voxel_grid_binary is None:
            self.voxel_grid_binary = torch.zeros(tuple(self.discretizations), device=self.device, dtype=torch.bool)
            self.voxel_grid_values = torch.zeros(tuple(self.discretizations) + (self.val_ndim,), device=self.device, dtype=torch.float16)
        
        self.cell_sizes = (self.upper_bound - self.lower_bound) / self.discretizations

        grid_pts = torch.meshgrid([torch.linspace(self.lower_bound[i], self.upper_bound[i], self.discretizations[i]+1, device=self.device) for i in range(self.ndim)])
        self.grid_vertices = torch.stack(grid_pts, dim=-1)      # n1 x n2 x n3 x 3

        if self.ndim == 2:
            self.voxel_grid_centers = 0.5 * (self.grid_vertices[:-1, :-1] + self.grid_vertices[1:, 1:])
            self.lower_grid_center = self.voxel_grid_centers[0, 0]

        elif self.ndim == 3:
            self.voxel_grid_centers = 0.5 * (self.grid_vertices[:-1, :-1, :-1] + self.grid_vertices[1:, 1:, 1:])
            self.lower_grid_center = self.voxel_grid_centers[0, 0, 0]
        
    def compute_voxel_index(self, points, debug=False):
        # IMPORTANT !!! #
        # This function does NOT handle points outside the voxel grid. It is the user's responsibility to ensure that the points are within the voxel grid, because
        # there are many ways in which one could project outside points into the grid. 

        ### IMPORTANT !!! ###
        # We are going with the "round" convention for indexing.
        voxel_index = torch.round( (points - self.lower_grid_center) / self.cell_sizes ).to(torch.int32)

        # Sanity check
        is_in_bounds = []
        for i in range(self.ndim):
            # if debug:
            #     assert torch.all(voxel_index[:, i] >= 0) and torch.all(voxel_index[:, i] < self.discretizations[i]), "Points must be within the voxel grid"
     
            is_in_bounds.append( torch.logical_and( (voxel_index[:, i] >= 0), (voxel_index[:, i] < self.discretizations[i])) )

        is_in_bounds = torch.all( torch.stack(is_in_bounds, dim=-1), dim = -1 )

        return voxel_index, is_in_bounds
    
    def project_points_into_voxel_grid(self, points, directions):
        # Reference: https://tavianator.com/2022/ray_box_boundary.html#:~:text=Geometry&text=The%20slab%20method%20tests%20for,intersects%20the%20box%2C%20if%20any.&text=t%20%3D%20x%20%E2%88%92%20x%200%20x%20d%20.
        ### IMPORTANT !!! ###
        # The directions you pass DO NOT need to be normalized. They will be treated as line segments!

        # If the point is in the voxel grid, we don't do anything.
        # However, if the point is outside the voxel grid, we want to use 
        # directions such that the voxel we return is the first voxel that the ray intersects.
        voxel_index, in_bounds = self.compute_voxel_index(points)

        # indices are within the voxel grid
        in_bound_voxel_index = voxel_index[in_bounds]

        # indices are outside the voxel grid
        out_bound_directions = directions[~in_bounds] + torch.randn_like(directions[~in_bounds]) * 1e-6       # Add some noise to reduce degeneracy
        out_bound_points = points[~in_bounds]

        # Of these out of bound points, there are rays that ultimately intersect the voxel, and others that miss the voxel grid entirely.
        # The general test:

        t1 = (self.lower_bound[None] - out_bound_points) / out_bound_directions     # N x 3
        t2 = (self.upper_bound[None] - out_bound_points) / out_bound_directions     # N x 3

        t = torch.stack([t1, t2], dim=-1)       # N x 3 x 2

        # TODO: We can skip this first min/max step by looking at the signs of the directions directly!
        tmin = torch.min(t, dim=-1).values      # N x 3
        tmax = torch.max(t, dim=-1).values      # N x 3

        tmin = torch.max(tmin, dim=-1).values     # N
        tmax = torch.min(tmax, dim=-1).values     # N

        # If tmin > tmax for any dimension, the line misses the voxel grid
        misses = (tmin - tmax) > 0.     # N

        # TODO: CHECK THIS PROCEDURE!!!

        # Of the rays that we think intersect the voxel grid, we need to make sure that the line segment actually intersects the voxel grid (t < 1)
        # AND we need to make sure the t's are also positive!!!
        short = tmin > 1.     # N
        reversed = tmin < 0.  # N

        # # These points will NEVER intersect the voxel grid
        misses = torch.logical_or(misses, short)
        misses = torch.logical_or(misses, reversed)

        # These are the points that do intersect the voxel grid but are outside the voxel grid
        out_bound_intersect_dirs = out_bound_directions[~misses]
        out_bound_intersect_pts = out_bound_points[~misses]

        # and the progress along the line in order to hit the voxel grid
        t_progress = tmin[~misses]
        remaining_progress = 1. - t_progress

        # and the point where the ray intersects the voxel grid
        intersect_pts = out_bound_intersect_pts + out_bound_intersect_dirs * (t_progress[:, None])     # Add some noise to avoid numerical instability
        intersect_directions = out_bound_intersect_dirs * remaining_progress[:, None]
        intersect_voxel_index, _ = self.compute_voxel_index(intersect_pts)

        in_bound_voxel_index = torch.clamp(in_bound_voxel_index, torch.zeros_like(self.discretizations).unsqueeze(0), self.discretizations.unsqueeze(0) - 1)
        intersect_voxel_index = torch.clamp(intersect_voxel_index, torch.zeros_like(self.discretizations).unsqueeze(0), self.discretizations.unsqueeze(0) - 1)

        # This mask never intersects
        not_intersecting = torch.zeros_like(in_bounds, dtype=torch.bool)
        not_intersecting[~in_bounds] = misses

        output = {
            'in_bounds': in_bounds,
            'not_intersecting': not_intersecting,       # mask of not intersecting data points
            'intersecting_out_bounds': torch.logical_and(~not_intersecting, ~in_bounds), # mask of intersecting data points  (not clearly in bounds but intersecting gives you the points outside the grid that intersect the grid)
            'in_bounds_voxel_index': in_bound_voxel_index,
            'in_bounds_points': points[in_bounds],               # directions remain unchanged since we are already in the voxel grid
            'in_bounds_directions': directions[in_bounds],
            'out_bounds_points': out_bound_intersect_pts,           
            'out_bounds_directions': out_bound_intersect_dirs,
            'out_bounds_intersect_points': intersect_pts,           # project points outside of voxel grid that do intersect the grid into the edge of the grid
            'out_bounds_intersect_direction': intersect_directions, # remaining vector after intersecting the grid
            'out_bounds_intersect_voxel_index': intersect_voxel_index,  # voxel index of the intersecting point with the voxel grid
        }
     
        # Returns the voxel index, the point, the truncated direction for points in the grid, and the indices of points that do intersect the grid
        # Also returns the indices of points that don't ever intersect the voxel
        return output

    #@torch.compile
    def compute_voxel_ray_intersection(self, points, directions, camera_ids, n_cameras):
        # We assume directions is such that points + t * directions, where t = 1, is the termination point

        # tnow = time.time()
        # torch.cuda.synchronize()

        # Project the points into the voxel grid
        output = self.project_points_into_voxel_grid(points, directions)

        ray_indices = torch.arange(len(points), device=self.device)
        not_intersecting_ray_indices = ray_indices[output['not_intersecting']]
        intersecting_ray_indices = ray_indices[~output['not_intersecting']]
        intersecting_camera_ids = camera_ids[~output['not_intersecting']]

        # Segment out the rays that intersect the voxel grid
        in_bounds_points = output['in_bounds_points']
        in_bounds_directions = output['in_bounds_directions']
        in_bounds_voxel_index = output['in_bounds_voxel_index']

        # These are the Out-of-bounds rays, truncated so that they intersect the voxel grid
        out_bounds_intersect_points = output['out_bounds_intersect_points']
        out_bounds_intersect_directions = output['out_bounds_intersect_direction']
        out_bounds_intersect_voxel_index = output['out_bounds_intersect_voxel_index']

        # Concatenate the in-bounds and out-bounds intersecting rays
        points = torch.cat([in_bounds_points, out_bounds_intersect_points], dim=0)
        directions = torch.cat([in_bounds_directions, out_bounds_intersect_directions], dim=0)
        voxel_index = torch.cat([in_bounds_voxel_index, out_bounds_intersect_voxel_index], dim=0)

        frac_indices = ( (points - self.lower_grid_center) / self.cell_sizes ).to(torch.float16)

        # NOTE: It shouldn't be possible to get -1 or self.discretizations[None] in the voxel index
        voxel_index = torch.clamp(voxel_index, torch.zeros_like(self.discretizations).unsqueeze(0), self.discretizations.unsqueeze(0) - 1).to(torch.int32)

        # exiting_ray_indices = []
        # out_of_length_ray_indices = []

        terminated_voxel_index = torch.zeros((len(intersecting_ray_indices), 3), device=self.device, dtype=torch.int32)
        terminated_ray_index = torch.zeros(len(intersecting_ray_indices), device=self.device, dtype=torch.int32)
        terminated_camera_id = torch.zeros(len(intersecting_ray_indices), device=self.device, dtype=torch.int32)
        terminated_voxel_directions = torch.zeros((len(intersecting_ray_indices), 3), device=self.device, dtype=torch.float16)

        # NOTE: This could be computed just once outside of this function!
        indices_directions = (directions / self.cell_sizes.unsqueeze(0)).to(torch.float16)
        sign_directions = torch.sign(directions).to(torch.int32)
        half_sign_directions = (sign_directions / 2).to(torch.float16)

        num_terminated = 0
        remaining_rays = len(intersecting_ray_indices)

        traversed_voxels = torch.zeros_like(self.voxel_grid_binary, dtype=torch.bool)

        # DATA TENSOR
        data = TensorDict({
            'intersecting_ray_indices': intersecting_ray_indices,
            'frac_indices': frac_indices,
            'voxel_index': voxel_index,
            'indices_directions': indices_directions,
            'sign_directions': sign_directions,
            'half_sign_directions': half_sign_directions,
            'intersecting_camera_ids': intersecting_camera_ids
        }, batch_size=(len(intersecting_ray_indices),))

        while remaining_rays > 0:

            # Break out components of data tensor. NOTE: In-place!
            intersecting_ray_indices_ = data['intersecting_ray_indices']
            frac_indices_ = data['frac_indices']
            voxel_index_ = data['voxel_index']
            indices_directions_ = data['indices_directions']
            sign_directions_ = data['sign_directions']
            half_sign_directions_ = data['half_sign_directions']
            camera_ids_ = data['intersecting_camera_ids']

            # Begin the incremental traversal procedure
            terminated = self.is_terminated(voxel_index_)

            traversed_voxels[voxel_index_[:, 0], voxel_index_[:, 1], voxel_index_[:, 2]] = True

            # Store the terminated rays
            num_terminated_now = num_terminated + torch.sum(terminated)
            terminated_ray_index[ num_terminated:num_terminated_now ] = intersecting_ray_indices_[terminated]
            terminated_camera_id[ num_terminated:num_terminated_now ] = camera_ids_[terminated]
            terminated_voxel_index[ num_terminated:num_terminated_now ] = voxel_index_[terminated]
            terminated_voxel_directions[ num_terminated:num_terminated_now ] = indices_directions_[terminated]
            num_terminated = num_terminated_now

            frac_indices_, voxel_index_, indices_directions_, min_t, min_idx = one_step_voxel_ray_intersection(frac_indices_, voxel_index_, indices_directions_, sign_directions_, half_sign_directions_)

            # If the min_t is greater than 1, we have reached the end of the ray
            out_of_length = (min_t >= 1.)

            # NOTE: Indices cannot exceed 256!
            voxel_index_ = voxel_index_.to(torch.uint8)
            out_of_bounds = ( torch.gather(voxel_index_, 1, min_idx) - self.discretizations[min_idx] <= -1 ).squeeze()
 
            not_keep = out_of_length | ~out_of_bounds | terminated

            # NOTE! It is possible to have rays that are both out of bounds and out of length! We will treat them as out of bounds.
            # exiting_ray_indices.append(intersecting_ray_indices[out_of_bounds])
            # out_of_length_ray_indices.append(intersecting_ray_indices[out_of_length & ~out_of_bounds])

            # If the min_t is less than 1, we have not reached the end of the ray
            keep = ~not_keep
 
            # Cache indexing results for reuse
            keep_indices = keep.nonzero(as_tuple=True)[0]  # Avoid redundant computation

            data = data[keep_indices]
  
            remaining_rays = len(keep_indices)

        terminated_ray_index = terminated_ray_index[:num_terminated]
        terminated_camera_id = terminated_camera_id[:num_terminated]
        terminated_voxel_index = terminated_voxel_index[:num_terminated]
        terminated_voxel_directions = terminated_voxel_directions[:num_terminated]

        meta = {
            'n_cameras': n_cameras,
            'camera_ids': terminated_camera_id
        }
        
        terminated_voxel_values = self.compute_termination_values(terminated_voxel_index, terminated_voxel_directions, extras=meta)
        
        # We want to store
        output = {
            # For rays that terminate due to the termination fn, we store the voxel index, 
            # the ray index, and the value of the grid at termination
            'terminated_voxel_index': terminated_voxel_index.to(torch.int32),
            'terminated_ray_index': terminated_ray_index.to(torch.int32),
            'terminated_voxel_values': terminated_voxel_values.to(torch.float32),
            'terminated_voxel_directions': terminated_voxel_directions.to(torch.float32),
            'terminated_camera_id': terminated_camera_id,
            'traversed_voxels': traversed_voxels,   # N x N x N voxel grid of booleans

            # For rays that (1) don't hit the voxel grid at all, (2) exit the voxel grid, or (3) reach the end of the ray length
            # are stored by their ray index. Although functionally, these three categories are treated the same (i.e. they don't render to anything),
            # the designations could be used for downstream tasks.
            # 'not_intersecting_ray_index': not_intersecting_ray_indices,
            # 'exiting_ray_index': torch.cat(exiting_ray_indices, dim=0) if len(exiting_ray_indices) > 0 else None,
            # 'out_of_length_ray_index': torch.cat(out_of_length_ray_indices, dim=0) if len(out_of_length_ray_indices) > 0 else None,
        }
        #print('Time taken for intersections: ', tdiff)
        return output
    
    def is_terminated(self, indices):
        return self.voxel_grid_binary[indices[:, 0], indices[:, 1], indices[:, 2]]
    
    def compute_termination_values(self, indices, directions, extras=None):
        #terminated_values = torch.zeros((len(indices), self.val_ndim + 1), device=self.device)
        voxel_values = self.voxel_grid_values[ indices[:, 0], indices[:, 1], indices[:, 2] ]
        #terminated_values[:, :voxel_values.shape[-1]] = voxel_values
        return voxel_values

    # Renders the voxel grid, given the voxel grid, camera extrinsics, and intrinsics.
    # Makes use of the nerfacc library for fast rendering.
    # We can play with some of the approximations, or make it exact if need be.

    # TODO: Might implement a near clip by just shifting the camera origin along some near_clip distance along ray direction.
    def camera_voxel_intersection(self, K, c2w, far_clip, near_clip=0.):
        W, H = int(K[0, 2] * 2), int(K[1, 2] * 2)

        if c2w.dim() == 2:
            origins, directions = get_rays(H, W, K, c2w, self.device)
            camera_ids = torch.zeros(H * W, dtype=torch.int32, device=self.device)
            B = 1
        else:
            B = c2w.shape[0]    # batch number of poses
            origins, directions = get_rays_batch(H, W, K, c2w, self.device)
            camera_ids = torch.arange(B, device=self.device).repeat_interleave(H * W)

        # Flatten origins and directions
        origins = origins.reshape(-1, 3)
        directions = directions.reshape(-1, 3)

        #normalize directions
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        # Move the camera origins a near clip distance
        if near_clip > 0.:
            origins = origins + directions * near_clip
            
        # Extend directions to far_clip
        directions = directions * far_clip

        # Compute the voxel intersections
        output = self.compute_voxel_ray_intersection(origins, directions, camera_ids, B)

        n_channels = output['terminated_voxel_values'].shape[-1]

        if c2w.dim() == 2:
            # render one image

            # Create image and depth
            image = torch.zeros((H, W, n_channels), device=self.device)
            depth = torch.zeros((H, W), device=self.device)

            # Compute the depth and image
            image = image.view(-1, n_channels)
            image[output['terminated_ray_index']] = output['terminated_voxel_values']
            image = image.view(H, W, n_channels)

            depth = depth.view(-1)
            depth[output['terminated_ray_index']] = torch.linalg.norm(self.voxel_grid_centers[output['terminated_voxel_index'][:, 0], 
                                                                            output['terminated_voxel_index'][:, 1], 
                                                                            output['terminated_voxel_index'][:, 2]] - c2w[:3, -1].unsqueeze(0), dim=-1)
            depth = depth.view(H, W, 1) 

        else:
            # render batch of images
            # Create image and depth
            image = torch.zeros((B, H, W, n_channels), device=self.device)
            depth = torch.zeros((B, H, W), device=self.device)

            # Compute the depth and image
            image = image.reshape(-1, n_channels)
            image[output['terminated_ray_index']] = output['terminated_voxel_values']
            image = image.reshape(B, H, W, n_channels)

            depth = depth.reshape(-1)

            depth[output['terminated_ray_index']] = torch.linalg.norm(self.voxel_grid_centers[output['terminated_voxel_index'][:, 0], 
                                                                            output['terminated_voxel_index'][:, 1], 
                                                                            output['terminated_voxel_index'][:, 2]] - origins[output['terminated_ray_index']], dim=-1)
            depth = depth.reshape(B, H, W)

        output['rays_o'] = origins
        output['rays_d'] = directions

        return image, depth, output

    def populate_voxel_grid_from_points(self, points, values):
        voxel_index, is_in_grid = self.compute_voxel_index(points)

        # Only store the points that are in the grid
        voxel_index = voxel_index[is_in_grid]

        # Populate the voxel grid with the values
        # NOTE: THIS WILL OVERWRITE THE VALUES! WE MIGHT WANT TO DO A SCATTER ADD OR SOMETHING MORE SOPHISTICATED.
        self.voxel_grid_values[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]] = values[is_in_grid].to(torch.float16)
        self.voxel_grid_binary[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]] = True
    
    def create_mesh(self):
        grid_centers = self.voxel_grid_centers[self.voxel_grid_binary]
        grid_centers = grid_centers.view(-1, 3).cpu().numpy()

        grid_colors = self.voxel_grid_values[self.voxel_grid_binary]
        grid_colors = grid_colors.view(-1, 3).cpu().numpy()

        scene = o3d.geometry.TriangleMesh()
        for cell_color, cell_center in zip(grid_colors, grid_centers):
            box = o3d.geometry.TriangleMesh.create_box(width=self.cell_sizes[0].cpu().numpy(), 
                                                        height=self.cell_sizes[1].cpu().numpy(), 
                                                        depth=self.cell_sizes[2].cpu().numpy())
            box = box.translate(cell_center, relative=False)
            box.paint_uniform_color(cell_color)
            scene += box

        return scene

    def create_mesh_from_points(self, points, colors=None):
        # colors needs to be thhe same length as points

        # Create a mesh from the voxel grid
        voxel_index, in_bounds = self.compute_voxel_index(points)

        #assert torch.all(in_bounds), "All points must be within the voxel grid"

        colors = colors[in_bounds]
        voxel_index = voxel_index[in_bounds]

        voxel_index_flatten = torch.tensor(np.ravel_multi_index(voxel_index.T.cpu().numpy(), tuple(self.discretizations)), device=self.device)

        # Store the number of counts for each voxel
        voxel_counts = torch.zeros(tuple(self.discretizations), device=self.device, dtype=voxel_index_flatten.dtype).flatten()
        voxel_counts.scatter_add_(0, voxel_index_flatten, torch.ones_like(voxel_index_flatten))

        voxel_counts = voxel_counts.reshape(tuple(self.discretizations))

        # Store the colors for each voxel
        voxel_colors = torch.zeros(tuple(self.discretizations) + (3,), device=self.device, dtype=colors.dtype).reshape(-1, 3)
        voxel_colors.scatter_add_(0, voxel_index_flatten.unsqueeze(1).expand(-1, 3), colors)

        voxel_colors = voxel_colors.reshape(tuple(self.discretizations) + (3,))

        # Create the mesh
        mask = voxel_counts > 0

        grid_centers = self.voxel_grid_centers[mask]
        grid_centers = grid_centers.view(-1, 3).cpu().numpy()

        grid_colors = voxel_colors[mask]
        grid_colors = grid_colors.view(-1, 3).cpu().numpy()

        grid_colors = grid_colors / voxel_counts[mask].unsqueeze(1).cpu().numpy()

        scene = o3d.geometry.TriangleMesh()
        for cell_color, cell_center in zip(grid_colors, grid_centers):
            box = o3d.geometry.TriangleMesh.create_box(width=self.cell_sizes[0].cpu().numpy(), 
                                                        height=self.cell_sizes[1].cpu().numpy(), 
                                                        depth=self.cell_sizes[2].cpu().numpy())
            box = box.translate(cell_center, relative=False)
            box.paint_uniform_color(cell_color)
            scene += box

        return scene