import torch
import time

@torch.compile
def compute_voxel_ray_intersection(self, points, directions):
    # We assume directions is such that points + t * directions, where t = 1, is the termination point

    # tnow = time.time()
    # torch.cuda.synchronize()

    # Project the points into the voxel grid
    output = self.project_points_into_voxel_grid(points, directions)

    ray_indices = torch.arange(len(points), device=self.device)
    not_intersecting_ray_indices = ray_indices[output['not_intersecting']]
    intersecting_ray_indices = ray_indices[~output['not_intersecting']]

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
    terminated_voxel_values = torch.zeros((len(intersecting_ray_indices), 3), device=self.device, dtype=torch.float32)

    # torch.cuda.synchronize()
    # print("Time taken for initialization: ", time.time() - tnow)

    # NOTE: This could be computed just once outside of this function!
    indices_directions = (directions / self.cell_sizes.unsqueeze(0)).to(torch.float16)
    sign_directions = torch.sign(directions + 1e-6*torch.randn_like(directions)).to(torch.int32)
    half_sign_directions = (sign_directions / 2).to(torch.float16)

    counter = 0
    num_terminated = 0
    remaining_rays = len(intersecting_ray_indices)
    # tnow = time.time()
    # torch.cuda.synchronize()
    tdiff = 0
    while remaining_rays > 0:
        # Begin the incremental traversal procedure
        terminated, values = self.termination_fn(voxel_index, indices_directions)

        # Store the terminated rays
        num_terminated_now = num_terminated + torch.sum(terminated).item()
        terminated_ray_index[ num_terminated:num_terminated_now ] = intersecting_ray_indices[terminated]
        terminated_voxel_index[ num_terminated:num_terminated_now ] = voxel_index[terminated]
        terminated_voxel_values[ num_terminated:num_terminated_now ] = values[terminated]
        num_terminated = num_terminated_now

        next_frac_indices, next_indices, min_t = one_step_voxel_ray_intersection(frac_indices, voxel_index, indices_directions, sign_directions, half_sign_directions)


        # If the min_t is greater than 1, we have reached the end of the ray
        out_of_length = (min_t >= 1.)

        # We also want to terminate if the ray leaves the voxel grid (NOTE: !!! IMPORTANT !!! This is true only for a single voxel grid!!! We
        # just need to truncate and store the ray if we have multiple voxel grids)
        out_of_bounds = torch.any( (next_indices < 0) | (next_indices - self.discretizations.unsqueeze(0) > -1) , dim=-1)     #N
        not_keep = out_of_length | out_of_bounds | terminated

        # NOTE! It is possible to have rays that are both out of bounds and out of length! We will treat them as out of bounds.
        # exiting_ray_indices.append(intersecting_ray_indices[out_of_bounds])
        # out_of_length_ray_indices.append(intersecting_ray_indices[out_of_length & ~out_of_bounds])

        # If the min_t is less than 1, we have not reached the end of the ray
        keep = ~not_keep

        tnow = time.time()
        torch.cuda.synchronize()
        # Update the rays that continue marching
        voxel_index = next_indices[keep]
        frac_indices = next_frac_indices[keep]
        sign_directions = sign_directions[keep]
        half_sign_directions = half_sign_directions[keep]
        intersecting_ray_indices = intersecting_ray_indices[keep]

        # We want to truncate the directions to account for traversing by min_t
        indices_directions = (1. - min_t[keep]).unsqueeze(-1) * indices_directions[keep]

        remaining_rays = keep.sum().item()

        torch.cuda.synchronize()
        tdiff += time.time() - tnow

        # torch.cuda.synchronize()
        # print("Time taken for cleanup: ", time.time() - tnow)
        if counter % 50 == 0:
            print('Ray Tracing Step: ', counter)
            print("Number of rays remaining: ", remaining_rays)

        counter += 1
    # torch.cuda.synchronize()
    # print('Ray tracing time', time.time() - tnow)

    terminated_ray_index = terminated_ray_index[:num_terminated]
    terminated_voxel_index = terminated_voxel_index[:num_terminated]
    terminated_voxel_values = terminated_voxel_values[:num_terminated]
    
    # We want to store
    output = {
        # For rays that terminate due to the termination fn, we store the voxel index, 
        # the ray index, and the value of the grid at termination
        'terminated_voxel_index': terminated_voxel_index,
        'terminated_ray_index': terminated_ray_index,
        'terminated_voxel_values': terminated_voxel_values,

        # For rays that (1) don't hit the voxel grid at all, (2) exit the voxel grid, or (3) reach the end of the ray length
        # are stored by their ray index. Although functionally, these three categories are treated the same (i.e. they don't render to anything),
        # the designations could be used for downstream tasks.
        # 'not_intersecting_ray_index': not_intersecting_ray_indices,
        # 'exiting_ray_index': torch.cat(exiting_ray_indices, dim=0) if len(exiting_ray_indices) > 0 else None,
        # 'out_of_length_ray_index': torch.cat(out_of_length_ray_indices, dim=0) if len(out_of_length_ray_indices) > 0 else None,
    }
    print('Time taken for intersections: ', tdiff)
    return output