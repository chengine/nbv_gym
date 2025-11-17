import torch
import numpy as np
import open3d as o3d
from .voxel_traversal.traversal_utils import VoxelGrid
# from distributions.vmf import FisherKLDivergence
from .coverage.utils import VoxelCoverage

# class Vista(VoxelGrid):
#     def __init__(self, param_dict, pointcloud, ndim, device):
#         super().__init__(param_dict, ndim, device)

#         self.K = param_dict['K']
#         self.far_clip = param_dict['far_clip']

#         self.eigvals = None
#         self.eigvecs = None

#         self.val_ndim = param_dict['val_ndim']

#         self.pointcloud = pointcloud        # point cloud is a dictionary (rather than a tensor so we don't have to hard code anything in here.)

#         points = self.pointcloud['points']
#         values = self.pointcloud['values']
#         origins = self.pointcloud['origins']
#         directions = points - origins

#         self.populate_voxel_grid_from_pointcloud(points, values, origins, directions)
    
#     def populate_voxel_grid_from_pointcloud(self, points, values, origins, directions):
#         assert values.shape[-1] == self.val_ndim

#         voxel_index, is_in_grid = self.compute_voxel_index(points)

#         # Only store the points that are in the grid
#         voxel_index = voxel_index[is_in_grid]

#         # Populate the voxel grid with the values
#         self.voxel_grid_values[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]] = values[is_in_grid].to(torch.float16)
#         self.voxel_grid_binary[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]] = True

#         # Uses the viewing directions to calculate a covariance for each voxel
#         self.eigvals, self.eigvecs, output = self.calculate_view_covariance(origins, points, directions)
#         free_and_occupied = output['traversed_voxels']

#         # Calculate the totality of the occupied and unobserved
#         self.voxel_grid_binary = self.voxel_grid_binary | ~free_and_occupied

#     def calculate_view_covariance(self, origins, ends, directions):

#         # Compute the voxel intersections from the training views
#         output = self.compute_voxel_ray_intersection(origins, directions)
        
#         # Make sure that the voxel intersections are in bounds
#         voxel_index, in_bounds = self.compute_voxel_index(ends)
#         voxel_index = voxel_index[in_bounds]

#         directions = directions[in_bounds]

#         # Normalize the directions
#         directions = directions / torch.norm(directions, dim=-1, keepdim=True)

#         # Create outer product from normalized directions
#         outer_product = torch.einsum("bi,bj->bij", directions, directions)

#         voxel_index_flatten = torch.tensor( np.ravel_multi_index( voxel_index.T.cpu().numpy(), tuple(self.discretizations) ), device=self.device)

#         # Store the number of counts for each voxel
#         voxel_counts = torch.zeros( tuple(self.discretizations), device=self.device, dtype=voxel_index_flatten.dtype).flatten()

#         voxel_counts.scatter_add_( 0, voxel_index_flatten, torch.ones_like(voxel_index_flatten) )
#         voxel_counts = voxel_counts.reshape(tuple(self.discretizations))

#         # Store the covariance matrix for each voxel
#         voxel_covariance = torch.zeros( tuple(self.discretizations) + (3, 3), device=self.device, dtype=outer_product.dtype ).reshape(-1, 3, 3)
#         voxel_covariance.scatter_add_( 0, voxel_index_flatten.unsqueeze(1).unsqueeze(1).expand(-1, 3, 3), outer_product )

#         voxel_covariance = voxel_covariance / ( voxel_counts.reshape(-1).unsqueeze(1).unsqueeze(1) + 1e-6 )

#         # Compute the eigenvalues and vectors associated with each covariance in each voxel
#         eigvals, eigvecs = torch.linalg.eigh(voxel_covariance)

#         # Reshaped to be a grid of eigenvalues and eigenvectors (N x N x N x (ndim or ndim x ndim))
#         eigvals = eigvals.reshape(tuple(self.discretizations) + (3,) )

#         eigvecs = eigvecs.reshape( tuple(self.discretizations) + (3,3) )

#         return eigvals, eigvecs, output

#     def compute_termination_values(self, indices, directions):
#         terminated_values = torch.zeros((len(directions), self.val_ndim + 1), device=self.device)

#         voxel_values = self.voxel_grid_values[ indices[:, 0], indices[:, 1], indices[:, 2] ]

#         terminated_values[:, :voxel_values.shape[-1]] = voxel_values

#         if self.eigvals is not None:
#             print('Calculating View Diversity Values')
#             vecs = self.eigvecs[indices[:, 0], indices[:, 1], indices[:, 2]]

#             vals = torch.nn.functional.relu( self.eigvals[indices[:, 0], indices[:, 1], indices[:, 2]] )

#             values = ( torch.bmm(directions[..., None, :], vecs).squeeze() / torch.norm(directions, dim=-1, keepdim=True) ) ** 2

#             values = values / (vals + 1.) #+ 1e-4)
#             values = torch.sum(values, dim=-1)

#             terminated_values[:, -1] = values

#         return terminated_values

#     def next_best(self, candidate_poses):
#         # Evaluate the methods on these candidate poses. TODO: We need to train the method before this part.

#         image, depth, output = self.camera_voxel_intersection(self.K, candidate_poses, self.far_clip)

#         # Sum up all the values in the image in all dimensions instead for the batch dimension. Outputs a tensor of shape B
#         loss = image.sum(dim=tuple(range(1, image.ndim)))

#         best_idx = torch.argmax(loss)

#         best_pose = candidate_poses[best_idx]

#         meta = {
#             'image': image,
#             'depth': depth,
#             'output': output
#         }

#         return best_pose, best_idx, meta
    
#     def update_view_diversity(self, candidate_poses):
#         # TODO: Implement this method
#         raise NotImplementedError
    
# class VistaProbabilistic(VoxelGrid):
#     def __init__(self, param_dict, pointcloud, ndim, device):
#         super().__init__(param_dict, ndim, device)

#         self.K = param_dict['K']
#         self.far_clip = param_dict['far_clip']

#         self.base_directions = None
#         self.base_indices = None

#         self.val_ndim = param_dict['val_ndim']

#         self.pointcloud = pointcloud        # point cloud is a dictionary (rather than a tensor so we don't have to hard code anything in here.)

#         points = self.pointcloud['points']
#         values = self.pointcloud['values']
#         origins = self.pointcloud['origins']
#         directions = points - origins

#         self.fisher_kl = FisherKLDivergence(device, self.discretizations[0] * self.discretizations[1] * self.discretizations[2])

#         self.populate_voxel_grid_from_pointcloud(points, values, origins, directions)
    
#     def populate_voxel_grid_from_pointcloud(self, points, values, origins, directions):
#         assert values.shape[-1] == self.val_ndim

#         voxel_index, is_in_grid = self.compute_voxel_index(points)

#         # Only store the points that are in the grid
#         voxel_index = voxel_index[is_in_grid]

#         # Populate the voxel grid with the values
#         self.voxel_grid_values[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]] = values[is_in_grid].to(torch.float16)
#         self.voxel_grid_binary[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]] = True

#         # Uses the viewing directions to calculate a covariance for each voxel
#         output = self.calculate_view_diversity(origins, points, directions)
#         free_and_occupied = output['traversed_voxels']

#         # Calculate the totality of the occupied and unobserved
#         self.voxel_grid_binary = self.voxel_grid_binary | ~free_and_occupied

#     def calculate_view_diversity(self, origins, ends, directions):

#         # Compute the voxel intersections from the training views
#         output = self.compute_voxel_ray_intersection(origins, directions)
        
#         # Make sure that the voxel intersections are in bounds
#         voxel_index, in_bounds = self.compute_voxel_index(ends)
#         voxel_index = voxel_index[in_bounds]

#         directions = directions[in_bounds]

#         # Normalize the directions
#         directions = directions / torch.norm(directions, dim=-1, keepdim=True)

#         self.base_directions = directions
#         self.base_indices = torch.tensor( np.ravel_multi_index( voxel_index.T.cpu().numpy(), tuple(self.discretizations) ), device=self.device) # Flattened indices

#         self.fisher_kl.fit_dist0(self.base_directions, self.base_indices, num_iters=10)

#         return output

#     def compute_termination_values(self, indices, directions):
#         terminated_values = torch.zeros((len(directions), self.val_ndim + 1), device=self.device)

#         voxel_values = self.voxel_grid_values[ indices[:, 0], indices[:, 1], indices[:, 2] ]

#         terminated_values[:, :voxel_values.shape[-1]] = voxel_values

#         if self.base_directions is not None:
#             print('Calculating View Diversity Values')
#             directions = directions / torch.norm(directions, dim=-1, keepdim=True)
#             indices_ = torch.tensor( np.ravel_multi_index( indices.T.cpu().numpy(), tuple(self.discretizations) ), device=self.device) # Flattened indices

#             data = torch.cat([self.base_directions, directions], dim=0)
#             index = torch.cat([self.base_indices, indices_], dim=0)

#             print(self.base_directions.shape, data.shape)

#             self.fisher_kl.fit_dist1(data, index, num_iters=10)

#             voxel_kl = self.fisher_kl.evaluate()

#             terminated_values[:, -1] = voxel_kl[indices_] #.reshape(tuple(self.discretizations))[indices[:, 0], indices[:, 1], indices[:, 2]]

#         return terminated_values

#     def next_best(self, candidate_poses):
#         # Evaluate the methods on these candidate poses. TODO: We need to train the method before this part.

#         image, depth, output = self.camera_voxel_intersection(self.K, candidate_poses, self.far_clip)

#         # Sum up all the values in the image in all dimensions instead for the batch dimension. Outputs a tensor of shape B
#         loss = image.sum(dim=tuple(range(1, image.ndim)))

#         best_idx = torch.argmax(loss)

#         best_pose = candidate_poses[best_idx]

#         meta = {
#             'image': image,
#             'depth': depth,
#             'output': output
#         }

#         return best_pose, best_idx, meta
    
#     def update_view_diversity(self, candidate_poses):
#         # TODO: Implement this method
#         raise NotImplementedError
    
class VistaCoverage(VoxelGrid):
    def __init__(self, param_dict, pointcloud, ndim, device):
        super().__init__(param_dict, ndim, device)

        self.far_clip = param_dict['far_clip']

        self.base_directions = None
        self.base_indices = None

        self.val_ndim = param_dict['val_ndim']
        self.coverage = VoxelCoverage(device, param_dict['discretizations'].prod().item(), 3)

        self.pointcloud = pointcloud        # point cloud is a dictionary (rather than a tensor so we don't have to hard code anything in here.)

        points = self.pointcloud['points']
        values = self.pointcloud['values']

        if 'origins' not in self.pointcloud.keys():
            cameras = self.pointcloud['cameras']
            self.populate_voxel_grid_from_pointcloud_cameras(points, values, cameras)
        else:
            origins = self.pointcloud['origins']
            directions = points - origins

            # Near clip
            if 'near_clip' in self.pointcloud.keys():
                self.near_clip = self.pointcloud['near_clip']
                # Normalize directions
                directions_normalized = directions / torch.norm(directions, dim=-1, keepdim=True)
                origins = origins + directions_normalized * self.near_clip
            self.populate_voxel_grid_from_pointcloud_rays(points, values, origins, directions)

    def populate_voxel_grid_from_pointcloud_cameras(self, points, values, cameras):
        assert values.shape[-1] == self.val_ndim
        self.populate_voxel_grid_from_points(points, values)

        # Uses the viewing directions to calculate a covariance for each voxel
        output = self.calculate_view_diversity_cameras(cameras)
        free_and_occupied = output['traversed_voxels']

        # Calculate the totality of the occupied and unobserved
        self.voxel_grid_binary = torch.logical_or(self.voxel_grid_binary, ~free_and_occupied)

    def populate_voxel_grid_from_pointcloud_rays(self, points, values, origins, directions):
        assert values.shape[-1] == self.val_ndim
        self.populate_voxel_grid_from_points(points, values)

        # Uses the viewing directions to calculate a covariance for each voxel
        output = self.calculate_view_diversity_rays(origins, points, directions)
        free_and_occupied = output['traversed_voxels']

        # Calculate the totality of the occupied and unobserved
        self.voxel_grid_binary = torch.logical_or(self.voxel_grid_binary, ~free_and_occupied)

    ### TODO: ABILITY TO UPDATE VOXEL GRID FROM POINTCLOUD
    # def update_voxel_grid_from_pointcloud(self, pointcloud):
    #     points = pointcloud['points']
    #     values = pointcloud['values']
    #     origins = pointcloud['origins']
    #     directions = points - origins

    #     assert values.shape[-1] == self.val_ndim

    #     voxel_index, is_in_grid = self.compute_voxel_index(points)

    #     # Only store the points that are in the grid
    #     voxel_index = voxel_index[is_in_grid]

    #     # Populate the voxel grid with the values
    #     self.voxel_grid_values[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]] = values[is_in_grid].to(torch.float16)
    #     self.voxel_grid_binary[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]] = True

    #     # Uses the viewing directions to calculate a covariance for each voxel
    #     output = self.update_view_diversity(origins, points, directions)
    #     free_and_occupied = output['traversed_voxels']

    #     # Calculate the totality of the occupied and unobserved
    #     self.voxel_grid_binary = torch.logical_or(self.voxel_grid_binary, ~free_and_occupied)

    # def update_view_diversity(self, origins, ends, directions):
    #     # Compute the voxel intersections from the training views
    #     output = self.compute_voxel_ray_intersection(origins, directions, torch.zeros(origins.shape[0], dtype=torch.int32, device=self.device), 1)
        
    #     #NOTE: Is it OK for rays to run into the far clip in the grid? Probably
    #     # safer to make the far clip very large?

    #     # Make sure that the voxel intersections are in bounds
    #     voxel_index, in_bounds = self.compute_voxel_index(ends)
    #     voxel_index = voxel_index[in_bounds]

    #     directions = directions[in_bounds]

    #     # Normalize the directions
    #     directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    #     self.base_directions = directions
    #     self.base_indices = torch.tensor( np.ravel_multi_index( voxel_index.T.cpu().numpy(), tuple(self.discretizations) ), device=self.device) # Flattened indices

    #     self.coverage.load_data(self.base_directions, self.base_indices)

    #     return output

    def calculate_view_diversity_cameras(self, cameras):

        # Compute the voxel intersections from the training views
        K = cameras["K"]    # tensor
        c2w = cameras["c2w"]        # 3D tensor of shape (n_cameras, 4, 4)
        far_clip = cameras["far_clip"]  # float
        near_clip = cameras["near_clip"]  # float
        _, _, output = self.camera_voxel_intersection(K, c2w, far_clip, near_clip)
        
       # Make sure that the terminated voxel intersections are in bounds
        voxel_index = output['terminated_voxel_index']
        ray_directions_total = output['rays_d']
        directions = ray_directions_total[output['terminated_ray_index']]

        assert voxel_index.shape[0] == directions.shape[0]

        # Normalize the directions
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        self.base_directions = directions
        self.base_indices = torch.tensor( np.ravel_multi_index( voxel_index.T.cpu().numpy(), tuple(self.discretizations) ), device=self.device) # Flattened indices

        self.coverage.load_data(self.base_directions, self.base_indices)

        return output

    def calculate_view_diversity_rays(self, origins, ends, directions):
        # Compute the voxel intersections from the training views
        output = self.compute_voxel_ray_intersection(origins, directions, torch.zeros(origins.shape[0], dtype=torch.int32, device=self.device), 1)
        
        #NOTE: Is it OK for rays to run into the far clip in the grid? Probably
        # safer to make the far clip very large?
        # Make sure that the voxel intersections are in bounds
        voxel_index, in_bounds = self.compute_voxel_index(ends)
        voxel_index = voxel_index[in_bounds]

        directions = directions[in_bounds]

        # Normalize the directions
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        self.base_directions = directions
        self.base_indices = torch.tensor( np.ravel_multi_index( voxel_index.T.cpu().numpy(), tuple(self.discretizations) ), device=self.device) # Flattened indices

        self.coverage.load_data(self.base_directions, self.base_indices)

        return output

    @torch.compile
    def compute_termination_values(self, indices, directions, extras=None):
        terminated_values = torch.zeros((len(directions), self.val_ndim + 1), device=self.device)

        voxel_values = self.voxel_grid_values[ indices[:, 0], indices[:, 1], indices[:, 2] ]

        terminated_values[:, :voxel_values.shape[-1]] = voxel_values

        if self.base_directions is not None:
            print('Calculating View Diversity Values')
            directions = directions / torch.norm(directions, dim=-1, keepdim=True)

            ### THIS IS FOR SINGLE CAMERA ###
            # indices = torch.tensor( np.ravel_multi_index( indices.T.cpu().numpy(), tuple(self.discretizations) ), device=self.device) # Flattened indices

            # query = torch.zeros((self.discretizations.prod().item(), 3), device=self.device)
            # query[indices] = directions

            # coverage_metric = self.coverage.compute_existing_coverage(query)        # n_voxels x n_dim

            ### THIS IS FOR BATCHED CAMERAS ###
            n_cameras = extras['n_cameras']
            indices = torch.tensor( np.ravel_multi_index( indices.T.cpu().numpy(), tuple(self.discretizations) ), device=self.device) # Flattened indices
            indices = torch.stack([extras['camera_ids'], indices], dim=-1) #cat with camera indices
             
            query = torch.zeros((n_cameras, self.discretizations.prod().item(), 3), device=self.device, dtype=torch.float16) # n_cameras x n_voxels x ndim
            query[indices[:, 0], indices[:, 1]] = directions

            coverage_metric = self.coverage.compute_existing_coverage_batched(query)       # n_cameras x n_voxels
            terminated_values[:, -1] = (coverage_metric[indices[:, 0], indices[:, 1]] + 1.) / 2. # normalizes the coverage metric to be between 0 and 1. 

        return terminated_values

    # def next_best(self, candidate_poses):
    #     # Evaluate the methods on these candidate poses. TODO: We need to train the method before this part.

    #     image, depth, output = self.camera_voxel_intersection(self.K, candidate_poses, self.far_clip)

    #     # Sum up all the values in the image in all dimensions instead for the batch dimension. Outputs a tensor of shape B
    #     loss = image.sum(dim=tuple(range(1, image.ndim)))

    #     best_idx = torch.argmax(loss)

    #     best_pose = candidate_poses[best_idx]

    #     meta = {
    #         'image': image,
    #         'depth': depth,
    #         'output': output
    #     }

    #     return best_pose, best_idx, meta
    
    # def update_view_diversity(self, candidate_poses):
    #     # TODO: Implement this method
    #     raise NotImplementedError