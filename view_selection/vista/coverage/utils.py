import torch

#TODO: Half-precision. Compile. 
class VoxelCoverage:
    def __init__(self, device, n_voxels, n_dim):
        self.device = device
        self.n_voxels = n_voxels
        self.n_dim = n_dim

    def load_data(self, data, indices):
        # Data is in the form of a tensor of shape (N, 3), where N is the number of data points. 
        # Indices need to be 1D (3D indices must be converted to 1D)
        # This data is intrinsic to the instance of VoxelCoverage, meaning its to query
        # coverage with the extant dataset since rollouts build on the dataset.
        self.data_values = data
        # Normalize data
        self.data_values = self.data_values / torch.norm(self.data_values, dim=-1, keepdim=True)
        self.data_indices = indices

    def compute_coverage_rollout(self, rollouts, rollout_indices):
        # rollouts: list of N tensors, for N time steps. Each tensor is the aggregation of viewpoints
        # at time step k. Note that every camera pose in this tensor can have different number of 
        # direction vectors, and so the tensor is a B x 3 tensor, with a corresponding B x 1 indices
        # tensor that is a flattened (N_cameras, i, j, k) index. NOTE: This index tensor is flattened 4D, which          
        # is different than the flattened 3D index in load_data.

        n_steps = len(rollout_indices)

        # For the first time step, all camera trajectories depend on the same set of data. 

        # For subsequent steps, we compute the coverage metric with respect to the static extant dataset like
        # for the first time step. Then we compute the coverage metric for a growing dataset that consists of only
        # the hypothetical view directions from the rollouts. This way, we can reduce computation by not copying the extant
        # data for every camera trajectory. Then we find the min between the two tensors. 

    # NOTE: For all coverage computations, we assume that each voxel has at most 
    # one direction vector associated with it in the QUERY (multiple directions in the dataset is OK.)
    def compute_existing_coverage(self, query):
        # New data is the view directions that we would like to add the voxels, so we want to compute the coverage of every single entry in the new_data
        # Returns the coverage of the new data, of length n_voxels.

        # query: tensor of size # n_voxels x n_dims (importantly implicitly sorted by voxel index). Every voxel gets a query direction vector. If
        # the voxel is not queried, it is the 0 vector. The query is assumed to be normalized (for non-origin vectors)

        # This is for voxels that exist in the extant dataset.
        dot_product = torch.sum( -self.data_values * query[self.data_indices] , dim = -1 )
        
        # If the voxel is newly covered in query (i.e. not exist in dataset), dot product is 1.
        output = torch.ones( self.n_voxels , device=self.device )

        # Populate output with observed voxel dot products. For dot products corresponding to 
        # the same voxel, we take the min dot product.
        output.scatter_reduce_( 0, self.data_indices, dot_product, reduce='amin' )

        return output       # n_voxel. NOTE: Values will not make sense (i.e. they'll be 0) for
                            # values in query that are dummy variables. 

    def compute_existing_coverage_batched(self, query):
        # query: n_cameras x n_voxels x ndim

        n_cameras = query.shape[0]

        # This is for voxels that exist in the extant dataset.
        dot_product = - torch.sum( self.data_values[None] * query[:, self.data_indices, :] , dim = -1 ) # n_cameras x n_data
        
        # If the voxel is newly covered in query (i.e. not exist in dataset), dot product is 1.
        output = torch.ones( (n_cameras, self.n_voxels) , device=self.device )

        # Populate output with observed voxel dot products. For dot products corresponding to 
        # the same voxel, we take the min dot product.
        output.scatter_reduce_( 1, self.data_indices.reshape(1, -1).expand(n_cameras, -1), dot_product, reduce='amin')  # n_cameras x n_voxels

        return output       # NOTE: Values will not make sense (i.e. they'll be 0) for
                                                            # values in query that are dummy variables. 

    def compute_rollout_coverage(self, rollout):
        # rollout:n_steps x n_cameras x n_voxels x ndim

        n_steps, n_cameras = rollout.shape[:2]

        #NOTE: Because the dataset keeps growing, it would be wise to simply pre-allocate a large
        # tensor rather than incrementally growing the tensor (slow).
        dataset = torch.zeros( (n_steps, n_cameras, self.n_voxels, self.n_dim), device=self.device)


