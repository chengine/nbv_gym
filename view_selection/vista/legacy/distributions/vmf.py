import torch
import numpy as np

# Closed form of the ratio of modified Bessel functions of the first kind of order 3/2 and 1/2
def A_kappa(kappa):
    kappa_nonzero_mask = (kappa > 0.)

    output = torch.zeros_like(kappa)
    kappa_nonzero = kappa[kappa_nonzero_mask]

    output[kappa_nonzero_mask] = 1./ torch.tanh(kappa_nonzero) - 1./ kappa_nonzero #(torch.cosh(kappa_nonzero) - torch.sinh(kappa_nonzero) / kappa_nonzero) / torch.sinh(kappa_nonzero)

    return output

class FisherDistribution:
    def __init__(self, device, n_clusters):
        self.device = device
        self.p = 3
        self.n_clusters = n_clusters
        self.fisher_kappa = torch.zeros(n_clusters, device=self.device)  # uniform
        self.fisher_mean = torch.zeros((n_clusters, self.p), device=self.device)

    @torch.compile
    # NOTE: IF YOU PASS IN ZERO VECTORS (OR UNORMALIZED VECTORS), THIS FUNCTION WILL RETURN UNINTENDED PARAMETERS.
    def fit(self, data, index, num_iters=10):
        # data: N x D. 3D unit vectors 
        # index: N. Index of the cluster each data point belongs to.
        # Returns: K x D. K is the number of clusters.

        assert torch.linalg.norm(data, dim=-1).allclose(torch.ones(data.shape[0], device=self.device)), "Data must be unit vectors."

        cluster_sum = torch.zeros((self.n_clusters, 3), device=self.device)
        cluster_counts = torch.zeros(self.n_clusters, device=self.device, dtype=torch.int32)
        cluster_sum.index_add_(0, index, data)
        cluster_counts.index_add_(0, index, torch.ones_like(index, device=self.device))

        # only compute the mean for clusters that have at least one point
        mask = cluster_counts > 0
        cluster_sum = cluster_sum[mask]
        cluster_counts = cluster_counts[mask]

        cluster_mean = cluster_sum / cluster_counts[:, None]
        cluster_mean_norm = torch.norm(cluster_mean, dim=-1, keepdim=True)

        # Calculates the concentration parameter
        R = torch.clamp(cluster_mean_norm, max=0.99).squeeze()        # clamp to avoid NANs when we only have one point in the cluster

        kappa = R * (self.p - R**2) / (1. - R**2)       # Initialization
        for i in range(num_iters):
            # Calculate the function I_3/2 / I_1/2, where I is the modified Bessel function of the first kind of order 3/2 or 1/2. For 
            # Fisher distribution, this has the closed form of (sqrt(2/(pi*kappa)) * (cosh(kappa) - sinh(kappa)/kappa)) / (sqrt(2/(pi*kappa)) * sinh(kappa)) = (cosh(kappa) - sinh(kappa)/kappa) / sinh(kappa)
            A = A_kappa(kappa)
            #print(kappa.min(), kappa.max(), A.min(), A.max())

            # Update kappa estimate
            kappa -= (A - R) / (1. - A**2 - (self.p - 1.) / kappa * A)

        self.fisher_mean = torch.zeros((self.n_clusters, 3), device=self.device)
        self.fisher_kappa = torch.zeros(self.n_clusters, device=self.device)

        # Calculates the VMF mean direction
        self.fisher_mean[mask] = cluster_mean / cluster_mean_norm 

        # Calculates the concentration parameter
        self.fisher_kappa[mask] = torch.clamp(kappa, min=0., max=5.)        # Clamp kappa to avoid numerical instability

        # print(self.fisher_mean)
        #print(self.fisher_kappa)

        return self.fisher_mean, self.fisher_kappa

    @torch.compile
    def evaluate(self, input):
        # input: K x D. 3D unit vectors.
        # Returns: K. The probability of each data point in input belonging to each cluster.
        # NOTE: We need to be very careful with numerical stability here since we are dealing with exponentials.

        # The Fisher density for kappa = 0 evaluates to 1 / 4 pi
        kappa_nonzero_mask = (self.fisher_kappa > 0.)
        output = (1./ (4*np.pi)) * torch.ones(input.shape[0], device=self.device)

        kappa_nonzero = self.fisher_kappa[kappa_nonzero_mask]
        output[kappa_nonzero_mask] = kappa_nonzero * torch.exp(-kappa_nonzero * (1. - torch.sum(input[kappa_nonzero_mask] * self.fisher_mean[kappa_nonzero_mask], dim=-1))) / (2*np.pi * (1. - torch.exp(-2. * kappa_nonzero)))
        
        return output

    # TODO: ALLOW FOR UPDATES TO THE DISTRIBUTION
    
class FisherKLDivergence:
    def __init__(self, device, n_clusters):
        self.device = device
        self.n_clusters = n_clusters

        self.dist0 = FisherDistribution(device, n_clusters)
        self.dist1 = FisherDistribution(device, n_clusters)

    # @torch.compile
    # def fit(self, data, index, num_iters=10):
    #     # data: 2 x N x D. 3D unit vectors 
    #     # index: 2 x N. Index of the cluster each data point belongs to.
    #     # n_clusters: Number of clusters.

    #     # NOTE: NONE indicates we don't initialize, so fit a uniform distribution
    #     if data[0] is not None:
    #         self.dist0.fit(data[0], index[0], num_iters)

    #     if data[1] is not None:
    #         self.dist1.fit(data[1], index[1], num_iters)

    #     return 
    
    @torch.compile
    def fit_dist0(self, data, index, num_iters=10):
        if data is not None:
            self.dist0.fit(data, index, num_iters)

    @torch.compile
    def fit_dist1(self, data, index, num_iters=10):
        if data is not None:
            self.dist1.fit(data, index, num_iters)

    @torch.compile
    def evaluate(self):
        # Returns: K. The KL divergence for each cluster between the two Fisher distributions.
        kappa0 = self.dist0.fisher_kappa
        mean0 = self.dist0.fisher_mean

        input = A_kappa(kappa0)[..., None] * mean0

        prob0 = self.dist0.evaluate(input)
        prob1 = self.dist1.evaluate(input)

        return torch.log(prob0 / prob1)