import torch
import torch.nn.functional as F


def upsample_dem_torch(dem_xyz: torch.Tensor, scale: int) -> torch.Tensor:
    # dem_xyz: [N, N, 3] → [1, 3, N, N]
    dem = dem_xyz.permute(2, 0, 1).unsqueeze(0)
    upsampled = F.interpolate(dem, scale_factor=scale, mode="bilinear", align_corners=True)
    return upsampled.squeeze(0).permute(1, 2, 0)  # [scale*N, scale*N, 3]


def bilinear_interpolate(grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    grid: Tensor of shape [H, W] — the scalar field
    coords: Tensor of shape [N, 2] — xy coordinates to sample, in pixel space (not normalized)

    Returns:
        interpolated values: Tensor of shape [N]
    """
    H, W = grid.shape
    x = coords[:, 0]
    y = coords[:, 1]

    # Clamp coords to be within image bounds
    x0 = torch.floor(x).long().clamp(0, W - 2)
    x1 = x0 + 1
    y0 = torch.floor(y).long().clamp(0, H - 2)
    y1 = y0 + 1

    # Get the values at the corner pixels
    Ia = grid[y0, x0]
    Ib = grid[y1, x0]
    Ic = grid[y0, x1]
    Id = grid[y1, x1]

    # Compute interpolation weights
    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


class DEM:
    """Digital Elevation Model that supports bilinear interpolation queries."""

    def __init__(self, points: torch.Tensor):
        """Initialize DEM with grid of xyz points.

        Args:
            points: Tensor of shape (N, N, 3) containing xyz coordinates
        """
        assert (
            len(points.shape) == 3 and points.shape[0] == points.shape[1] and points.shape[2] == 3
        )
        self.points = points
        self.size = points.shape[0]
        self.device = points.device

    def get_heights(self, query_points: torch.Tensor) -> torch.Tensor:
        """Get interpolated heights for query points.

        Args:
            query_points: Tensor of shape (N, 2) containing xy coordinates in position space

        Returns:
            Tensor of shape (N,) containing interpolated z values
        """
        # Convert position space coordinates to array indices in [0, size-1] range
        x = (query_points[:, 0] + 1) * (self.size - 1) / 2
        y = (query_points[:, 1] + 1) * (self.size - 1) / 2

        # Get integer coordinates for bilinear interpolation
        x0 = torch.floor(x).long()
        x1 = torch.min(x0 + 1, torch.tensor(self.size - 1, device=self.device))
        y0 = torch.floor(y).long()
        y1 = torch.min(y0 + 1, torch.tensor(self.size - 1, device=self.device))

        # Get weights
        wx = x - x0
        wy = y - y0

        # Get corner values
        c00 = self.points[y0, x0, 2]  # z values only
        c01 = self.points[y0, x1, 2]
        c10 = self.points[y1, x0, 2]
        c11 = self.points[y1, x1, 2]

        # Bilinear interpolation
        c0 = c00 * (1 - wx) + c01 * wx
        c1 = c10 * (1 - wx) + c11 * wx
        return c0 * (1 - wy) + c1 * wy

    def sample_points(self, num_points: int) -> torch.Tensor:
        """Sample points from DEM.

        Args:
            num_points: Number of points to sample

        Returns:
            Tensor of shape (num_points, 3) containing sampled xyz coordinates in position space
        """
        # Sample random xy coordinates in position space [-1, 1]
        xy = 2 * torch.rand(num_points, 2, device=self.device) - 1

        # Get interpolated z values
        z = self.get_heights(xy)

        # Combine into xyz points
        points = torch.cat([xy, z.unsqueeze(-1)], dim=-1)

        return points
