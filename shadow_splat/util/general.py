import torch
import numpy as np


def generate_plane_points(width: float = 1.0, height: float = 1.0, num_points: int = 100):
    """
    Generate points on a plane.
    """
    x = torch.linspace(-width / 2, width / 2, num_points)
    y = torch.linspace(-height / 2, height / 2, num_points)
    x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
    plane_points = torch.stack([x_grid, y_grid, torch.zeros_like(x_grid)], dim=-1).reshape(-1, 3)
    return plane_points


def generate_cylinder_points():
    """
    Generate points on a cylinder surface.
    """
    num_surface_points = 100  # Number of points around the cylinder surface
    num_layers = 200  # Number of layers in z direction
    num_interior_points = 10000  # Number of interior points
    radius = 0.25  # Cylinder radius
    z_start = 0.0  # Starting z coordinate
    z_end = 0.5  # Ending z coordinate

    # Create surface points
    angles = torch.linspace(0, 2 * torch.pi, num_surface_points)
    z_coords = torch.linspace(z_start, z_end, num_layers)
    angles_grid, z_grid = torch.meshgrid(angles, z_coords, indexing="ij")
    x_coords = radius * torch.cos(angles_grid)
    y_coords = radius * torch.sin(angles_grid)
    surface_points = torch.stack([x_coords, y_coords, z_grid], dim=-1).reshape(-1, 3)

    # Create interior points using rejection sampling
    interior_points = []
    while len(interior_points) < num_interior_points:
        # Generate random points in a cube
        x = torch.rand(num_interior_points) * 2 * radius - radius
        y = torch.rand(num_interior_points) * 2 * radius - radius
        z = torch.rand(num_interior_points) * (z_end - z_start) + z_start

        # Stack coordinates
        points = torch.stack([x, y, z], dim=-1)

        # Keep only points inside the cylinder (x^2 + y^2 <= radius^2)
        mask = (points[:, 0] ** 2 + points[:, 1] ** 2) <= radius**2
        interior_points.append(points[mask])

    interior_points = torch.cat(interior_points, dim=0)[:num_interior_points]

    # Combine surface and interior points
    cylinder_points = torch.cat([surface_points, interior_points], dim=0)
    return cylinder_points


def fibonacci_hemisphere_angles(N, min_elevation_rad=0.0):
    """returns elevation and azimuth in radians"""
    golden_angle = np.pi * (3 - np.sqrt(5))
    indices = np.arange(N)
    azimuth = (indices * golden_angle) % (2 * np.pi)

    z_min = np.sin(min_elevation_rad)
    z = np.linspace(z_min, 1.0, N)
    elevation = np.arcsin(z)

    return elevation, azimuth


def fibonacci_hemisphere_points(N, radius=1.0, min_elevation_rad=0.0):
    """returns elevation and azimuth in radians"""
    elevation, azimuth = fibonacci_hemisphere_angles(N, min_elevation_rad)
    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)
    return np.stack([x, y, z], axis=-1) * radius


def generate_ply_from_points(
    points: np.ndarray, colors: np.ndarray, filename: str = "pc.ply"
) -> None:
    """Generate PLY file from points for gaussian splatting initialization"""
    # Check that both points and colors are Nx3 arrays
    assert points.shape[1] == 3, "Points array should have shape Nx3"
    assert colors.shape[1] == 3, "Colors array should have shape Nx3"
    assert points.shape[0] == colors.shape[0], (
        "Points and colors arrays must have the same number of rows"
    )

    num_vertices = points.shape[0]

    # Open file for writing
    with open(filename, "w") as ply_file:
        # Write PLY header
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {num_vertices}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uint8 red\n")
        ply_file.write("property uint8 green\n")
        ply_file.write("property uint8 blue\n")
        ply_file.write("end_header\n")

        # Write vertex data (points and colors)
        for point, color in zip(points, colors):
            x, y, z = point
            r, g, b = color.astype(int)  # Convert color to int if not already
            ply_file.write(f"{x} {y} {z} {r} {g} {b}\n")
