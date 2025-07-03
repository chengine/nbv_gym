import numpy as np
import open3d as o3d
from munch import Munch


def generate_lunar_scene(config: Munch) -> o3d.geometry.TriangleMesh:
    # Surface
    surface_mesh = None
    if config.map_path is not None:
        dem = np.load(config.map_path, allow_pickle=True)
        surface_mesh = create_mesh_from_grid(dem=dem)
        height_map = dem[..., 2]
        # TODO: Get x, y, z from dem
    elif config.surface is not None:
        x, y, z = generate_surface(config)
        height_map = z
    else:
        raise ValueError("No surface provided")

    # Craters
    if config.craters is not None and config.craters.num > 0:
        config = config.craters
        rng = np.random.RandomState(config.seed)
        N_craters = config.num
        crater_centers = rng.uniform(
            config.xlims[0] + config.radius[1],
            config.xlims[1] - config.radius[1],
            (N_craters, 2),
        )
        crater_radius = rng.uniform(config.radius[0], config.radius[1], N_craters)
        crater_depths = crater_radius * rng.uniform(config.depth[0], config.depth[1], N_craters)

        for center, radius, depth in zip(crater_centers, crater_radius, crater_depths):
            add_crater(x, y, z, center, radius, depth, noise=config.noise)

    # Surface mesh
    mesh_color = np.array(self.config.material.base_color)[:3]
    if self.surface_mesh is None:
        self.surface_mesh = self.create_mesh_from_grid(x, y, z)
    self.surface_mesh.paint_uniform_color(mesh_color)
    self.surface_mesh.compute_vertex_normals()
    self.surface_mesh.compute_triangle_normals()
    self.surface_mesh_tree = scipy.spatial.cKDTree(np.asarray(self.surface_mesh.vertices))
    self.surface_points = np.asarray(self.surface_mesh.vertices)

    # Rocks
    if self.config.rocks is not None and (self.config.rocks.num or self.config.rocks.density):
        config = self.config.rocks
        rng = np.random.RandomState(config.seed)
        if config.num is not None:
            N_rocks = config.num
        elif config.density is not None:
            Dx = self.config.xlims[1] - self.config.xlims[0]
            Dy = self.config.ylims[1] - self.config.ylims[0]
            N_rocks = int(config.density * Dx * Dy)
        else:
            raise ValueError("Either num or density must be provided")

        rock_x = rng.uniform(
            self.config.xlims[0] + config.radius[1],
            self.config.xlims[1] - config.radius[1],
            N_rocks,
        )
        rock_y = rng.uniform(
            self.config.ylims[0] + config.radius[1],
            self.config.ylims[1] - config.radius[1],
            N_rocks,
        )
        rock_radius = rng.uniform(config.radius[0], config.radius[1], N_rocks)
        rock_sink = -rock_radius * rng.uniform(config.sink[0], config.sink[1], N_rocks)
        self.rock_centers = np.vstack((rock_x, rock_y, rock_sink)).T
        self.rocks = []

        for center, radius in Logger.tqdm(
            zip(self.rock_centers, rock_radius),
            total=N_rocks,
            desc="Generating rocks",
            name=self.name,
        ):
            rock = self.generate_rock(
                rng,
                center,
                self.surface_mesh,
                radius=radius,
                noise=config.noise,
                resolution=self.config.resolution,
            )
            self.rocks.append(rock)

        self.rock_map = self.create_rock_grid()
        Logger.info(f"Added {N_rocks} rocks", name=self.name)
    else:
        self.rocks = []

    # Background color
    self.background_color = np.array(self.config.background_color)

    self.scene = o3d.t.geometry.RaycastingScene()
    self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.surface_mesh))
    for rock in self.rocks:
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(rock))


def generate_surface(config: Munch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a lunar surface terrain.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: X, Y coordinates and Z heights of the terrain grid.
    """
    config = config.surface
    length_x = config.xlims[1] - config.xlims[0]
    length_y = config.ylims[1] - config.ylims[0]
    x = np.linspace(config.xlims[0], config.xlims[1], int(length_x / config.resolution))
    y = np.linspace(config.ylims[0], config.ylims[1], int(length_y / config.resolution))
    x, y = np.meshgrid(x, y)

    # Sinusoidal surface with noise
    kx = 2 * np.pi / config.lambda_x
    ky = 2 * np.pi / config.lambda_y
    z = config.amplitude * np.sin(kx * x) * np.cos(ky * y)
    z += config.noise * np.random.randn(*x.shape)
    z += config.slope_x / length_x * x + config.slope_y / length_y * y
    return x, y, z


def create_mesh_from_grid(
    x: np.ndarray = None, y: np.ndarray = None, z: np.ndarray = None, dem: np.ndarray = None
) -> o3d.geometry.TriangleMesh:
    """Create a triangle mesh from a grid of points.

    Args:
        x (np.ndarray, optional): X coordinates of the grid.
        y (np.ndarray, optional): Y coordinates of the grid.
        z (np.ndarray, optional): Z heights of the grid.
        dem (np.ndarray, optional): Digital Elevation Model data.

    Returns:
        o3d.geometry.TriangleMesh: Triangle mesh representing the terrain.
    """
    if dem is not None:
        # Case 1: Input is a single 3D array
        H, W = dem.shape[:2]
        vertices = dem[..., :3].reshape(-1, 3)
        rows, cols = H, W
    else:
        # Case 2: Input is three 2D arrays
        if x is None or y is None or z is None:
            raise ValueError("Either dem or all of x, y, z must be provided")
        vertices = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        rows, cols = x.shape

    # Create triangles
    triangles = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx = i * cols + j
            # First triangle
            triangles.append([idx, idx + cols, idx + 1])
            # Second triangle
            triangles.append([idx + 1, idx + cols, idx + cols + 1])

    # Create and return mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def create_rock_grid(config: Munch, rocks: list[o3d.geometry.TriangleMesh]):
    """Create a grid of rock positions and properties.

    Returns:
        tuple: Grid of rock positions and properties.
    """
    # Extract limits
    x_min, x_max = config.xlims
    y_min, y_max = config.ylims
    res = config.resolution

    # Define grid dimensions
    grid_shape = (
        int(np.ceil((y_max - y_min) / res)),
        int(np.ceil((x_max - x_min) / res)),
    )
    grid = np.zeros(grid_shape, dtype=np.uint8)  # 0 = empty, 1 = rock

    # Convert world (x, y) coordinates to grid indices
    def world_to_grid(x, y):
        i = int((y - y_min) / res)  # Row index
        j = int((x - x_min) / res)  # Col index
        return i, j

    # Rasterize each mesh into the grid
    for mesh in rocks:
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)[:, :2]  # Extract (x, y), ignore z

        for tri in triangles:
            v0, v1, v2 = vertices[tri]  # Get triangle vertices

            # Get bounding box of triangle in grid space
            x_tri_min, y_tri_min = np.min([v0, v1, v2], axis=0)
            x_tri_max, y_tri_max = np.max([v0, v1, v2], axis=0)

            # Convert bounding box to grid indices
            i_min, j_min = world_to_grid(x_tri_min, y_tri_min)
            i_max, j_max = world_to_grid(x_tri_max, y_tri_max)

            # Clamp indices within the grid bounds
            i_min, i_max = max(0, i_min), min(grid.shape[0] - 1, i_max)
            j_min, j_max = max(0, j_min), min(grid.shape[1] - 1, j_max)

            # Mark grid cells as occupied (1)
            grid[i_min : i_max + 1, j_min : j_max + 1] = 1

    return grid


def add_crater(
    x,
    y,
    z,
    center,
    radius,
    depth,
    noise=0.01,
    center_width=0.8,
    rim_height=0.05,
) -> None:
    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = distance <= radius
    d_norm = distance[mask] / radius

    def f_crater(x):
        return -1 + np.exp(-((1 - x / center_width) ** 2) * 5)

    def f_rim(x):
        return rim_height * (-1 + np.exp(-((x - center_width) ** 2) * 100))

    crater_profile = f_crater(d_norm)
    crater_profile[d_norm > center_width] = f_rim(d_norm[d_norm > center_width])
    crater_profile -= f_crater(0)
    crater_profile /= f_rim(1.0) - f_crater(0)
    crater_profile -= 1
    crater_profile *= depth
    crater_profile += noise * np.random.randn(*crater_profile.shape)
    z[mask] += crater_profile
