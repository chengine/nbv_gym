from typing import Any, Dict

import numpy as np
import open3d as o3d
import scipy.spatial

from lunar_slam.applications.segmentation import Labels
from lunar_slam.environments.environment import Environment
from lunar_slam.utils.config import Config, load_config
from lunar_slam.utils.frames import OCV_T_FLU
from lunar_slam.utils.logger import Logger
from lunar_slam.utils.math import dilate_mask
from lunar_slam.utils.path import world2grid


class Open3DEnvironment:
    """Open3D-based lunar environment implementation.

    This class implements a lunar environment using Open3D for 3D rendering and mesh manipulation.
    It provides a realistic simulation of lunar terrain with features such as:
    - Procedural terrain generation
    - Crater placement and modification
    - Rock placement and collision detection
    - Dynamic lighting based on sun position
    - Real-time rendering with Open3D

    The environment can be configured through a YAML file that specifies:
    - Terrain parameters (size, resolution, features)
    - Lighting conditions (sun position, intensity)
    - Material properties (surface color, rock properties)
    - Feature placement (craters, rocks)

    Attributes:
        default_config (str): Path to the default configuration file.
        config (Config): Environment configuration containing all parameters.
        d_light (np.ndarray): Direction vector of the light source in world coordinates.
        renderers (dict): Dictionary of renderers for different resolutions, keyed by (height, width).
        surface_mesh (o3d.geometry.TriangleMesh): Mesh representing the lunar surface.
        height_map (np.ndarray): Height map of the terrain in grid coordinates.
        rock_centers (np.ndarray): Centers of placed rocks in world coordinates.
        rocks (list): List of rock meshes as Open3D TriangleMesh objects.
        background_color (np.ndarray): RGB color for the environment background.

    Example:
        >>> config = load_config("configs/env.yaml")
        >>> env = Open3DEnvironment(config)
        >>> camera = Camera(config.camera)
        >>> rendered = env.render(camera)
    """

    default_config = "configs/env.yaml::open3d_env"

    def __init__(self, config: Config = default_config):
        self.config = load_config(config)
        self.name = self.__class__.__name__

        # Direction that it comes from
        self.d_light = get_light_direction(
            np.deg2rad(self.config.sun.az), np.deg2rad(self.config.sun.el)
        )

        # Renderers
        self.renderers = {}

        # Surface
        self.surface_mesh = None
        if self.config.map_path is not None:
            dem = np.load(self.config.map_path, allow_pickle=True)
            self.surface_mesh = Open3DEnvironment.create_mesh_from_grid(dem=dem)
            self.height_map = dem[..., 2]
            # TODO: Get x, y, z from dem
        elif self.config.surface is not None:
            x, y, z = self.generate_surface()
            self.height_map = z
        else:
            raise ValueError("No surface provided")

        Logger.info(
            f"Added surface with xlims={self.config.xlims} m, ylims={self.config.ylims} m, resolution={self.config.resolution} m",
            name=self.name,
        )

        # Craters
        if self.config.craters is not None and self.config.craters.num > 0:
            config = self.config.craters
            rng = np.random.RandomState(config.seed)
            N_craters = config.num
            crater_centers = rng.uniform(
                self.config.xlims[0] + config.radius[1],
                self.config.xlims[1] - config.radius[1],
                (N_craters, 2),
            )
            crater_radius = rng.uniform(config.radius[0], config.radius[1], N_craters)
            crater_depths = crater_radius * rng.uniform(config.depth[0], config.depth[1], N_craters)

            for center, radius, depth in zip(crater_centers, crater_radius, crater_depths):
                add_crater(x, y, z, center, radius, depth, noise=config.noise)

            Logger.info(f"Added {N_craters} craters", name=self.name)

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

    def set_sun_position(self, azimuth, elevation):
        self.d_light = get_light_direction(np.deg2rad(azimuth), np.deg2rad(elevation))

    def generate_surface(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a lunar surface terrain.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: X, Y coordinates and Z heights of the terrain grid.
        """
        config = self.config.surface
        length_x = self.config.xlims[1] - self.config.xlims[0]
        length_y = self.config.ylims[1] - self.config.ylims[0]
        x = np.linspace(
            self.config.xlims[0], self.config.xlims[1], int(length_x / self.config.resolution)
        )
        y = np.linspace(
            self.config.ylims[0], self.config.ylims[1], int(length_y / self.config.resolution)
        )
        x, y = np.meshgrid(x, y)

        # Sinusoidal surface with noise
        kx = 2 * np.pi / config.lambda_x
        ky = 2 * np.pi / config.lambda_y
        z = config.amplitude * np.sin(kx * x) * np.cos(ky * y)
        z += config.noise * np.random.randn(*x.shape)
        z += config.slope_x / length_x * x + config.slope_y / length_y * y
        return x, y, z

    def generate_rock(
        self, rng, center, surface_mesh, radius=1.0, noise=0.1, resolution=0.05
    ) -> o3d.geometry.TriangleMesh:
        """Generate a rock mesh and add it to the surface.

        Args:
            rng (np.random.RandomState): Random number generator.
            center (np.ndarray): Center position of the rock.
            surface_mesh (o3d.geometry.TriangleMesh): Surface mesh to place the rock on.
            radius (float, optional): Base radius of the rock. Defaults to 1.0.
            noise (float, optional): Amount of noise to add to the rock surface. Defaults to 0.1.
            num_points (int, optional): Number of points to generate the rock mesh. Defaults to 100.

        Returns:
            o3d.geometry.TriangleMesh: Generated rock mesh.
        """
        rock_area = 4 * np.pi * radius**2
        num_points = int(rock_area / (resolution**2))

        # Generate sphere mesh
        rock_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        vertices = np.asarray(rock_mesh.vertices)
        vertices += radius * rng.uniform(-noise, noise, vertices.shape)
        vertices *= rng.uniform(0.5, 1.5, 3)
        R = o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rng.uniform(-np.pi, np.pi, 3))
        vertices = np.dot(vertices, R.T)
        rock_mesh.vertices = o3d.utility.Vector3dVector(vertices)

        # Translate the rock to rest on the surface
        idx_closest = get_closest_point_on_surface(center[:2], self.surface_mesh)
        z_closest = self.surface_points[idx_closest][2]
        rock_mesh.translate([center[0], center[1], center[2] - np.min(vertices[:, 2]) + z_closest])

        # Squish bottom half of the rock
        rock_vertices = np.asarray(rock_mesh.vertices)
        _, idx = self.surface_mesh_tree.query(rock_vertices)
        surface_z = self.surface_points[idx][:, 2]
        below_mask = rock_vertices[:, 2] < surface_z
        rock_vertices[below_mask, 2] = surface_z[below_mask]
        rock_mesh.vertices = o3d.utility.Vector3dVector(rock_vertices)

        # Finalize
        rock_mesh.paint_uniform_color(self.config.rocks.color)
        rock_mesh.compute_vertex_normals()
        rock_mesh.compute_triangle_normals()
        return rock_mesh

    def render(self, camera: Dict[str, Any], semantic=True):
        """Render the environment from a camera's perspective.

        Args:
            camera (Camera): Camera to render from.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing rendered images and metadata.
        """
        # Pose
        flu_T_ocv = np.linalg.inv(OCV_T_FLU)
        world_T_cam_ocv = camera.world_T_cam.cpu().numpy() @ flu_T_ocv
        cam_ocv_T_world = np.linalg.inv(world_T_cam_ocv)

        # Camera
        cam = o3d.camera.PinholeCameraParameters()
        cam.extrinsic = cam_ocv_T_world
        cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            camera.image_width, camera.image_height, camera.fx, camera.fy, camera.cx, camera.cy
        )

        # RGB and depth
        renderer = self.get_renderer(camera.image_height, camera.image_width)
        renderer.setup_camera(cam.intrinsic, cam.extrinsic)
        image = np.asarray(renderer.render_to_image())
        depth = np.asarray(renderer.render_to_depth_image(z_in_view_space=True))

        # Semantic
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=cam.intrinsic.intrinsic_matrix,
            extrinsic_matrix=cam.extrinsic,
            width_px=camera.image_width,
            height_px=camera.image_height,
        )
        ans = self.scene.cast_rays(rays)
        ids = ans["geometry_ids"].numpy()
        semantic = np.full(
            (camera.image_height, camera.image_width),
            fill_value=Labels.SKY.value,
            dtype=np.uint8,
        )
        semantic[ids == 0] = Labels.REGOLITH.value
        semantic[(ids > 0) & (ids <= len(self.rocks))] = Labels.ROCK.value

        return {"rgb": image, "depth": depth, "label": semantic, **ans}

    def get_renderer(self, height, width):
        """Get or create a renderer for the specified resolution.

        Args:
            height (int): Height of the renderer viewport.
            width (int): Width of the renderer viewport.

        Returns:
            o3d.visualization.rendering.OffscreenRenderer: Renderer instance.
        """
        # Material
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"
        material.base_color = np.array(self.config.material.base_color)
        material.base_roughness = self.config.material.base_roughness
        material.base_metallic = self.config.material.base_metallic
        material.base_reflectance = self.config.material.base_reflectance

        # Renderer
        if self.renderers.get((height, width)) is not None:
            return self.renderers[(height, width)]

        renderer = o3d.visualization.rendering.OffscreenRenderer(width=width, height=height)
        renderer.scene.clear_geometry()
        for i, mesh in enumerate([self.surface_mesh] + self.rocks):
            assert mesh is not None
            renderer.scene.add_geometry(f"mesh_{i}", mesh, material)
        renderer.scene.set_lighting(renderer.scene.LightingProfile.HARD_SHADOWS, self.d_light)
        renderer.scene.set_background(np.array(self.config.background_color))
        self.renderers[(height, width)] = renderer
        Logger.info(f"Created {height}x{width} renderer", name=self.name)
        return renderer

    def get_agent_pose(self, xy_in, h):
        """Get the pose of an agent at a given position.

        Args:
            xy (np.ndarray): 2D position of the agent.
            h (float, optional): Height of the agent above the surface. Defaults to 1.0.

        Returns:
            np.ndarray: 4x4 transformation matrix representing the agent's pose.
        """
        if xy_in.ndim == 1:
            xy_in = xy_in[None, :]

        world_T_body = np.zeros((len(xy_in), 4, 4))

        for i, xy in enumerate(xy_in):
            r_cam = np.array([xy[0], xy[1], 0.0])
            idx_closest = get_closest_point_on_surface(r_cam, self.surface_mesh)
            z_closest = self.surface_mesh.vertices[idx_closest][2]
            n_closest = -self.surface_mesh.vertex_normals[idx_closest]
            r_cam[2] = z_closest
            r_cam += h * n_closest

            if len(xy) == 3:
                theta = xy[2]
                ex = np.array([np.cos(theta), np.sin(theta), 0.0])
                ez = n_closest / np.linalg.norm(n_closest)
                ey = np.cross(ez, ex)
                ey /= np.linalg.norm(ey)
                ex = np.cross(ey, ez)
                ex /= np.linalg.norm(ex)
                R = np.vstack((ex, ey, ez)).T
            else:
                R = np.eye(3)

            world_T_body[i] = np.eye(4)
            world_T_body[i, :3, :3] = R
            world_T_body[i, :3, 3] = r_cam

        if len(xy_in) == 1:
            return world_T_body[0]

        return world_T_body

    def create_rock_grid(self):
        """Create a grid of rock positions and properties.

        Returns:
            tuple: Grid of rock positions and properties.
        """
        # Extract limits
        x_min, x_max = self.config.xlims
        y_min, y_max = self.config.ylims
        res = self.config.resolution

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
        for mesh in self.rocks:
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

    def check_collision(self, position: np.ndarray, radius: float) -> bool:
        """Check if a position collides with any rocks.

        Args:
            position (np.ndarray): Position to check.
            radius (float): Radius to check for collisions.

        Returns:
            bool: True if there is a collision, False otherwise.
        """
        if self.rock_map is None:
            return False
        rock_map_mask = dilate_mask(self.rock_map, radius, self.config.resolution)
        xy_grid = world2grid(
            position[:2], self.config.xlims, self.config.ylims, self.config.resolution
        )
        return rock_map_mask[xy_grid[1], xy_grid[0]]

    def get_height_at(self, position: np.ndarray) -> float:
        """Get the height at a given position.

        Args:
            position (np.ndarray): Position to get the height at.

        Returns:
            float: Height at the given position.
        """
        if not isinstance(position, np.ndarray):
            position = np.array(position)

        ndim = position.ndim
        if ndim == 1:
            position = position[None, :]

        surface_vert = np.asarray(self.surface_mesh.vertices)
        tree = scipy.spatial.cKDTree(surface_vert[:, :2])
        _, idx = tree.query(position[:, :2])

        if ndim == 1:
            return surface_vert[idx[0], 2]

        return surface_vert[idx, 2]


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


def get_closest_point_on_surface(r: np.ndarray, surface_mesh: o3d.geometry.TriangleMesh) -> int:
    """Find the closest point on the surface mesh to a given point.

    Args:
        r (np.ndarray): Point to find the closest surface point to.
        surface_mesh (o3d.geometry.TriangleMesh): Surface mesh to search.

    Returns:
        int: Index of the closest vertex in the mesh.
    """
    terrain_vertices = np.asarray(surface_mesh.vertices)
    distances = np.linalg.norm(terrain_vertices[:, :2] - r[:2], axis=1)
    idx = np.argmin(distances)
    return idx


def get_light_direction(az: float, el: float) -> np.ndarray:
    """Calculate the direction vector of light from azimuth and elevation angles.

    Args:
        az (float): Azimuth angle in radians.
        el (float): Elevation angle in radians.

    Returns:
        np.ndarray: Normalized direction vector of the light.
    """
    return -1e4 * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
