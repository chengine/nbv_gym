"""
Class for Blender data generation.
"""

from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix


class BlenderScene:
    def __init__(self, scene_path: str | Path):
        """
        Initialize the Blender class.

        Args:
            scene_path: Path to the scene file.
        """
        bpy.ops.wm.open_mainfile(filepath=scene_path)
        self.scene = bpy.context.scene

        # Render settings
        self.scene.render.engine = "CYCLES"
        self.scene.render.film_transparent = True

        # Camera settings
        self.camera = bpy.context.scene.camera
        self.camera.data.lens = 35  # 35 mm (~52.5 deg FOV)

    def render(self, file_path: str | Path):
        """Render the scene and save the image to the given file path."""
        self.scene.render.filepath = file_path
        bpy.ops.render.render(write_still=True)

    def set_camera_pose(self, pose: np.ndarray):
        """Set the camera pose (OpenGL convention)."""
        self.camera.matrix_world = Matrix(pose)

    def get_depth(self):
        pass

    def get_albedo(self):
        pass

    def setup_depth(self):
        """
        Setup depth rendering.

        Mostly taken from https://github.com/weiaicunzai/blender_shapenet_render/blob/master/render_depth.py#L68
        """
        self.scene.use_nodes = True
        tree = self.scene.node_tree
        tree.nodes.clear()
        self.scene.view_layers[0].use_pass_z = True

        g_depth_color_mode = "BW"
        g_depth_color_depth = "16"
        g_depth_file_format = "PNG"
        g_depth_clip_start = 0.5
        g_depth_clip_end = 4

        for node in tree.nodes:
            tree.nodes.remove(node)

        render_layer_node = tree.nodes.new("CompositorNodeRLayers")
        map_value_node = tree.nodes.new("CompositorNodeMapValue")
        file_output_node = tree.nodes.new("CompositorNodeOutputFile")

        map_value_node.offset[0] = -g_depth_clip_start
        map_value_node.size[0] = 1 / (g_depth_clip_end - g_depth_clip_start)
        map_value_node.use_min = True
        map_value_node.use_max = True
        map_value_node.min[0] = 0.0
        map_value_node.max[0] = 1.0

        file_output_node.format.color_mode = g_depth_color_mode
        file_output_node.format.color_depth = g_depth_color_depth
        file_output_node.format.file_format = g_depth_file_format
        file_output_node.base_path = "."

        tree.links.new(render_layer_node.outputs[2], map_value_node.inputs[0])
        tree.links.new(map_value_node.outputs[0], file_output_node.inputs[0])
