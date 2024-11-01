"""Utilities for blender data generation

References:
[1] https://github.com/maximeraafat/BlenderNeRF 

"""

import os


# Taken from [1]
def get_camera_intrinsics(scene, camera):
    camera_angle_x = camera.data.angle_x
    camera_angle_y = camera.data.angle_y

    # camera properties
    f_in_mm = camera.data.lens # focal length in mm
    scale = scene.render.resolution_percentage / 100
    width_res_in_px = scene.render.resolution_x * scale # width
    height_res_in_px = scene.render.resolution_y * scale # height
    optical_center_x = width_res_in_px / 2
    optical_center_y = height_res_in_px / 2

    # pixel aspect ratios
    size_x = scene.render.pixel_aspect_x * width_res_in_px
    size_y = scene.render.pixel_aspect_y * height_res_in_px
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    # sensor fit and sensor size (and camera angle swap in specific cases)
    if camera.data.sensor_fit == 'AUTO':
        sensor_size_in_mm = camera.data.sensor_height if width_res_in_px < height_res_in_px else camera.data.sensor_width
        if width_res_in_px < height_res_in_px:
            sensor_fit = 'VERTICAL'
            camera_angle_x, camera_angle_y = camera_angle_y, camera_angle_x
        elif width_res_in_px > height_res_in_px:
            sensor_fit = 'HORIZONTAL'
        else:
            sensor_fit = 'VERTICAL' if size_x <= size_y else 'HORIZONTAL'

    else:
        sensor_fit = camera.data.sensor_fit
        if sensor_fit == 'VERTICAL':
            sensor_size_in_mm = camera.data.sensor_height if width_res_in_px <= height_res_in_px else camera.data.sensor_width
            if width_res_in_px <= height_res_in_px:
                camera_angle_x, camera_angle_y = camera_angle_y, camera_angle_x

    # focal length for horizontal sensor fit
    if sensor_fit == 'HORIZONTAL':
        sensor_size_in_mm = camera.data.sensor_width
        s_u = f_in_mm / sensor_size_in_mm * width_res_in_px
        s_v = f_in_mm / sensor_size_in_mm * width_res_in_px * pixel_aspect_ratio

    # focal length for vertical sensor fit
    if sensor_fit == 'VERTICAL':
        s_u = f_in_mm / sensor_size_in_mm * width_res_in_px / pixel_aspect_ratio
        s_v = f_in_mm / sensor_size_in_mm * width_res_in_px

    camera_intr_dict = {
        'camera_angle_x': camera_angle_x,
        'camera_angle_y': camera_angle_y,
        'fl_x': s_u,
        'fl_y': s_v,
        'k1': 0.0,
        'k2': 0.0,
        'p1': 0.0,
        'p2': 0.0,
        'cx': optical_center_x,
        'cy': optical_center_y,
        'w': width_res_in_px,
        'h': height_res_in_px,
        'aabb_scale': scene.aabb
    }

    return {'camera_angle_x': camera_angle_x} if scene.nerf else camera_intr_dict


# Taken from [1]
def get_camera_extrinsics(scene, camera, mode='TRAIN', method='SOF'):
    assert mode == 'TRAIN' or mode == 'TEST'
    assert method == 'SOF' or method == 'TTC' or method == 'COS'

    if scene.splats and scene.splats_test_dummy and mode == 'TEST':
        return []

    initFrame = scene.frame_current
    step = scene.train_frame_steps if (mode == 'TRAIN' and method == 'SOF') else scene.frame_step
    if (mode == 'TRAIN' and method == 'COS'):
        end = scene.frame_start + scene.cos_nb_frames - 1
    elif (mode == 'TRAIN' and method == 'TTC'):
        end = scene.frame_start + scene.ttc_nb_frames - 1
    else:
        end = scene.frame_end

    camera_extr_dict = []
    for frame in range(scene.frame_start, end + 1, step):
        scene.frame_set(frame)
        filename = os.path.basename( scene.render.frame_path(frame=frame) )
        filedir = OUTPUT_TRAIN * (mode == 'TRAIN') + OUTPUT_TEST * (mode == 'TEST')

        frame_data = {
            'file_path': os.path.join(filedir, os.path.splitext(filename)[0] if scene.splats else filename),
            'transform_matrix': listify_matrix(camera.matrix_world)
        }

        camera_extr_dict.append(frame_data)

    scene.frame_set(initFrame) # set back to initial frame

    return camera_extr_dict


# Taken from [1]
def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list