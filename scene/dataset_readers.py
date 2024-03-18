#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
import math
from tqdm import tqdm
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    white_background: bool = False #For dataloader = list, Move to camera_utils.dataset

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, dataset_type):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        if dataset_type.lower() == 'list':
            image = Image.open(image_path)
        else:
            image = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path, max_num=None):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = None
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = None
    if max_num is not None:
        stride = max(1, int(positions.shape[0] / max_num))
        positions = positions[::stride]
        colors = colors[::stride]
        normals = normals[::stride]
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8, split_file=None, train_num_camera_ratio=None,
                        focal_length_scale=1.0, minus_depth=0.0, dataset_type="list", colmap_pcd="", max_pcd_num=1e10):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        cam_intrinsics[1].params[0:2] *= focal_length_scale
        for k in cam_extrinsics:
            cam_extrinsics[k].tvec[2] -= minus_depth
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                           images_folder=os.path.join(path, reading_dir), dataset_type=dataset_type)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        if split_file:
            name2cam_infos = {c.image_name: c for c in cam_infos}
            with open(split_file) as json_file:
                contents = json.load(json_file) #['train', 'test']
                train_cam_infos = [name2cam_infos[image_name] for image_name in contents['train']]
                test_cam_infos = {}
                for test_k, test_ids in contents.items():
                    if 'test' in test_k:
                        test_cam_infos[test_k] = [name2cam_infos[image_name] for image_name in test_ids] 
            print(f'Use Split file {split_file} ...')
            print('Train cameras:', len(train_cam_infos), 'Test cameras:', len(test_cam_infos))
        elif train_num_camera_ratio != None:
            step = math.floor(1/train_num_camera_ratio)
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % step==0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % step!=0]
        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if colmap_pcd is not "":
        ply_path = colmap_pcd
        print("Load point3d from ", ply_path,'number of points', max_pcd_num)
    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path, max_num=max_pcd_num)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", train_num_camera_ratio=1, dataset_type="list"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents.get("camera_angle_x", None)

        frames = contents["frames"]
        if train_num_camera_ratio!=1:
            step = math.floor(1/train_num_camera_ratio)
            frames = frames[::step]

        for idx, frame in tqdm(enumerate(frames)):
            if extension in frame["file_path"]:
                frame["file_path"] = frame["file_path"].replace(extension, "")
            cam_name = os.path.join(path, frame["file_path"] + extension)

            if 'camera_angle_x' in frame:
                fovx = frame['camera_angle_x'] # Different fov for each image

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            if 'image_name' in frame:
                image_name = os.path.splitext(frame['image_name'])[0]
            else:
                image_name = Path(cam_name).stem

            if dataset_type.lower() == 'list' or idx==0:
                image = Image.open(image_path)

                im_data = np.array(image.convert("RGBA"))

                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
                width=image.size[0]
                height=image.size[1]
                fovy = focal2fov(fov2focal(fovx, width), height) #We assume the same width, height for all images
                FovY = fovy 
                FovX = fovx
            
            if dataset_type.lower() == 'loader':
                if idx==0:
                    print('Currently we assume all images have the same height and width')
                image = None 
                #width, height = None, None
                FovY = focal2fov(fov2focal(fovx, width), height)
                FovX = fovx


            cam_infos.append(CameraInfo(uid=idx, R=R, T=T,FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=width, height=height, white_background=white_background))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png",train_num_camera_ratio=1, 
                        blender_train_json=None,
                        blender_test_jsons=None, dataset_type="list", blender_bbox=[1.3],
                        sample_from_pcd=''):
    
    train_json_file = blender_train_json if blender_train_json is not None else "transforms_train.json"
    print(f"Reading Training Transforms from {train_json_file} ", end=' ')
    train_cam_infos = readCamerasFromTransforms(
        path, train_json_file, white_background, extension, train_num_camera_ratio, dataset_type)
    print(f'#={len(train_cam_infos)}(train_num_camera_ratio={train_num_camera_ratio})')

    test_json_files = blender_test_jsons.split(',') if blender_test_jsons is not None else ["transforms_test.json"]
    print(f"Reading Test Transforms from {test_json_files}", end=' ')
    test_cam_infos = {}
    if test_json_files != [""]:
        for test_json_file in test_json_files:
            tag = test_json_file.replace('.json', '')
            test_cam_infos[tag] = readCamerasFromTransforms(
                path, test_json_file, white_background, extension, dataset_type=dataset_type)
            print(f'{test_json_file}, #={len(test_cam_infos[tag])}')
    
    if not eval:
        if test_cam_infos == {}:
            test_cam_infos = []
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        num_pts = 100_000
        if sample_from_pcd != '':
            plydata = PlyData.read(sample_from_pcd)
            vertices = plydata['vertex']
            xyz = np.vstack(
                [vertices['x'], vertices['y'], vertices['z']]).T
            num_pts = min(num_pts, xyz.shape[0])
            sample_indices = np.random.choice(xyz.shape[0], size=[num_pts], replace=False)
            xyz = xyz[sample_indices,:]
            print(f"Sampling {num_pts} points from {sample_from_pcd}...")
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
                shs), normals=np.zeros((num_pts, 3)))
            ply_path = os.path.join(path, "random_sample_from_pcd.ply")
            storePly(ply_path, xyz, SH2RGB(shs) * 255)
        # Since this data set has no colmap data, we start with random points
        else:
            print(f"Generating random point cloud ({num_pts})...")
            
            # We create random points inside the bounds of the synthetic Blender scenes
            if len(blender_bbox)==1:
                radius = blender_bbox[0]
                xyz = np.random.random((num_pts, 3)) * (2*radius) - radius
            else:
                x_max, y_max, z_max, x_min, y_min, z_min = blender_bbox
                D = np.array([float(x_max)-float(x_min), float(y_max)-float(y_min), float(z_max)-float(z_min)])
                xyz_min = np.array([float(x_min), float(y_min), float(z_min)])
                xyz = np.random.random((num_pts, 3)) * D + xyz_min
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

            #By Yutong, we refrain from storing the point cloud of random pcd in the data dir as points3d.ply
            ply_path = os.path.join(path, "random_bbox.ply")
            storePly(ply_path, xyz, SH2RGB(shs) * 255) 
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readMultiScale(path, white_background,split, only_highres=False):
    cam_infos = []
    
    print("read split:", split)
    with open(os.path.join(path, 'metadata.json'), 'r') as fp:
        meta = json.load(fp)[split]
        
    meta = {k: np.array(meta[k]) for k in meta}
    
    # should now have ['pix2cam', 'cam2world', 'width', 'height'] in self.meta
    for idx, relative_path in enumerate(meta['file_path']):
        if only_highres and not relative_path.endswith("d0.png"):
            continue
        image_path = os.path.join(path, relative_path)
        image_name = Path(image_path).stem
        
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = meta["cam2world"][idx]
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        fovx = focal2fov(meta["focal"][idx], image.size[0])
        fovy = focal2fov(meta["focal"][idx], image.size[1])
        FovY = fovy 
        FovX = fovx

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
    return cam_infos


def readMultiScaleNerfSyntheticInfo(path, white_background, eval, load_allres=False):
    print("Reading train from metadata.json")
    train_cam_infos = readMultiScale(path, white_background, "train", only_highres=(not load_allres))
    print("number of training images:", len(train_cam_infos))
    print("Reading test from metadata.json")
    test_cam_infos = readMultiScale(path, white_background, "test", only_highres=False)
    print("number of testing images:", len(test_cam_infos))
    if not eval:
        print("adding test cameras to training")
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Multi-scale": readMultiScaleNerfSyntheticInfo,
}
