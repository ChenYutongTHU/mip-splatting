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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import os
from pytorch3d.renderer import FoVPerspectiveCameras
from utils.graphics_utils import getWorld2View
WARNED = False

def sample_new_camera(scene_center, scene_radius, near, far, angle_factor, camera_like):
    #Step1: Sample camera position uniformly on a sphere
    theta = np.random.rand() * 2 * np.pi
    phi = np.random.rand() * np.pi
    r = np.random.rand() * (far - near) + near
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    camera_center = np.array([x, y, z]) + scene_center 

    #Step2: Sample where the camera faces towards
    #Sample in the sphere of (scene_center, scene_radius)
    theta = np.random.rand() * 2 * np.pi
    phi = np.random.rand() * np.pi
    r = np.random.rand() * scene_radius * angle_factor
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    scene_center = np.array([x, y, z]) + scene_center
    z_axis = scene_center - camera_center

    #Step3: Sample camera orientation
    #First face the camera towards the scene center
    z_axis = z_axis / np.linalg.norm(z_axis)
    c2w_R = np.eye(3)
    c2w_R[:, 2] = z_axis
    c2w_R[:, 0] = np.cross(np.array([0, 0, 1]), z_axis)
    c2w_R[:, 0] = c2w_R[:, 0] / np.linalg.norm(c2w_R[:, 0])
    c2w_R[:, 1] = np.cross(z_axis, c2w_R[:, 0])
    #Then rotate the camera randomly around the z_axis
    theta = np.random.rand() * 2 * np.pi
    R_around_z = np.array([[np.cos(theta), -np.sin(theta), 0],  
                  [np.sin(theta), np.cos(theta), 0], 
                  [0, 0, 1]])
    c2w_R = c2w_R@R_around_z
    c2w_T = camera_center
    
    c2w = np.eye(4)
    c2w[:3, :3] = c2w_R
    c2w[:3, 3] = c2w_T
    w2c = np.linalg.inv(c2w)
    w2c_R_t = w2c[:3, :3].transpose()
    w2c_T = w2c[:3, 3]
    #The R and T will be forwarded to graphics_utils.getWorld2View2 
    camera_for_GSras = Camera(colmap_id=camera_like.colmap_id, R=w2c_R_t, T=w2c_T,
                    FoVx=camera_like.FoVx, 
                    FoVy=camera_like.FoVy, 
                    image=None, gt_alpha_mask=None,
                    image_name=None, uid=0, data_device='cuda', 
                    width0=camera_like.image_width, height0=camera_like.image_height, 
                    point_cloud=None, 
                    kpt_depth_cache="") 
    
    #For pytorch3D, we also need transposed R
    c2w_flipobj = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]).dot(c2w) #WE NEED TO FLIP THE LOADED OBJ
    w2c_flipobj = np.linalg.inv(c2w_flipobj)
    w2c_t3d_flipobj = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]).dot(w2c_flipobj) #Pytorch3D the camera coordinate is different
    R = w2c_t3d_flipobj[:3, :3].T
    T = w2c_t3d_flipobj[:3, 3]
    device = camera_like.data_device
    camera_for_3dras = FoVPerspectiveCameras(device=device, 
                                R=torch.tensor(R,dtype=torch.float32,device=device).unsqueeze(0),
                                T=torch.tensor(T,dtype=torch.float32,device=device).unsqueeze(0),
                                fov=camera_like.FoVx, 
                                degrees=False)
    '''

    print(camera_like.T, camera_like.T)
    print('='*5)
    print(w2c_R_t, w2c_T)
    camera_for_GSras = Camera(colmap_id=camera_like.colmap_id,
            R = camera_like.R,
            T = camera_like.T,
            FoVx=camera_like.FoVx, 
            FoVy=camera_like.FoVy, 
            image=None, gt_alpha_mask=None,
            image_name=None, uid=0, data_device='cuda', 
            width0=camera_like.image_width, height0=camera_like.image_height, 
            point_cloud=None, 
            kpt_depth_cache="")   
    w2c_GSras = getWorld2View(camera_like.R, camera_like.T)
    c2w = np.linalg.inv(w2c_GSras)
    c2w = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]).dot(c2w) #WE NEED TO FLIP THE LOADED OBJ
    w2c = np.linalg.inv(c2w)
    w2c_t3d = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]).dot(w2c) #Pytorch3D the camera coordinate is different
    R = w2c_t3d[:3, :3].T
    T = w2c_t3d[:3, 3]
    camera_for_3dras = FoVPerspectiveCameras(device=device, 
                                R=torch.tensor(R, device=device).unsqueeze(0),
                                T=torch.tensor(T, device=device).unsqueeze(0),
                                fov=camera_like.FoVx, 
                                degrees=False)
    '''
    return camera_for_GSras, camera_for_3dras
    
        

class CameraDataset(Dataset):
    def __init__(self, cam_infos, resolution_scale, args, is_training, point_cloud):
        self.cam_infos = cam_infos
        self.resolution_scale = resolution_scale
        self.args = args
        self.point_cloud = point_cloud
        image = Image.open(self.cam_infos[0].image_path)

        print('Use dataset_type=loader, we assume all images have the same width and height')
        self.width0, self.height0 = image.size
        self.kpt_depth_cache = args.kpt_depth_cache

        self.wo_image = [
            Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, 
                  FoVy=focal2fov(fov2focal(cam_info.FovX, self.width0), self.height0), 
                  image=None, gt_alpha_mask=None,
                  image_name=cam_info.image_name, uid=id, data_device='cpu', 
                  width0=self.width0, height0=self.height0, 
                  point_cloud=point_cloud, 
                  kpt_depth_cache="") \
                for id,cam_info in enumerate(self.cam_infos)]
        
        if args.rnd_background and is_training:
            self.bg_color = 'rnd'
        elif args.transparent_background:
            self.bg_color = 'transparent' #Use the alpha channel
        elif args.white_background:
            self.bg_color = np.array([1,1,1]) 
        else:
            self.bg_color = np.array([0, 0, 0])

    def __len__(self):
        return len(self.cam_infos)

    def __getitem__(self, idx):
        image = Image.open(self.cam_infos[idx].image_path)
        im_data = np.array(image.convert("RGBA"))
        if type(self.bg_color)==str:
            if self.bg_color == 'rnd':
                bg = np.random.rand(3)
            elif self.bg_color == 'transparent':
                bg = np.zeros(3)
        else:
            bg = self.bg_color
        norm_data = im_data / 255.0
        if type(self.bg_color)==str and self.bg_color == 'transparent':
            image = image.convert("RGBA")
        else:
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        return loadCam(self.args, idx, self.cam_infos[idx], self.resolution_scale, 
                       image=image, data_device='cpu', bg=bg, 
                       point_cloud=self.point_cloud,
                       kpt_depth_cache=os.path.join(self.kpt_depth_cache, self.cam_infos[idx].image_name+'.pt') if self.kpt_depth_cache!="" else "",
                       dense_depth_cache=os.path.join(self.args.dense_depth_cache, self.cam_infos[idx].image_name+'.pt') if self.args.dense_depth_cache!="" else "",)
    
def loadCam(args, id, cam_info, resolution_scale, bg, point_cloud, kpt_depth_cache, dense_depth_cache, image=None, data_device=None):
    if image is None:
        image = cam_info.image 
        FovY = cam_info.FovY
    else:
        image = image 
        width, height = image.size
        FovY = focal2fov(fov2focal(cam_info.FovX, width), height)

    data_device = args.data_device if data_device is None else data_device

    orig_w, orig_h = image.size

    if args.resolution in [1, 2, 4, 8, 16, 32, 64]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                # global WARNED
                # if not WARNED:
                #     print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                #         "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                #     WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(image, resolution)
    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=data_device, bg=bg, 
                  point_cloud=point_cloud, kpt_depth_cache=kpt_depth_cache, dense_depth_cache=dense_depth_cache,)


def Camera_Collate_fn(batch):
    return batch[0]

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_training, point_cloud):
    if args.dataset_type.lower() == 'list': #preload image
        camera_list = []
        #assert args.rnd_background == False, "rnd_background is not supported for dataset_type=list"
        bg = np.array([1,1,1]) if args.white_background else np.array([0, 0, 0])
        for id, c in tqdm(enumerate(cam_infos)):
            camera_list.append(loadCam(args, id, c, resolution_scale, bg=bg, point_cloud=point_cloud, 
                                       kpt_depth_cache=os.path.join(args.kpt_depth_cache, c.image_name+'.pt') if args.kpt_depth_cache!="" else "",
                                       dense_depth_cache=os.path.join(args.dense_depth_cache, c.image_name+'.pt') if args.dense_depth_cache!="" else "",))
        return camera_list
    elif args.dataset_type.lower() == 'loader':
        dataset = CameraDataset(cam_infos, resolution_scale, args, is_training=is_training, point_cloud=point_cloud)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=is_training, collate_fn=Camera_Collate_fn, num_workers=4)
        return dataloader

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt) #It should be C2W here?
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height) if camera.height is not None else None,
        'fx' : fov2focal(camera.FovX, camera.width) if camera.width is not None else None,
        'FovX' : camera.FovX,
        'FovY' : camera.FovY,
    }
    return camera_entry
