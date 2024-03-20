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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import os
from time import time 
from utils.depth_utils import estimate_depth

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 width0=None, height0=None, fovx=None, fovy=None, bg=np.array([0.0, 0.0, 0.0]),
                 point_cloud=None, kpt_depth_cache="", 
                 dense_depth_cache="",
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.bg = bg
        self.kpt_depth_cache = kpt_depth_cache

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if image is not None:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        else:
            self.image_width = width0
            self.image_height = height0
            self.original_image = None


        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        if kpt_depth_cache !="":
            if os.path.exists(kpt_depth_cache) is False:
                point_cloud = torch.tensor(point_cloud, device=self.data_device)
                PROJ_t = self.full_proj_transform #4x4
                point_cloud = torch.cat([point_cloud, torch.ones((point_cloud.shape[0], 1), device=point_cloud.device)], dim=1) #Nx4
                keypoint = point_cloud @ PROJ_t #Nx4
                m_w = 1. / (keypoint[:,-1:] + 0.0000001) #N,1 -- the depth in the camera space
                keypoint_depth = keypoint[:,-1]
                keypoint_xy = keypoint[:,:2] * m_w #Nx2 (x,y)
                in_frustum = (keypoint_depth > 0.02)*(keypoint_xy[:,0]>-1)*(keypoint_xy[:,0]<1)*(keypoint_xy[:,1]>-1)*(keypoint_xy[:,1]<1)
                keypoint_depth = keypoint_depth[in_frustum]
                keypoint_xy = keypoint_xy[in_frustum]
                keypoint_u = torch.clip(((keypoint_xy[:,1]+1)/2 * self.image_height).long(),0,self.image_height-1)
                keypoint_v = torch.clip(((keypoint_xy[:,0]+1)/2 * self.image_width).long(),0,self.image_width-1)
                keypoint_i = keypoint_u * self.image_width + keypoint_v

                sort_by_depth_i = torch.argsort(keypoint_depth, dim=0, descending=False)
                keypoint_idepth = torch.stack([keypoint_i[sort_by_depth_i], keypoint_depth[sort_by_depth_i]], dim=1) #N,3
                keypoint_idepth = sorted(keypoint_idepth, key=lambda x: x[0]) #sorted by id, stable
                keypoint_idepth = torch.stack(keypoint_idepth, dim=0) #N,3

                _, counts = torch.unique_consecutive(keypoint_idepth[:,0], return_counts=True)
                ids = torch.zeros_like(counts)
                ids[1:] = torch.cumsum(counts[:-1], dim=0)
                self.keypoint_uv = torch.stack([torch.divide(keypoint_idepth[ids, 0], self.image_width, rounding_mode='floor'), 
                                           keypoint_idepth[ids, 0] % self.image_width], dim=1)
                self.keypoint_depth = keypoint_idepth[ids, 1]

                # print(counts)
                # print(keypoint_idepth[:3])
                # print(self.keypoint_uv[:3], self.keypoint_depth[:3])

                os.makedirs(os.path.dirname(kpt_depth_cache), exist_ok=True)
                torch.save((self.keypoint_uv.cpu(), self.keypoint_depth.cpu()), kpt_depth_cache)
            else:
                self.keypoint_uv, self.keypoint_depth = torch.load(kpt_depth_cache)
                self.keypoint_uv = self.keypoint_uv.to(self.data_device).long()
                self.keypoint_depth = self.keypoint_depth.to(self.data_device)[:,None]
        
        if dense_depth_cache !="":
            if os.path.exists(dense_depth_cache) is False:
                self.dense_depth = estimate_depth(self.original_image, mode='test')
                os.makedirs(os.path.dirname(dense_depth_cache), exist_ok=True)
                torch.save(self.dense_depth.cpu(), dense_depth_cache)
            else:
                self.dense_depth = torch.load(dense_depth_cache)
                self.dense_depth = self.dense_depth.to(self.data_device)

        # print('Check!')
        # import cv2
        # image_array = (image.cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
        # cv2.imwrite('debug_rgb.png', image_array)
        # for uv in self.keypoint_uv:
        #     cv2.circle(image_array, (int(uv[1]), int(uv[0])), 3, (0, 0, 255), -1)
        # cv2.imwrite('debug_keypoint.png', image_array)  
        # print('Check!')

        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        tan_fovx = np.tan(self.FoVx / 2.0)
        tan_fovy = np.tan(self.FoVy / 2.0)
        self.focal_y = self.image_height / (2.0 * tan_fovy)
        self.focal_x = self.image_width / (2.0 * tan_fovx)

    def move_to_device(self, data_device):
        self.data_device = torch.device(data_device)
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.to(self.data_device))
         
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

