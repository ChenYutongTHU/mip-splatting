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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def keypoint_depth_loss(keypoint_uv, keypoint_depth, depth_map, loss_type='l1'):
    #depth_map [1,H,W]
    #keypoint_xy [N,2] -> [1,1,N,2]
    '''
    pred_depth = torch.nn.functional.grid_sample(depth_map.unsqueeze(1), #1,1,H,W 
                                                 keypoint_xy.unsqueeze(0).unsqueeze(0), align_corners=True).view(-1,1) #1,1,1,N->N
    print(depth_map.shape, keypoint_xy.shape,pred_depth.shape)
    print(keypoint_xy[0], keypoint_depth[0], pred_depth[0], depth_map[0, int(keypoint_xy[0,1]), int(keypoint_xy[0,0])])
    print(keypoint_xy[0], keypoint_depth[0], pred_depth[0], depth_map[0, int(keypoint_xy[0,0]), int(keypoint_xy[0,1])])
    '''
    # H, W = depth_map.shape[1], depth_map.shape[2]
    # keypoint_u = torch.clip(((keypoint_xy[:,1]+1)/2 * H).long(),0,H-1)
    # keypoint_v = torch.clip(((keypoint_xy[:,0]+1)/2 * W).long(),0,W-1)
    pred_depth = depth_map[0,keypoint_uv[:,0], keypoint_uv[:,1]].view(-1,1)
    if loss_type == 'l1':
        return l1_loss(pred_depth, keypoint_depth)
    elif loss_type == 'l2':
        return l2_loss(pred_depth, keypoint_depth)

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

