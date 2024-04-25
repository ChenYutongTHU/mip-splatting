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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
from utils.loss_utils import l1_loss, ssim
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import numpy as np
from utils.depth_utils import INF_VALUE
def visualize_depth_map(depth_map, alpha_map, high=None, low=None):
    alpha_mask = alpha_map > 0.2
    def normalize_depth_map(depth_map, high=None, low=None):
        if high is None:
            high = np.max(depth_map[alpha_mask])
        if low is None:
            low = np.min(depth_map[alpha_mask]) #To remove the background map
        normalized_depth_map = (depth_map - low) / (high - low)
        normalized_depth_map = np.clip(normalized_depth_map, 0, 1)
        return normalized_depth_map
    normalized_depth_map = normalize_depth_map(depth_map, high=high, low=low)
    colormap = cm.get_cmap('turbo')
    colored_depth_map = (colormap(normalized_depth_map[0]) * 255).astype(np.uint8)
    colored_depth_map = np.expand_dims(alpha_mask[0], axis=-1) * colored_depth_map
    image = Image.fromarray(colored_depth_map)
    return image

def visualize_alpha_map(alpha_map):
    alpha_map = alpha_map[0]*255
    alpha_map = alpha_map.astype(np.uint8)
    return Image.fromarray(alpha_map)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, scale_factor, 
               save_as_idx=False, save_grad=False, opt=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"test_preds_{scale_factor}")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"gt_{scale_factor}")\

    makedirs(render_path, exist_ok=True)
    makedirs(render_path+'_depth', exist_ok=True)
    makedirs(render_path+'_depth_mode', exist_ok=True)
    makedirs(render_path+'_alpha', exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    if save_grad:
        grad_path = os.path.join(model_path, name, "ours_{}".format(iteration), "grad")
        makedirs(grad_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        view.move_to_device(args.data_device)
        render_pkg = render(view, gaussians, pipeline, background, kernel_size=kernel_size)
        rendering, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt = view.original_image[0:3, :, :]
        if save_as_idx:
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        else:
            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))
            # if 'sergey' in render_path:
            #     _,h,w = render_pkg['depth'].shape
            #     render_pkg['depth'] = render_pkg['depth'][:,h//2:,w//4:3*w//4]
            #     # render_pkg['alpha'] = render_pkg['alpha'][:,h//2:,w//4:3*w//4]
            #     torchvision.utils.save_image(rendering[:,h//2:,w//4:3*w//4], os.path.join(render_path+'_depth', view.image_name + ".png"))
            
            for depth_key in ['depth','depth_mode']:
                depth_viz = visualize_depth_map(render_pkg[depth_key].cpu().numpy(), render_pkg['alpha'].cpu().numpy())
                depth_viz.save(os.path.join(render_path+f'_{depth_key}', view.image_name + f"_{depth_key}.png"))
                render_pkg[depth_key][render_pkg['alpha']< 0.5] = INF_VALUE
                torch.save(render_pkg[depth_key].cpu(), os.path.join(render_path+f'_{depth_key}', view.image_name + f"_{depth_key}.pt"))
            alpha_viz = visualize_alpha_map(render_pkg['alpha'].cpu().numpy())
            alpha_viz.save(os.path.join(render_path+'_alpha', view.image_name + "_alpha.png"))
        
        if save_grad:
            # import ipdb;ipdb.set_trace()
            Ll1 = l1_loss(rendering, gt)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(rendering, gt))
            loss.backward()
            grad_output = {
                'viewspace_grad': viewspace_point_tensor.grad[:,:2].cpu(),
                'visibility_filter': visibility_filter.cpu(),
                'radii': radii.cpu(),
            }
            torch.save(grad_output, os.path.join(grad_path, view.image_name + ".pt"))

def render_sets(dataset : ModelParams, iteration : str, pipeline : PipelineParams, skip_train : bool, skip_test : bool, save_as_idx : bool, save_grad : bool, opt=None):
    torch.set_grad_enabled(save_grad)
    gaussians = GaussianModel(
        dataset.sh_degree, dataset.apply_3Dfilter_off, dataset.isotropic)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    scale_factor = dataset.resolution
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    kernel_size = dataset.kernel_size
    if not skip_train:
        render_set(dataset.model_path, f"train-{dataset.focal_length_scale}_minus-depth-{dataset.minus_depth:.2f}", scene.loaded_iter, scene.getTrainCameras(
        ), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor, save_as_idx=save_as_idx, save_grad=save_grad, opt=opt)

    if not skip_test:
        if type(scene.test_cameras) == dict:
            for test_name in scene.test_cameras.keys():
                render_set(dataset.model_path, f"{test_name}-{dataset.focal_length_scale}_minus-depth-{dataset.minus_depth:.2f}", scene.loaded_iter, scene.getTestCameras(
                    test_name=test_name), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor, save_as_idx=save_as_idx, save_grad=save_grad, opt=opt)
        else:
            render_set(dataset.model_path, f"test-{dataset.focal_length_scale}_minus-depth-{dataset.minus_depth:.2f}", scene.loaded_iter, scene.getTestCameras(
            ), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor, save_as_idx=save_as_idx, save_grad=save_grad, opt=opt)
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default='-1', type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_as_idx", action="store_true")
    parser.add_argument("--save_grad", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.save_grad:
        op = OptimizationParams(parser)
        opt = op.extract(op)
    else:
        opt = None
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, 
                args.save_as_idx, args.save_grad, opt=opt)
