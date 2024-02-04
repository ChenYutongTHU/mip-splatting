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

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, scale_factor, 
               save_as_idx=False, save_grad=False, opt=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"test_preds_{scale_factor}")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"gt_{scale_factor}")\

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    if save_grad:
        grad_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"grad")
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

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, save_as_idx : bool, save_grad : bool, opt=None):
    torch.set_grad_enabled(save_grad)
    gaussians = GaussianModel(dataset.sh_degree)
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
    parser.add_argument("--iteration", default=-1, type=int)
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
