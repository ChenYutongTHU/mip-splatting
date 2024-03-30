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
import numpy as np
import open3d as o3d
import cv2
import torch
import random
from random import randint
from utils.loss_utils import l1_loss, ssim, keypoint_depth_loss
from utils.pcd_utils import chamfer_loss
from utils.depth_utils import depth_related_loss
from utils.mask_utils import mask_related_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from tqdm import tqdm 
import wandb 
from PIL import Image

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

@torch.no_grad()
def create_offset_gt(image, offset):
    height, width = image.shape[1:]
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).cuda()
    
    id_coords = id_coords.permute(1, 2, 0) + offset
    id_coords[..., 0] /= (width - 1)
    id_coords[..., 1] /= (height - 1)
    id_coords = id_coords * 2 - 1
    
    image = torch.nn.functional.grid_sample(image[None], id_coords[None], align_corners=True, padding_mode="border")[0]
    return image

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, show_wandb):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.apply_3Dfilter_off, dataset.isotropic)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background_still = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    if dataset.dataset_type == "list":
        viewpoint_stack = None
    else:
        viewpoint_loader = scene.getTrainCameras()
        viewpoint_iter = iter(viewpoint_loader)

    if args.dataset_type == "list":
        trainCameras = scene.getTrainCameras().copy()
        #testCameras = scene.getTestCameras().copy()
        #allCameras = trainCameras + testCameras
    elif args.dataset_type == "loader":
        trainCameras = viewpoint_loader.dataset.wo_image
    else:
        raise Exception("Unknown dataset type!")
    
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)
    gaussians.compute_3D_filter(cameras=trainCameras)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background_still, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if dataset.dataset_type == "list":
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        elif dataset.dataset_type == "loader":
            try:
                viewpoint_cam = next(viewpoint_iter)
                viewpoint_cam.move_to_device(args.data_device)
            except StopIteration:
                viewpoint_iter = iter(viewpoint_loader)
                viewpoint_cam = next(viewpoint_iter)
                viewpoint_cam.move_to_device(args.data_device)
        else:
            assert False, "Could not recognize dataset type!"

        
        # Pick a random high resolution camera
        if random.random() < 0.3 and dataset.sample_more_highres:
            viewpoint_cam = trainCameras[highresolution_index[randint(0, len(highresolution_index)-1)]]
            
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        #TODO ignore border pixels
        if dataset.ray_jitter:
            subpixel_offset = torch.rand((int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
            # subpixel_offset *= 0.0
        else:
            subpixel_offset = None
        background = torch.tensor(viewpoint_cam.bg,device='cuda',dtype=torch.float32) 
        # Loss
        loss = 0
        gt_image = viewpoint_cam.original_image.cuda()
        if not opt.only_pcd_debug:
            if opt.only_color:
                gaussians.detach_everything_but_color()
            if opt.only_color_scale:
                gaussians.detach_everything_but_color_scale()
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, subpixel_offset=subpixel_offset)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            # sample gt_image with subpixel offset
            if dataset.resample_gt_image:
                gt_image = create_offset_gt(gt_image, subpixel_offset)
            Ll1 = l1_loss(image, gt_image)
            loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        else:
            Ll1 = torch.tensor(0.)
        if opt.keypoint_depth_loss_weight+opt.dense_depth_loss_weight > 0:
            depth_loss_total, depth_loss_dic = depth_related_loss(pred_depth=render_pkg['depth'], 
                                    keypoint_depth=viewpoint_cam.keypoint_depth, keypoint_uv=viewpoint_cam.keypoint_uv,
                                    keypoint_depth_loss_weight=opt.keypoint_depth_loss_weight,
                                    keypoint_depth_loss_type=opt.keypoint_depth_loss_type,
                                    dense_depth=viewpoint_cam.dense_depth, 
                                    dense_depth_loss_weight=opt.dense_depth_loss_weight,
                                    dense_depth_loss_type=opt.dense_depth_loss_type,
                                    patch_range=opt.patch_range, gn_weight=opt.gn_weight, ln_weight=opt.ln_weight)

            loss += depth_loss_total
        else:
            depth_loss_dic = {}
        
        if opt.mask_loss_weight > 0:
            mask_loss, mask_loss_dic = mask_related_loss(
                pred_mask=render_pkg['alpha'], gt_mask=viewpoint_cam.gt_alpha_mask,
                loss_type=opt.mask_loss_type, loss_weight=opt.mask_loss_weight)
            loss += mask_loss
        else:
            mask_loss_dic = {}

        if opt.pcd_loss_weight > 0:
            scene.point_cloud_complete = scene.point_cloud_complete.to("cuda")
            if opt.cd_sample_type == 'mean':
                pcd_loss = chamfer_loss(gaussians._xyz, scene.point_cloud_complete, opt.chamfer_n)* opt.pcd_loss_weight
            elif opt.cd_sample_type == 'gaussian':
                xyz = gaussians.sample_points_from_gaussians(opt.chamfer_n)
                pcd_loss = chamfer_loss(xyz, scene.point_cloud_complete, opt.chamfer_n)* opt.pcd_loss_weight
            loss += pcd_loss
            pcd_loss_dic = {"pcd_loss": pcd_loss}

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background_still, dataset.kernel_size), 
                            show_wandb=show_wandb, 
                            eval_train_interval=dataset.eval_train_interval,
                            eval_test_interval=dataset.eval_test_interval)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if show_wandb and iteration % 100 == 0:
                wandb.log({"loss": loss.item(), "loss_l1": Ll1.item(), "loss_ssim": (
                    1.0 - ssim(image, gt_image)).item() if not opt.only_pcd_debug else 0.0, 
                    **{kk:vv.item() for kk, vv in depth_loss_dic.items()},
                    **{kk:vv.item() for kk, vv in mask_loss_dic.items()},
                    **{kk:vv.item() for kk, vv in pcd_loss_dic.items()}, }, step=iteration)
                wandb.log({"learning_rate/"+param_group["name"]:   param_group["lr"]
                            for param_group in gaussians.optimizer.param_groups}, step=iteration)
                
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if (iteration==0 or iteration % (opt.densification_interval*1) == 0) and show_wandb:
                    def wandb_percentile(data, name, step, percentiles=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]):
                        data = sorted(data.detach().cpu().numpy())
                        N = len(data)
                        for p in percentiles:
                            n = min(int(N*p//100), N-1)
                            wandb.log({name+f'_percentile{p}%': data[n]}, step=step)
                    
                    for name, scale in [('scale', gaussians.get_scaling),
                                        ('scale_with_3D_filter', gaussians.get_scaling_with_3D_filter)]:
                        wandb_percentile(torch.max(scale, dim=1).values, f"{name}-max", iteration)
                        wandb_percentile(torch.min(scale, dim=1).values, f"{name}-min", iteration)
                        wandb_percentile(torch.median(scale, dim=1).values, f"{name}-median", iteration)
                        wandb_percentile(torch.mean(scale, dim=1), f"{name}-mean", iteration)
                        wandb_percentile(torch.max(scale, dim=1).values/torch.min(scale, dim=1).values, f"{name}-max-div-min", iteration)

                    wandb_percentile(gaussians._scaling.grad.view(-1), "scaling_grad", iteration)
                    wandb_percentile(gaussians.get_opacity.view(-1), "opacity", iteration)
                    wandb_percentile(gaussians.max_radii2D[visibility_filter].view(-1), "visible-max_radii2D", iteration)
                    # wandb_percentile(gaussians.min_radii2D[visibility_filter].view(-1), "visibile-min_radii2D", iteration)

                    wandb_percentile(radii[visibility_filter].view(-1),"visible-radii2D (the max lambda)", iteration)
                    # wandb_percentile(radii_min[visibility_filter].view(-1),"visible-radii2D (the min lambda)", iteration)
                    # wandb_percentile(render_pkg["radiiBeforeFilter"][visibility_filter].view(-1),"visible-radii2D-before-filter (the max lambda)", iteration)
                    # wandb_percentile(render_pkg["radii_minBeforeFilter"][visibility_filter].view(-1),"visible-radii2D-before-filter (the min lambda)", iteration)                
                    
                    #wandb_percentile(torch.norm(viewspace_point_tensor.grad[:, :2], dim=-1), "mean2D_grad", iteration)
                    grads = gaussians.xyz_gradient_accum / gaussians.denom
                    grads[grads.isnan()] = 0
                    wandb_percentile(grads, "average-acc-mean2D-grad", iteration,[0, 20, 40, 60, 80, 90, 95, 97, 99])
                    densify_number = torch.sum(grads>opt.densify_grad_threshold)
                    wandb_percentile(gaussians.filter_3D.view(-1), "filter_3D", iteration)
                    wandb.log({"number-gaussians": gaussians.get_xyz.shape[0],
                            "densify_grad_threshold": opt.densify_grad_threshold,
                            "split-or-clone_threshold": gaussians.percent_dense*scene.cameras_extent,
                            "ge-threshold_number": densify_number}, step=iteration)
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    densify_and_prune_stats = gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    gaussians.compute_3D_filter(cameras=trainCameras)
                    if show_wandb and iteration % (opt.densification_interval*1) == 0:
                        wandb.log(densify_and_prune_stats, step=iteration)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)
        
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, 
                    show_wandb, eval_train_interval, eval_test_interval):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        if eval_train_interval>0:
            if args.dataset_type == "list":
                validation_configs = [{'name': 'train', 'cameras' : scene.getTrainCameras()[::eval_train_interval]}]
            else:
                validation_configs = [{'name': 'train', 
                                    'cameras': [scene.getTrainCameras().dataset[ii] \
                                                for ii in np.arange(0, len(scene.getTrainCameras().dataset), eval_train_interval)]}]
        else:
            validation_configs = [] #skip train
        if type(scene.test_cameras) == dict:
            for test_name in scene.test_cameras.keys():
                if args.dataset_type == "list":
                    validation_configs.append({'name': test_name, 
                                               'cameras': [scene.getTestCameras(test_name=test_name)[ii] \
                                                for ii in np.arange(0, len(scene.getTestCameras(test_name=test_name)), eval_test_interval)]})
                else:
                    validation_configs.append({'name': test_name, 
                                           'cameras': [scene.getTestCameras(test_name=test_name).dataset[ii]
                                                for ii in np.arange(0, len(scene.getTestCameras(test_name=test_name).dataset), eval_test_interval)]})
        elif type(scene.test_cameras) == list:
            validation_configs.append({'name': 'test', 'cameras': scene.getTestCameras()[::eval_test_interval]})

        for config in validation_configs:
            output_dir = os.path.join(args.model_path, 'eval', str(iteration), config['name'])
            os.makedirs(output_dir, exist_ok=True)
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in tqdm(enumerate(config['cameras']), desc="Evaluating " + config['name'], total=len(config['cameras'])):
                    if args.dataset_type == 'loader':
                        viewpoint.move_to_device(args.data_device)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    #save image
                    image = image.permute(1, 2, 0).cpu().numpy()
                    image = Image.fromarray((image * 255).astype(np.uint8))
                    image.save(os.path.join(output_dir, viewpoint.image_name + ".png"))
                    if idx==0 and show_wandb:
                        wandb.log({f'eval/{config["name"]} - render': wandb.Image(image, caption=f'{config["name"]} - render')}, step=iteration)
                        # wandb.log({f'eval/{config["name"]} - ground_truth': wandb.Image(viewpoint.original_image.cpu().numpy().transpose(1, 2, 0), caption=f'{config["name"]} - ground_truth')}, step=iteration)

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if show_wandb:
                    wandb.log({'eval/'+f'{config["name"]} - psnr': psnr_test}, step=iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--test_save_interval", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    #args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    while True:
        try:
            network_gui.init(args.ip, args.port)
            break
        except Exception as e:
            print(e)
            print("Change port to {}".format(args.port + 1))
            args.port += 1
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project="gaussian", config=args, dir=args.model_path) #resume=?
        wandb.run.name = 'mipGS_'+os.path.basename(args.model_path)
        wandb.config.update(args, ) #notes=, tag=['','']

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if args.test_save_interval==-1:
        test_iterations = args.test_iterations
        save_iterations = args.save_iterations
    else:
        test_iterations = [1]+[i for i in np.arange(0, args.iterations+1, args.test_save_interval)][1:]
        save_iterations = test_iterations
    training(lp.extract(args), op.extract(args), pp.extract(args), test_iterations, save_iterations, args.checkpoint_iterations, args.start_checkpoint, 
             args.debug_from, args.wandb)

    # All done
    print("\nTraining complete.")
