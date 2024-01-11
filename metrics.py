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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim

import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from glob import glob 

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)

        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_path, scale):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("model_path:", model_path)

    for test_dir in glob(f'{model_path}/train*')+glob(f'{model_path}/test*'):
        split = test_dir.split('/')[-1]
        print("Split:", split)
        full_dict[split] = {}
        per_view_dict[split] = {}
        full_dict_polytopeonly[split] = {}
        per_view_dict_polytopeonly[split] = {}
        for method in os.listdir(test_dir):
            print("Method:", method)
            full_dict[split][method] = {}
            per_view_dict[split][method] = {}
            full_dict_polytopeonly[split][method] = {}
            per_view_dict_polytopeonly[split][method] = {}

            method_dir = Path(test_dir) / method
            gt_dir = method_dir/ "gt"
            if not os.path.isdir(gt_dir):
                gt_dir = method_dir/ "gt_-1"

            renders_dir = method_dir / "renders"
            if not os.path.isdir(renders_dir):
                renders_dir = method_dir / "test_preds_-1"

            renders, gts, image_names = readImages(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")

            full_dict[split][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                    "PSNR": torch.tensor(psnrs).mean().item(),
                                                    "LPIPS": torch.tensor(lpipss).mean().item()})
            per_view_dict[split][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

        with open(model_path + "/results.json", 'w') as fp:
            json.dump(full_dict, fp, indent=True)
        with open(model_path + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_path', '-m', required=True, type=str, default=[])
    parser.add_argument('--resolution', '-r', type=int, default=-1)
    
    args = parser.parse_args()
    evaluate(args.model_path, args.resolution)
