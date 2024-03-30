from pytorch3d.loss import chamfer_distance
import torch

def chamfer_loss(pred_xyz, gt_xyz, random_sample=100000):
    #pred xyz (P1,3)
    #gt_xyz (P2,3)
    #print(pred_xyz.shape, gt_xyz.shape)
    #random sample 1e5 points
    pred_xyz = pred_xyz[torch.randperm(pred_xyz.shape[0])[:random_sample]]
    gt_xyz = gt_xyz[torch.randperm(gt_xyz.shape[0])[:random_sample]]
    cd,_ = chamfer_distance(pred_xyz[None,...], gt_xyz[None,...])
    return cd
