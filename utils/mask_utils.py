import torch
import torch.nn.functional as F
from utils.loss_utils import l1_loss, l2_loss, ssim

def mask_related_loss(pred_mask, gt_mask, loss_weight, loss_type='l1'):
    if loss_type == 'l1':
        loss = l1_loss(pred_mask, gt_mask)*loss_weight
    elif loss_type == 'l2':
        loss = l2_loss(pred_mask, gt_mask)*loss_weight
    elif loss_type == 'ssim':
        loss = 1 - ssim(pred_mask, gt_mask)*loss_weight
    elif loss_type == 'bce':
        loss = F.binary_cross_entropy(pred_mask, gt_mask)*loss_weight
    else:
        raise NotImplementedError(f"Loss type {loss_type} not implemented!")
    mask_loss_dic = {f'mask_{loss_type}_loss': loss}
    return loss, mask_loss_dic