import torch
from utils.loss_utils import keypoint_depth_loss
INF_VALUE = 10000000000.0

def split2patch(x, patch_size):
    if x.dim() == 3:
        x = x[None,...]
    elif x.dim() == 4:
        pass
    elif x.dim() == 2:
        x = x[None,None,...]
    x = torch.nn.functional.unfold(x, patch_size, stride=patch_size) #1,kernel_size*kernel_size,N
    x = x[0] #kernel_size*kernel_size,N
    return x

def masked_mean_std(x, mask):
    #x Kernal_size*Kernal_size,N
    #mask kernel_size*kernel_size, N
    mean = (x*mask).sum(dim=0)/(mask.sum(dim=0)+1e-10)
    std = ((x-mean)**2*mask).sum(dim=0)/(mask.sum(dim=0)+1e-10)
    return mean, std

def DNGS_loss(pred_depth, dense_depth, dense_depth_mask, patch_range, total_weight, gn_weight, ln_weight):
    H,W = dense_depth_mask.shape
    #uniformly sample patch_size from patch_range
    patch_size = torch.randint(int(patch_range[0]), int(patch_range[1]), (1,)).item()
    pred_depth_patches = split2patch(pred_depth, patch_size) #1,1,patch_size,patch_size,N
    dense_depth_patches = split2patch(dense_depth, patch_size) #1,1,patch_size,patch_size,N
    dense_depth_mask_patches = split2patch(dense_depth_mask.float(), patch_size) #patch_size*patch_size, N


    #compute the mean and std of each patch
    pred_depth_mean, pred_depth_std = masked_mean_std(pred_depth_patches, dense_depth_mask_patches)
    dense_depth_mean, dense_depth_std = masked_mean_std(dense_depth_patches, dense_depth_mask_patches)

    pred_depth_std_global = pred_depth.std().detach()
    dense_depth_std_global = dense_depth.std().detach()

    pred_depth_ln = (pred_depth_patches - pred_depth_mean)/(pred_depth_std+1e-6) #N vs kernel_size*kernel_size,N
    dense_depth_ln = (dense_depth_patches - dense_depth_mean)/(dense_depth_std+1e-6)
    pred_depth_gn = (pred_depth_patches-pred_depth_mean)/(pred_depth_std_global+1e-6)
    dense_depth_gn = (dense_depth_patches-dense_depth_mean)/(dense_depth_std_global+1e-6) #N

    l2_loss = torch.nn.MSELoss(reduction='none')
    ln_loss = (l2_loss(pred_depth_ln, dense_depth_ln)*dense_depth_mask_patches).mean()
    gn_loss = (l2_loss(pred_depth_gn, dense_depth_gn)*dense_depth_mask_patches).mean()

    ln_loss *= ln_weight*total_weight
    gn_loss *= gn_weight*total_weight

    loss = ln_loss + gn_loss
    return loss, {'ln_loss':ln_loss, 'gn_loss':gn_loss}
    


def depth_related_loss(pred_depth, keypoint_uv, keypoint_depth, dense_depth, 
                    keypoint_depth_loss_weight, keypoint_depth_loss_type,
                    dense_depth_loss_weight, dense_depth_loss_type,
                    patch_range, gn_weight, ln_weight):
    depth_loss = 0
    depth_loss_dic = {}
    if keypoint_depth_loss_weight > 0:
        keypoint_depth_loss_value = keypoint_depth_loss(
            depth_map=pred_depth, 
            keypoint_uv=keypoint_uv, keypoint_depth=keypoint_depth, 
            loss_type=keypoint_depth_loss_type)
        depth_loss += keypoint_depth_loss_weight * keypoint_depth_loss_value
        depth_loss_dic['keypoint_depth_loss'] = keypoint_depth_loss_value
    if dense_depth_loss_weight > 0:
        dense_depth_mask = (dense_depth != INF_VALUE)
        if dense_depth_loss_type == 'l1':
            dense_depth_loss_value = torch.abs((dense_depth - pred_depth))[0,dense_depth_mask].mean()
            depth_loss_dic['dense_depth_loss'] = dense_depth_loss_value
        elif dense_depth_loss_type == 'DNGS':
            dense_depth_loss_value, depth_loss_dic_ = DNGS_loss(pred_depth, dense_depth, dense_depth_mask, patch_range, dense_depth_loss_weight, gn_weight, ln_weight)
            depth_loss_dic = {**depth_loss_dic, **depth_loss_dic_}
        else:
            raise NotImplementedError
        depth_loss += dense_depth_loss_weight * dense_depth_loss_value
    return depth_loss, depth_loss_dic



#https://github.com/VITA-Group/FSGS/blob/main/utils/depth_utils.py
import torch
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
for param in midas.parameters():
    param.requires_grad = False

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform
downsampling = 1

def estimate_depth(img, mode='test'):
    h, w = img.shape[1:3]
    norm_img = (img[None] - 0.5) / 0.5
    norm_img = torch.nn.functional.interpolate(
        norm_img,
        size=(384, 512),
        mode="bicubic",
        align_corners=False)

    if mode == 'test':
        with torch.no_grad():
            prediction = midas(norm_img)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h//downsampling, w//downsampling),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
    else:
        prediction = midas(norm_img)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h//downsampling, w//downsampling),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction
