import torch
from utils.loss_utils import keypoint_depth_loss

def depth_related_loss(pred_depth, keypoint_uv, keypoint_depth, 
                    keypoint_depth_loss_weight, keypoint_depth_loss_type):
    depth_loss = 0
    depth_loss_dic = {}
    if keypoint_depth_loss_weight > 0:
        keypoint_depth_loss_value = keypoint_depth_loss(
            depth_map=pred_depth, 
            keypoint_uv=keypoint_uv, keypoint_depth=keypoint_depth, 
            loss_type=keypoint_depth_loss_type)
        depth_loss += keypoint_depth_loss_weight * keypoint_depth_loss_value
        depth_loss_dic['keypoint_depth_loss'] = keypoint_depth_loss_value
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
