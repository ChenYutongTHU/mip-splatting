from pytorch3d.ops.knn import knn_points
import torch

def smooth_loss(gaussians, loss_type, n_neighbours, 
                smooth_xyz_weight, smooth_opacity_weight, smooth_cov_weight, smooth_color_weight):
    #xyz_
    xyz = gaussians.get_xyz #N,3 
    loss_dict = {}
    loss = 0
    if loss_type == 'laplacian':
        dists, idx,_ = knn_points(xyz.unsqueeze(0), 
                                xyz.unsqueeze(0), K=n_neighbours, return_sorted=True) #dists:1,N,K+1, idx:1,N,K+1
        idx = idx[0,:,1:] #N,K
        smooth_xyz_loss = smooth_xyz_weight*torch.abs(xyz - xyz[idx].mean(1)).mean() #N,3, (N,K,3)
        smooth_opacity_loss = smooth_opacity_weight*torch.abs(gaussians.get_opacity - gaussians.get_opacity[idx].mean(1)).mean()
        smooth_cov_loss = smooth_cov_weight*torch.abs(gaussians.get_covariance() - gaussians.get_covariance()[idx].mean(1)).mean()
        smooth_color_loss = smooth_color_weight*torch.abs(gaussians.get_features - gaussians.get_features[idx].mean(1)).mean()
        loss_dict = {
            'smooth_xyz_loss':smooth_xyz_loss,
            'smooth_opacity_loss':smooth_opacity_loss,
            'smooth_cov_loss':smooth_cov_loss,
            'smooth_color_loss':smooth_color_loss
        }
        loss = smooth_xyz_loss + smooth_opacity_loss + smooth_cov_loss + smooth_color_loss
    else:
        raise ValueError
    return loss, loss_dict