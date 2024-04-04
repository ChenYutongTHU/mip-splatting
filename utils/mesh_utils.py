from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer.mesh.shader import HardDepthShader
import numpy as np
from utils.camera_utils import sample_new_camera

class Mesh():
    def __init__(self, mesh_path, sigma=0, device='cuda'):
        self.mesh = load_objs_as_meshes([mesh_path], device=device)
        self.mesh_path = mesh_path
        self.device=device
        print("Loaded mesh from", mesh_path)

        self.sigma = sigma
        raster_settings_soft = RasterizationSettings(
            image_size=512, 
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
            faces_per_pixel=5, 
        )
        R, T = look_at_view_transform(dist=5, elev=0, azim=180) #It does not matter here
        camera = FoVPerspectiveCameras(device=device, R=R, T=T)
       # blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0)) #Black background
        blend_params = BlendParams(sigma=sigma, gamma=0, background_color=(0.0, 0.0, 0.0)) #Black background
        # Silhouette renderer 
        self.renderer_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera, 
                raster_settings=raster_settings_soft
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params),
        )
        self.lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])  #Also does not matter here?

        self.renderer_depth = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera, 
                raster_settings=raster_settings_soft
            ),
            shader=HardDepthShader(device=device),
        )

    def render_silhouette(self, camera):
        images = self.renderer_silhouette(self.mesh, camera = camera, lights=self.lights)
        return images

    def render_hard_depth(self, camera):
        depth = self.renderer_depth(self.mesh, cameras = camera, lights=self.lights)[0] #H,W,1
        return depth
    
    def render_hard_mask(self, camera):
        depth = self.renderer_depth(self.mesh, cameras = camera, lights=self.lights)[0] #H,W,1
        mask = (depth != depth.max()) #H,W,1
        return mask
