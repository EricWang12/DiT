import visu3d as v3d
import torch
import numpy as np
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
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
    TexturesUV,
    TexturesVertex
)

def intrinsics_to_matrix(fx, fy, cx, cy, skew=0):
    """
    Converts camera intrinsic parameters to a 3x3 intrinsic matrix.
    Assumes zero skew.
    
    Args:
        fx (float): focal length in x-direction
        fy (float): focal length in y-direction
        cx (float): principal point x-coordinate
        cy (float): principal point y-coordinate
        skew (float): skew coefficient (default 0)
        
    Returns:
        np.ndarray: 3x3 intrinsic matrix
    """
    K = np.array([[fx, skew, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)
    return K


R, T = look_at_view_transform(-.7, 0.5, 0) 
device = torch.device("cpu")
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
K = cameras.compute_projection_matrix(cameras.znear, cameras.zfar, cameras.fov, cameras.aspect_ratio, cameras.degrees)[0]

fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 3]
cy = K[1, 3]

H, W = 64, 64
# breakpoint()
world_from_cam = v3d.Transform(R=cameras.R.numpy(), t=cameras.T.numpy())
cam_spec = v3d.PinholeCamera(resolution=(H, W), K=intrinsics_to_matrix(fx, fy, cx, cy))
rays = v3d.Camera(spec=cam_spec, world_from_cam=world_from_cam).rays()

# breakpoint()

def posenc_nerf(x, min_deg=0, max_deg=15):
    """Concatenate x and its positional encodings, following NeRF."""
    if min_deg == max_deg:
        return x
    scales = np.array([2**i for i in range(min_deg, max_deg)])
    xb = np.reshape(
        (x[..., None, :] * scales[:, None]), list(x.shape[:-1]) + [-1])
    emb = np.sin(np.concatenate([xb, xb + np.pi / 2.], axis=-1))
    return np.concatenate([x, emb], axis=-1)

pose_emb_pos = posenc_nerf(rays.pos, min_deg=0, max_deg=15)
pose_emb_dir = posenc_nerf(rays.dir, min_deg=0, max_deg=8)
pose_emb = np.concatenate([pose_emb_pos, pose_emb_dir], axis=-1)

breakpoint()