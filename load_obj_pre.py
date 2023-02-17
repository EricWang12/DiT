import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

import pytorch3d

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
from tqdm import tqdm
# add path for demo utils functions 
import sys
import os

# Load the .obj file
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Set paths
DATA_DIR = "results/THuman_random_64/"
# obj_filename = os.path.join(DATA_DIR, "/media/exx/8TB1/ewang/CODE/ICON/data/thuman2/scans/0525/0525.obj")
# obj_filename = "/media/exx/8TB1/ewang/CODE/dataset/THuman/0525/0525.obj"
# mesh = load_objs_as_meshes([obj_filename], device=device)



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


for file_id in tqdm(range(525)):
    obj_filename = f"//media/exx/8TB1/ewang/CODE/dataset/THuman/{file_id:04d}/{file_id:04d}.obj"
    mesh = load_objs_as_meshes([obj_filename], device=device)
    # mesh = mesh.extend(1)
    #
    os.makedirs(f"{DATA_DIR}{file_id:04d}", exist_ok=True)
    camera_dict = {}
    for i, angle in enumerate(tqdm(range(-180, 181, 2), leave=False)):
        # print(angle)
        R, T = look_at_view_transform(0.8, np.random.randint(-180, 181), np.random.randint(-180, 181)) 
        # breakpoint()
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        K =  cameras.compute_projection_matrix(cameras.znear, cameras.zfar, cameras.fov, cameras.aspect_ratio, cameras.degrees)[0].cpu().numpy()
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 3]
        cy = K[1, 3]

        K = intrinsics_to_matrix(fx, fy, cx, cy)
        raster_settings = RasterizationSettings(
            image_size=64, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device, 
                cameras=cameras,
            )
        )

        images = renderer(mesh)
        out = images[0, ..., :3].cpu().numpy()

        image = Image.fromarray(np.uint8(out * 255))

        # Save the image to disk
        image.save(f"{DATA_DIR}{file_id:04d}/{i}.png")
        camera_dict[i] = {"K": K, "R": R.cpu().numpy(), "T": T.cpu().numpy()}
    np.save(f"{DATA_DIR}{file_id:04d}/camera_pose.npy", camera_dict)
    # breakpoint()
