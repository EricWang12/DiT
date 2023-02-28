
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

# from models import DiT_models
from models3d import DiT_B_4, DiT_models, encode_camera_pose

from diffusion import create_diffusion
from diffusion.image_datasets import THumanDataset
from diffusers.models import AutoencoderKL


import torch
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
import argparse
import random
from tqdm.auto import tqdm


debug_mode = True
use_vae = False
#################################################################################
#                                  Training Loop                                #
#################################################################################


def get_human(image_path, image_idxs=None, transform=None):
    """
    args:
        image_idxs: image indexes of (source, target)
        image_path: list of image paths
        cameras: camera poses
        transform: image transform

    """

    if not image_idxs:
        view_length = len(os.listdir(image_path))
        image_idxs = random.sample(range(0, view_length-1), 2)
    
    object_path = image_path

    image_path1 = os.path.join(object_path, f"{image_idxs[0]}.png")
    image_path2 = os.path.join(object_path, f"{image_idxs[1]}.png")

    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    image1 = torch.tensor(np.array(image1)).float().permute(2, 0, 1) / 255.0
    image2 = torch.tensor(np.array(image2)).float().permute(2, 0, 1) / 255.0

    x = torch.stack([image1, image2], dim=0)

    cameras = np.load( os.path.join(object_path,"camera_pose.npy"),  allow_pickle=True).item()


    K0, R0, t0 = cameras[image_idxs[0]]["K"], cameras[image_idxs[0]]["R"], cameras[image_idxs[0]]["T"]
    K1, R1, t1 = cameras[image_idxs[1]]["K"], cameras[image_idxs[1]]["R"], cameras[image_idxs[1]]["T"]

    K = np.stack([K0, K1], axis=0)[None,:]
    R = np.concatenate([R0, R1], axis=0)[None,:]
    t = np.concatenate([t0, t1], axis=0)[None,:]

    # breakpoint()
    if transform:
        x = transform(x)

    x = x.permute(1,0,2,3)
    P = {"K": K, "R": R, "t": t}
    return  x, P


def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    # breakpoint()

    model = DiT_models[args.model](
        input_size=latent_size,
        in_channels=4,
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"

    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained("/media/exx/8TB1/ewang/CODE/DiT/pretrained_models/vae_diffuser",local_files_only=True).to(device)

    # Create sampling noise:
    for i in tqdm(range(100), position = 0, leave = False):
        z = torch.randn(1, 4, latent_size, latent_size, device=device)
        id = f"{random.randint(512, 512+len(os.listdir(args.image_path))-1):04d}"
        ref, P = get_human( os.path.join(args.image_path, id), args.image_idxs, transform=transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True))
        ref_original = deepcopy(ref)
        ref = ref.unsqueeze(0).to(device)
        # breakpoint()
        ref_x = vae.encode(ref[:,:,0,:,:]).latent_dist.sample().mul_(0.18215)
        ref = torch.stack([ref_x, z], dim=2)
        P =  encode_camera_pose(P, latent_size, device=device)
        P_null = torch.zeros_like(P)
        P = torch.cat([P, P_null], 0)
        model_kwargs = dict(y=P_null, cfg_scale=args.cfg_scale)
        
        ref = torch.cat([ref, ref], 0)

        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, ref.shape, ref, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample
        out_image = torch.concatenate([ref_original.permute(1,0,2,3) , samples.cpu()], dim=0)
        save_image(out_image, f"{args.output_path}/sample_{i}.png", normalize=True, value_range=(-1, 1))
        # breakpoint()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B/4")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[64,256, 512], default=64)
    parser.add_argument("--num-classes", type=int, default=200)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="results/024-DiT-B-4/checkpoints/0095000.pt",
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--image-path", type=str, default="results/THuman_random_64/test", help="path to the image folder")
    parser.add_argument("--image-idxs", type=int, nargs=2, default=None, help="image indexes of (source, target)")
    parser.add_argument("--output-path", type=str, default="./output/out_vae", help="path to the image folder")

    args = parser.parse_args()
    main(args)

# def main(args):
#     """
#     Trains a new DiT model.
#     """
#     assert torch.cuda.is_available(), "Training currently requires at least one GPU."

#     # Setup DDP:
#     dist.init_process_group("nccl")
#     assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
#     rank = dist.get_rank()
#     device = rank % torch.cuda.device_count()
#     seed = args.global_seed * dist.get_world_size() + rank
#     torch.manual_seed(seed)
#     print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

#     # Setup an experiment folder:
#     if rank == 0:
#         os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
#         experiment_index = len(glob(f"{args.results_dir}/*"))-1
#         model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
#         if debug_mode:
#             experiment_index -= 1
#         experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
#         checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
#         os.makedirs(checkpoint_dir, exist_ok=True)
#         logger = create_logger(experiment_dir)
#         logger.info(f"Experiment directory created at {experiment_dir}")
#     else:
#         logger = create_logger("./logs.txt")

#     # Create model:
#     assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
#     if use_vae:
#         latent_size = args.image_size // 8
#     else:
#         latent_size = args.image_size

#     model = DiT_models[args.model](
#         input_size=latent_size,
#     )
#     # Note that parameter initialization is done within the DiT constructor
#     ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
#     model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
#     diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
#     # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
#     logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

#     # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
#     opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

#     # Setup data:
#     transform = transforms.Compose([
#         # transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
#         transforms.RandomHorizontalFlip(),
#         # transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
#     ])
#     dataset = ImageFolder(args.data_path, transform=transform)
#     dataset = THumanDataset(args.data_path, transform=transform)

#     sampler = DistributedSampler(
#         dataset,
#         num_replicas=dist.get_world_size(),
#         rank=rank,
#         shuffle=True,
#         seed=args.global_seed
#     )
#     loader = DataLoader(
#         dataset,
#         batch_size=int(args.global_batch_size // dist.get_world_size()),
#         shuffle=False,
#         sampler=sampler,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         drop_last=True
#     )
#     logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

#     # Prepare models for training:
#     update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
#     model.train()  # important! This enables embedding dropout for classifier-free guidance
#     ema.eval()  # EMA model should always be in eval mode

#     # Variables for monitoring/logging purposes:
#     train_steps = 0
#     log_steps = 0
#     running_loss = 0
#     start_time = time()

#     logger.info(f"Training for {args.epochs} epochs...")
#     for epoch in range(args.epochs):
        # sampler.set_epoch(epoch)
        # logger.info(f"Beginning epoch {epoch}...")
        # for x, y in loader:
            # x = x.to(device)