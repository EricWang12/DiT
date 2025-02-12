# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from itertools import repeat
import collections.abc
# from torchviz import make_dot
# from torchview import draw_graph
import visu3d as v3d


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class PatchEmbed3D(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patch_size = (1, patch_size[0], patch_size[1]) # change to 3D
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[1], img_size[0] // patch_size[2]) # change to 1,2 for 3D
        self.num_patches = self.grid_size[0] * self.grid_size[1]        # num_patches remains thet same and double the embeddings
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size,  stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, D, H, W = x.shape
        # assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        # breakpoint()
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        
        x = self.norm(x)
        return x





class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y, gate_mca):
        assert x.shape == y.shape
        B, N, C = x.shape
        qkv_x = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_y = self.qkv(y).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_x, k_x, v_x = qkv_x.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q_y, k_y, v_y = qkv_y.unbind(0)

        attn_x = (q_x @ k_y.transpose(-2, -1)) * self.scale
        attn_x = attn_x.softmax(dim=-1)
        attn_x = self.attn_drop(attn_x)

        attn_y = (q_y @ k_x.transpose(-2, -1)) * self.scale
        attn_y = attn_y.softmax(dim=-1)
        attn_y = self.attn_drop(attn_y)


        x = (attn_x @ v_y).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        y = (attn_y @ v_x).transpose(1, 2).reshape(B, N, C)
        y = self.proj(y)
        y = self.proj_drop(y)

        return torch.concat((gate_mca[:,0,:].unsqueeze(1) * x, gate_mca[:,1,:].unsqueeze(1) * y), dim=1)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_rays(R, t, K, H, W):
    i, j = torch.meshgrid(torch.arange(0, W), torch.arange(0, H))
    i = i.reshape(-1)
    j = j.reshape(-1)
    dirs = torch.stack([(i - K[0, 2]) / K[0, 0], -(j - K[1, 2]) / K[1, 1], -torch.ones_like(i)], dim=-1)
    rays_d = torch.sum(dirs[:, :, None, :] * R[None, None, :, :], dim=-1)
    rays_o = t[None, None, :].repeat(H, W, 1)
    return rays_o, rays_d

def get_rays_vis(R, t, K, H, W):

    world_from_cam = v3d.Transform(R=R, t=t)
    cam_spec = v3d.PinholeCamera(resolution=( H, W), K=K)
    rays = v3d.Camera(spec=cam_spec, world_from_cam=world_from_cam).rays() #2xHxW
    
    return rays

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        breakpoint()
        return embeddings


def encode_camera_pose(P, image_size, device):
    R, t, K = P["R"], P["t"], P["K"]
    if type(R) == torch.Tensor:
        R = R.numpy()
        t = t.numpy()
        K = K.numpy()

    rays = get_rays_vis(R, t, K, image_size, image_size)
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
    pose_emb = torch.from_numpy(pose_emb).float().to(device=device)

    pose_emb = pose_emb.permute(0,4,1,2,3) # N x C x D x H x W

    return pose_emb

class CameraPoseEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, image_size,  hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        # self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        # self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.image_size = image_size
        # self.device = device
        self.conv = nn.Conv3d(144, hidden_size, kernel_size=(1,3,3),  stride=(1, 2, 2), padding=(0,1,1),  bias=True)

        self.linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size * (image_size//2)* (image_size//2), hidden_size, bias=True),
            # nn.SiLU(),
        )

        # mlp_hidden_dim = int(hidden_size * (image_size//2))
        # silu = lambda: nn.SiLU()
        # self.mlp = Mlp(in_features=hidden_size * (image_size//2)* (image_size//2), hidden_features=mlp_hidden_dim, out_features=hidden_size, act_layer=silu, drop=0.1)


    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        
        labels = torch.where(drop_ids[:,None,None,None,None], torch.zeros_like(labels), labels)
        return labels
    

    def forward(self, pose_emb, train, force_drop_ids=None):

        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            pose_emb = self.token_drop(pose_emb, force_drop_ids)

        embeddings = self.conv(pose_emb)
        embeddings = embeddings.permute(0,2,3,4,1)
        embeddings = embeddings.view(embeddings.shape[0], embeddings.shape[1], -1)
        embeddings = self.linear(embeddings)
        return embeddings # N x D x T




#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.crsAttn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=2)
        # breakpoint()
        # norm1 stays the same, layernorm normalize in dimention -1, which is 768 in hidden_size
        N, T, D = x.shape
        x0 = x[:, :T//2, :]
        x1 = x[:, T//2:, :]        
        # self_attn = torch.concat((self.attn(modulate(self.norm1(x0), shift_msa, scale_msa)), self.attn(modulate(self.norm1(x1), shift_msa, scale_msa))), dim=1)
        # x = x + gate_msa.unsqueeze(1) * self_attn


        x0 = x0 + gate_msa[:,0,:].unsqueeze(1) * self.attn(modulate(self.norm1(x0), shift_msa[:,0,:], scale_msa[:,0,:]))
        x1 = x1 + gate_msa[:,1,:].unsqueeze(1) * self.attn(modulate(self.norm1(x1), shift_msa[:,1,:], scale_msa[:,1,:]))
        x = x + self.crsAttn(modulate(self.norm2(x0), shift_mca[:,0,:], scale_mca[:,0,:]), modulate(self.norm2(x1), shift_mca[:,1,:], scale_mca[:,1,:]), gate_mca)
        x = x +  torch.concat((
            gate_mlp[:,0,:].unsqueeze(1) * self.mlp(modulate(self.norm3(x[:, :T//2, :]), shift_mlp[:,0,:], scale_mlp[:,0,:])),
            gate_mlp[:,1,:].unsqueeze(1) * self.mlp(modulate(self.norm3(x[:, T//2:, :]), shift_mlp[:,1,:], scale_mlp[:,1,:]))
        ), dim=1)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT3d(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=3,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.x_embedder = PatchEmbed3D(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.P_embedder = CameraPoseEmbedder(input_size, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        # pos_embed = np.concatenate([pos_embed, pos_embed], axis=0)

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize label embedding table:
        nn.init.normal_(self.P_embedder.conv.weight, std=0.02)
        nn.init.normal_(self.P_embedder.linear[1].weight, std=0.02)


        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[1] # recall that the patch size is  (1, p, p) for 3D
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, P):
        """
        Forward pass of DiT.
        x: (N, C, D, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        P: (N, C, D, H, W) tensor, embedded from {"R": R (2, 3, 3), "t": t (2, 3), "K": K (2, 3, 3)} dictionary of Camera parameters

        """
        # N, C, 2, H, W = x.shape
        # D => self.hidden_size
        x = self.x_embedder(x) + torch.concat([self.pos_embed, self.pos_embed], dim = 1)  # (N, T, D), where T = H * W * 2 / patch_size ** 2
        N, T, D = x.shape
        t = self.t_embedder(t)                   # (N, 2, D)
        t = torch.stack([t,t],1)

        y = self.P_embedder(P, self.training)    # (N, 2, D)

        c = t + y                                # (N, 2, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x[:,:T//2,:], c[:,0,:])                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT3d(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT3d(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT3d(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT3d(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT3d(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT3d(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT3d(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT3d(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_S_4(**kwargs):
    return DiT3d(depth=8, hidden_size=384, patch_size=8, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT3d(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT3d(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT3d(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT3d(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def DiT_M_4(**kwargs):
    return DiT3d(depth=2, hidden_size=384, patch_size=4, num_heads=12, **kwargs)

DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
      'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,  
    'DiT-XL/2': DiT_XL_2,
    'DiT-L/2':  DiT_L_2,  
    'DiT-B/4':  DiT_B_4,
    'DiT-S/4':  DiT_S_4,
       'DiT-B/8':  DiT_B_8,  'DiT-S/2':  DiT_S_2,      'DiT-S/8':  DiT_S_8,
}


def get_camera_rays(R, t, K, H, W):
    # Generate a grid of pixel coordinates in the image plane
    x = torch.linspace(0, W - 1, W)
    y = torch.linspace(0, H - 1, H)
    xx, yy = torch.meshgrid(x, y)
    grid = torch.stack((xx, yy), dim=-1)

    # Reshape the grid into a tensor of size (H * W, 2) and convert to homogeneous coordinates
    grid = grid.reshape(-1, 2)
    homogeneous_grid = torch.cat((grid, torch.ones_like(grid[:, :1])), dim=1)

    # Convert the grid from image coordinates to camera coordinates using the inverse of the intrinsic matrix
    inv_K = torch.inverse(K)
    camera_grid = torch.matmul(inv_K, homogeneous_grid.t()).t()

    # Apply the extrinsic matrix to get the world coordinates of each pixel
    world_grid = torch.matmul(camera_grid, torch.inverse(torch.cat((R, t), dim=1).t()))

    # Compute the direction of each ray by normalizing the world coordinates
    directions = world_grid[:, :3] / torch.norm(world_grid[:, :3], dim=1, keepdim=True)

    return directions

if __name__ == "__main__":
    device = "cuda"
    B = 8
    input_size = 64
    x1 = torch.ones(B, 4, 64, 64).to(device="cuda")
    x2 = torch.zeros(B, 4, 64, 64).to(device="cuda")
    x = torch.stack([x1, x2], dim=2).to(device="cuda")
    time_stamp = torch.randint(0, 1000, (x.shape[0],), device=device)
    y = torch.randint(0, 200, (x.shape[0],), device=device)
    # breakpoint()

    c_d = np.load("/media/exx/8TB1/ewang/CODE/DiT/results/THuman_random_64/0000/camera_pose.npy",  allow_pickle=True).item()
    # breakpoint()
    K0, R0, t0 = c_d[0]["K"], c_d[0]["R"], c_d[0]["T"]
    K1, R1, t1 = c_d[1]["K"], c_d[1]["R"], c_d[1]["T"]

    K = np.stack([K0, K1], axis=0)[None,:]
    R = np.concatenate([R0, R1], axis=0)[None,:]
    t = np.concatenate([t0, t1], axis=0)[None,:]

    K, R, t = [], [], []

    for i in range(0, B*2, 2):
        K.append(np.stack([c_d[i]["K"], c_d[i+1]["K"]], axis=0)[None,:])
        R.append(np.concatenate([c_d[i]["R"], c_d[i+1]["R"]], axis=0)[None,:])
        t.append(np.concatenate([c_d[i]["T"], c_d[i+1]["T"]], axis=0)[None,:])

    # breakpoint()
    P = {"K": np.concatenate(K), "R": np.concatenate(R), "t": np.concatenate(t)}

    model = DiT_M_4(
        input_size=input_size,
        in_channels=4,
    ).to(device="cuda")
    out = model(x,time_stamp , encode_camera_pose(P, input_size, device=device))
    breakpoint()

    # make_dot(out.mean(), params=dict(model.named_parameters())).render("DiT_B_4", format="png")

    from torchview import draw_graph
    # breakpoint()
    model_graph = draw_graph(model, input_data=[x,time_stamp,  encode_camera_pose(P, input_size, device=device)], expand_nested=True, depth=4, device=device)
    model_graph.visual_graph.render("DiT_M_4_torchview", format="jpg")
    breakpoint()



    # C = CameraPoseEmbedder(64,368, True, device="cuda")
    # C = C.to(device="cuda")
    # out = C(K=K[None,:],R=R[None,:],t=t[None,:], train=True)
    # breakpoint()