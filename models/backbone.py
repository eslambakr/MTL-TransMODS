# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding
from config import Config


# --------------------------------------------------------------------------------------------------------------
#                                           DETR Backbone
# --------------------------------------------------------------------------------------------------------------


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        if (Config.concate and Config.exp_type != "depth_pos_enc_arch4" and Config.exp_type != "baseline") \
                or Config.exp_type == "depth_pos_enc" \
                or Config.exp_type == "shared_rgb_of" or Config.exp_type == "shared_rgb_of_N" \
                or Config.exp_type == "depth_pos_enc_arch2":
            return_layers = {'layer3': "0"}
        # TODO: Eslam: I added it for sematic segmentation, I should recheck it
        if Config.exp_type == "shared_rgb_of_N" and Config.variant == "N_B_N_proj_1_T":
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        # print(tensor_list.decompose()[0].shape)
        # print(tensor_list.decompose()[1].shape)
        # torch.Size([2, 3, 873, 1201])
        # torch.Size([2, 873, 1201])
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)

        # print("backbone output = ", out['0'].decompose()[0].shape)
        # print(out['0'].decompose()[1].shape)
        # dict_keys(['0'])
        # torch.Size([2, 2048, 28, 38])  "tensor"
        # torch.Size([2, 28, 38])   "mask"
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        if Config.exp_type == "depth_pos_enc" or Config.exp_type == "shared_rgb_of" \
                or Config.exp_type == "depth_pos_enc_arch2" or Config.exp_type == "shared_rgb_of_N":
            num_channels = 1024
        # TODO: Eslam: I added it for sematic segmentation, I should recheck it
        if Config.exp_type == "shared_rgb_of_N" and Config.variant == "N_B_N_proj_1_T":
            num_channels = 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        if (Config.exp_type == "depth_pos_enc" and Config.encode_channels) \
                or Config.exp_type == "depth_pos_enc_arch2":
            pos_channel = []

        for name, x in xs.items():
            out.append(x)
            # position encoding
            if (Config.exp_type == "depth_pos_enc" and Config.encode_channels) \
                    or Config.exp_type == "depth_pos_enc_arch2":
                pos.append(self[1](x)[0].to(x.tensors.dtype))
                pos_channel.append(self[1](x)[1].to(x.tensors.dtype))
            else:
                pos.append(self[1](x).to(x.tensors.dtype))

        # print("Joiner = ", pos[-1].shape)
        # torch.Size([2, 256, 28, 38])
        if (Config.exp_type == "depth_pos_enc" and Config.encode_channels) \
                or Config.exp_type == "depth_pos_enc_arch2":
            return out, pos, pos_channel
        return out, pos


# --------------------------------------------------------------------------------------------------------------
#                                           ViT Backbone
# --------------------------------------------------------------------------------------------------------------
import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ViT(nn.Module):
    def __init__(self, Config, channels=3, emb_dropout=0.1):
        super().__init__()
        self.num_channels = Config.backbone_channels
        image_height, image_width = pair(Config.input_size)
        patch_height, patch_width = pair(Config.patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.num_patches = num_patches
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, self.num_channels),
        )

        if Config.ViT_learnable_pos:
            self.pos_embedding = nn.Parameter(torch.randn(num_patches, 1, self.num_channels))
        else:
            self.num_pos_feats = Config.backbone_channels
            self.temperature = 10000

        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, tensor_list: NestedTensor):
        x = self.to_patch_embedding(tensor_list.tensors)
        x = x.permute(1, 0, 2)  # [L, N, C]
        if not Config.ViT_learnable_pos:  # Sinusoidal
            patch_embed = torch.ones((x.shape[1], self.num_patches),
                                     dtype=torch.bool, device=x.device)
            patch_embed = patch_embed.cumsum(1, dtype=torch.float32)
            # Normalize
            eps = 1e-6
            patch_embed = patch_embed / (patch_embed[:, -1:] + eps) * 2 * math.pi

            dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
            dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
            pos_patch = patch_embed[:, :, None] / dim_t
            pos_patch = torch.stack((pos_patch[:, :, 0::2].sin(), pos_patch[:, :, 1::2].cos()), dim=3).flatten(2)
            # pos_patch is (B, num_patches, Config.backbone_channels) (1, 49, 256)
            self.pos_embedding = pos_patch.permute(1, 0, 2)  # [L, N, C]

        x += self.pos_embedding
        x = self.dropout(x)
        # features shape is [batch_size, 2048, 5*15]
        # pos shape is [batch_size, 256, 5*15]
        return x, self.pos_embedding


def build_backbone(args):
    if Config.backbone_type == "DETR":
        position_embedding = build_position_encoding(args)
        train_backbone = args.lr_backbone > 0
        return_interm_layers = args.masks or Config.seg_task_status
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
        model = Joiner(backbone, position_embedding)
        model.num_channels = backbone.num_channels
        # print("backbone.num_channels = ", backbone.num_channels)   "2048"
    elif Config.backbone_type == "ViT":
        model = ViT(Config, channels=3, emb_dropout=0.1)
    return model
