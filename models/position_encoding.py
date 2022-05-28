# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor
from config import Config


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        # Mask shape is (B, H, W)
        not_mask = ~mask
        if Config.exp_type == "depth_pos_enc" and Config.TPE_type == "NHW_C" and False:
            b, h, w = not_mask.shape
            # TODO: Eslam this 2 should be generic
            not_mask = torch.ones((b, h, w, 2), dtype=torch.float32, device=x.device)
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
            n_embed = not_mask.cumsum(3, dtype=torch.float32)
            if self.normalize:
                eps = 1e-6
                y_embed = y_embed / (y_embed[:, -1:, :, :] + eps) * self.scale
                x_embed = x_embed / (x_embed[:, :, -1:, :] + eps) * self.scale
                n_embed = n_embed / (n_embed[:, :, :, -1:] + eps) * self.scale
            xy_pos_feats = self.num_pos_feats - int(self.num_pos_feats/3)
            n_pos_feats = (self.num_pos_feats - xy_pos_feats)*2
            xy_dim_t = torch.arange(xy_pos_feats, dtype=torch.float32, device=x.device)
            xy_dim_t = self.temperature ** (2 * (xy_dim_t // 2) / xy_pos_feats)
            n_dim_t = torch.arange(n_pos_feats, dtype=torch.float32, device=x.device)
            n_dim_t = self.temperature ** (2 * (n_dim_t // 2) / n_pos_feats)
            pos_x = x_embed[:, :, :, :, None] / xy_dim_t
            pos_y = y_embed[:, :, :, :, None] / xy_dim_t
            pos_n = n_embed[:, :, :, :, None] / n_dim_t
            pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
            pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
            pos_n = torch.stack((pos_n[:, :, :, :, 0::2].sin(), pos_n[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
            pos = torch.cat((pos_y, pos_x, pos_n), dim=4).permute(0, 4, 1, 2, 3)
            return pos
        # Accumulative summation to order the places of the pixels
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # W_k
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        if ((Config.exp_type == "depth_pos_enc" or Config.exp_type == "cat_4frames_res34") and Config.encode_channels)\
                or Config.exp_type == "depth_pos_enc_arch2":
            """
            channel_embed = torch.ones(size=(y_embed.size(0), y_embed.size(1), 256),
                                       dtype=torch.float32, device=x.device)
            channel_embed = channel_embed.cumsum(2)
            """
            channel_embed = torch.cat((torch.zeros(size=(y_embed.size(0), y_embed.size(1), 128),
                                                   dtype=torch.float32, device=x.device),
                                       torch.ones(size=(y_embed.size(0), y_embed.size(1), 128),
                                                  dtype=torch.float32, device=x.device)), dim=2)
            if self.normalize:
                channel_embed = channel_embed / (channel_embed[:, :, -1:] + eps) * self.scale
            dim_t = torch.arange(y_embed.size(2), dtype=torch.float32, device=x.device)
            dim_t = self.temperature ** (2 * (dim_t // 2) / y_embed.size(2))
            pos_ch = channel_embed[:, :, :, None] / dim_t  # (32, 10, 256, 30)
            if Config.exp_type == "cat_4frames_res34":
                pos_ch = torch.cat((pos_ch[:, :, :, 0::2].sin(), pos_ch[:, :, :, 1::2].cos()), dim=-1).flatten(3)
            else:
                pos_ch = torch.stack((pos_ch[:, :, :, 0::2].sin(), pos_ch[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos_ch = pos_ch.permute(0, 2, 1, 3)

            return pos, pos_ch

        # print("PositionEmbeddingSine = ", pos.shape)
        # torch.Size([2, 256, 28, 38])
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    if Config.exp_type == "shared_rgb_of_N" and Config.variant == "N_B_1_T":
        N_steps = args.hidden_dim*Config.num_of_repeated_blocks // 2
    else:
        N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
