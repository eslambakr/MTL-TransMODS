# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from config import Config
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer
from collections import defaultdict


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False, backbone2=None,
                 transformer2=None, transformer3=None, args=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        if Config.shared_dec_shared_q:
            self.trans_out = None
            from models.matcher import HungarianMatcher
            self.matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        self.num_queries = num_queries
        self.transformer = transformer
        self.transformer2 = transformer2
        self.transformer3 = transformer3
        if Config.exp_type == "shared_rgb_of_N" and Config.variant == "N_B_N_T_proj_1_T" and Config.sharing == False:
            hidden_dim = transformer[0].d_model
        else:
            hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim

        if Config.exp_type == "depth_pos_enc_arch4" and Config.concate:
            #self.class_embed = nn.Linear(hidden_dim * 2, num_classes + 1)
            self.class_embed = nn.Linear(num_queries, num_classes + 1)
            #self.bbox_embed = MLP(hidden_dim * 2, hidden_dim * 2, 4, 3)
            self.bbox_embed = MLP(num_queries, num_queries, 4, 3)
            self.query_embed1 = nn.Embedding(num_queries, hidden_dim)
            self.query_embed2 = nn.Embedding(num_queries, hidden_dim)
            #self.query_embed3 = nn.Embedding(num_queries, hidden_dim*2)
            self.query_embed3 = nn.Embedding(hidden_dim, num_queries)
        elif Config.exp_type == "depth_pos_enc_arch4":
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
            self.query_embed1 = nn.Embedding(num_queries, hidden_dim)
            self.query_embed2 = nn.Embedding(num_queries, hidden_dim)
            self.query_embed3 = nn.Embedding(num_queries, hidden_dim)
        elif Config.exp_type == "depth_pos_enc_arch2":
            if Config.TPE_type == "NC_HW" and False:
                # TODO:Eslam make this generic
                self.query_embed = nn.Embedding(num_queries, 300)
                self.class_embed = nn.Linear(300, num_classes + 1)
                self.bbox_embed = MLP(300, 300, 4, 3)
            elif Config.TPE_type == "N_CHW":
                self.query_embed = nn.Embedding(num_queries, hidden_dim*300)
                self.class_embed = nn.Linear(hidden_dim*300, num_classes + 1)
                self.bbox_embed = MLP(hidden_dim*300, hidden_dim*300, 4, 3)
            else:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
                self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
                self.query_embed = nn.Embedding(num_queries, hidden_dim)
        elif Config.exp_type == "shared_rgb_of_N" and Config.concate:
            if Config.variant == "N_B_N_T" and Config.sub_variant == "concat":
                self.class_embed = nn.Linear(hidden_dim*Config.num_of_repeated_blocks, num_classes + 1)
                self.bbox_embed = MLP(hidden_dim*Config.num_of_repeated_blocks,
                                      hidden_dim*Config.num_of_repeated_blocks, 4, 3)
                self.query_embed = nn.Embedding(num_queries, hidden_dim)
            elif Config.variant == "N_B_N_T" and Config.sub_variant == "FC":
                self.fuse_queries = nn.Linear(hidden_dim * Config.num_of_repeated_blocks, hidden_dim)
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
                self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
                self.query_embed = nn.Embedding(num_queries, hidden_dim)
            elif Config.variant == "N_B_N_T" and Config.sub_variant == "NQ_C":
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
                self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
                self.query_embed = []
                for i in range(Config.num_of_repeated_blocks):
                    self.query_embed.append(nn.Embedding(num_queries, hidden_dim))
                self.query_embed = nn.ModuleList(self.query_embed)

                if Config.sharing:
                    self.class_embed_aux_Q = nn.Linear(hidden_dim, num_classes + 1)
                    self.bbox_embed_aux_Q = MLP(hidden_dim, hidden_dim, 4, 3)

                from models.transformer import TransformerEncoderLayer
                from models.transformer import TransformerEncoder
                self.fusion_enc_layer = []
                for i in range(6):
                    self.fusion_enc_layer.append(TransformerEncoder(TransformerEncoderLayer(d_model=256, nhead=8,
                                                                                            dim_feedforward=1024,
                                                                                            fuse=False, dropout=0.1,
                                                                                            activation="relu",
                                                                                            normalize_before=False),
                                                                    num_layers=Config.num_fusion_layers, fuse=True))
                self.fusion_enc_layer = nn.ModuleList(self.fusion_enc_layer)
                import math
                self.eps = 1e-6
                self.scale = 2 * math.pi
                self.temperature = 10000
            elif Config.variant == "N_B_N_T" and Config.sub_variant == "NQ_C_Decoder":
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
                self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
                self.query_embed = []
                for i in range(Config.num_of_repeated_blocks):
                    self.query_embed.append(nn.Embedding(num_queries, hidden_dim))
                self.query_embed = nn.ModuleList(self.query_embed)
                self.fusion_query_embed = nn.Embedding(num_queries, hidden_dim)

                from models.transformer import TransformerDecoderLayer
                from models.transformer import TransformerDecoder
                self.fusion_dec_layer = []
                for i in range(6):
                    self.fusion_dec_layer.append(TransformerDecoder(TransformerDecoderLayer(d_model=256, nhead=8,
                                                                                            dim_feedforward=1024,
                                                                                            dropout=0.1,
                                                                                            activation="relu",
                                                                                            normalize_before=False),
                                                                    num_layers=Config.num_fusion_layers))
                self.fusion_dec_layer = nn.ModuleList(self.fusion_dec_layer)
                import math
                self.eps = 1e-6
                self.scale = 2 * math.pi
                self.temperature = 10000
            elif Config.variant == "N_B_1_T":
                self.class_embed = nn.Linear(hidden_dim * Config.num_of_repeated_blocks, num_classes + 1)
                self.bbox_embed = MLP(hidden_dim * Config.num_of_repeated_blocks,
                                      hidden_dim * Config.num_of_repeated_blocks, 4, 3)
                self.query_embed = nn.Embedding(num_queries, hidden_dim * Config.num_of_repeated_blocks)
            elif Config.variant == "N_B_proj_1_T" or Config.variant == "N_B_N_proj_1_T":
                if Config.seg_task_status:
                    from models.semantic_seg.seg_head import TransformerHead
                    if args.backbone == "resnet18":
                        self.seg_head = TransformerHead(c1_channels=32, hid_dim=32, args=args)
                    elif args.backbone == "resnet50":
                        self.seg_head = TransformerHead(c1_channels=128, hid_dim=32, args=args)
                if Config.det_task_status:
                    self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
                    self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
                if Config.MTL:
                    self.query_embed = []
                    if Config.det_task_status:
                        self.query_embed.append(nn.Embedding(num_queries[0], hidden_dim))
                    else:
                        self.query_embed.append(0)
                    if Config.seg_task_status:
                        self.query_embed.append(nn.Embedding(num_queries[1], hidden_dim))
                    else:
                        self.query_embed.append(0)
                    self.query_embed = nn.ModuleList(self.query_embed)
                else:
                    self.query_embed = nn.Embedding(num_queries, hidden_dim)
            elif Config.variant == "N_B_N_T_proj_1_T":
                self.class_embed = nn.Linear(num_queries, num_classes + 1)
                self.bbox_embed = MLP(num_queries, num_queries, 4, 3)
                self.query_embed = []
                if Config.sharing:
                    self.query_embed = nn.Embedding(num_queries, hidden_dim)
                else:
                    for i in range(Config.num_of_repeated_blocks):
                        self.query_embed.append(nn.Embedding(num_queries, hidden_dim))
                    self.query_embed = nn.ModuleList(self.query_embed)
                self.query_embed2 = nn.Embedding(hidden_dim, num_queries)
        else:
            if Config.seg_task_status:
                from models.semantic_seg.seg_head import TransformerHead
                if args.backbone == "resnet18":
                    self.seg_head = TransformerHead(c1_channels=64, hid_dim=32, args=args)
                elif args.backbone == "resnet50":
                    self.seg_head = TransformerHead(c1_channels=256, hid_dim=32, args=args)
            if Config.det_task_status:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
                self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
            if Config.MTL:
                self.query_embed = []
                if Config.det_task_status:
                    self.query_embed.append(nn.Embedding(num_queries[0], hidden_dim))
                else:
                    self.query_embed.append(0)
                if Config.seg_task_status:
                    self.query_embed.append(nn.Embedding(num_queries[1], hidden_dim))
                else:
                    self.query_embed.append(0)
                self.query_embed = nn.ModuleList(self.query_embed)
            else:
                self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if (Config.exp_type == "depth_pos_enc" and Config.TPE_type == "HW_NC") or Config.exp_type == "shared_rgb_of"\
                or Config.exp_type == "shared_rgb_of_N":
            self.input_proj = nn.Conv2d(backbone.num_channels, int(hidden_dim/2), kernel_size=1)
        elif Config.exp_type == "cat_4frames_res34":
            self.input_proj = nn.Conv2d(backbone.num_channels, int(hidden_dim / 4), kernel_size=1)
        else:
            self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        if Config.exp_type == "shared_rgb_of_N":
            if Config.variant == "N_B_proj_1_T" or Config.variant == "N_B_N_T_proj_1_T":
                self.fuse_proj = nn.Conv2d(hidden_dim*Config.num_of_repeated_blocks, hidden_dim, kernel_size=1)
            elif Config.variant == "N_B_N_proj_1_T":
                self.fuse_proj = nn.Conv2d(hidden_dim, int(hidden_dim/Config.num_of_repeated_blocks), kernel_size=1)

        if (Config.exp_type == "depth_pos_enc_arch2" and not Config.sharing)\
                or Config.exp_type == "depth_pos_enc_arch4":
            self.input_proj2 = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.backbone2 = backbone2
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor, targets=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
               - targets: in case of activating "shared_dec_shared_q" only!

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        features = []
        pos = []
        if Config.exp_type == "depth_pos_enc_arch4":
            # 1
            features1, pos1 = self.backbone(samples[0])
            src1, mask1 = features1[-1].decompose()
            src1 = self.input_proj(src1)
            mask = mask1
            assert mask is not None
            pos.append(pos1[-1])
            hs1 = self.transformer(src1, mask, self.query_embed1.weight, pos[-1])[0]
            # 2
            features2, _ = self.backbone2(samples[1])
            src2, mask2 = features2[-1].decompose()
            src2 = self.input_proj2(src2)
            hs2 = self.transformer2(src2, mask, self.query_embed2.weight, pos[-1])[0]
            if Config.concate:
                hs_1_2 = torch.cat((hs1, hs2), dim=3)  # (6, 32, 100, 512)
            else:
                hs_1_2 = hs1 + hs2  # (6, 32, 100, 256)

            if Config.encode_channels:
                # Channel Encoding
                channel_embed = torch.cat((torch.zeros(size=(hs_1_2.size(1), int(hs_1_2.size(3) / 2)),
                                                       dtype=torch.float32, device=hs_1_2.device),
                                           torch.ones(size=(hs_1_2.size(1), int(hs_1_2.size(3) / 2)),
                                                      dtype=torch.float32, device=hs_1_2.device)), dim=1)
                scale = 6.283185307179586
                eps = 1e-06
                temperature = 10000
                channel_embed = channel_embed / (channel_embed[:, -1:] + eps) * scale
                dim_t = torch.arange(hs_1_2.size(2), dtype=torch.float32, device=hs_1_2.device)
                dim_t = temperature ** (2 * (dim_t // 2) / hs_1_2.size(2))
                pos_ch = channel_embed[:, :, None] / dim_t
                pos_ch = torch.stack((pos_ch[:, :, 0::2].sin(), pos_ch[:, :, 1::2].cos()), dim=3).flatten(2)
            else:
                pos_ch = None
            hs_1_2 = hs_1_2[-1]
            hs = self.transformer3(hs_1_2,
                                   mask=torch.zeros((hs_1_2.size(0), hs_1_2.size(2)),
                                                    dtype=torch.bool, device=hs_1_2.device)
                                   , query_embed=self.query_embed3.weight, pos_embed=pos_ch, fusion_transformer=True)[0]

        elif Config.exp_type == "depth_pos_enc_arch2":
            features1, pos1, pos_channel = self.backbone(samples[0])
            src1, mask1 = features1[-1].decompose()
            src1 = self.input_proj(src1)
            # position from the second image is not used
            if Config.sharing:
                features2, _, _ = self.backbone(samples[1])
                src2, mask2 = features2[-1].decompose()
                src2 = self.input_proj(src2)
            else:
                features2, _, _ = self.backbone2(samples[1])
                src2, mask2 = features2[-1].decompose()
                src2 = self.input_proj2(src2)
            mask = mask1
            pos.append(pos1[-1])

        elif Config.exp_type == "depth_pos_enc" and (Config.encode_channels or Config.TPE_type == "NHW_C"):
            if Config.TPE_type == "NHW_C":
                features1, pos1 = self.backbone(samples[0])
            elif Config.TPE_type == "HW_NC":
                features1, pos1, pos_channel = self.backbone(samples[0])
            src1, mask1 = features1[-1].decompose()
            src1 = self.input_proj(src1)
            # position from the second image is not used
            if Config.TPE_type == "NHW_C":
                features2, _ = self.backbone(samples[1])
            elif Config.TPE_type == "HW_NC":
                features2, _, _ = self.backbone(samples[1])
            src2, mask2 = features2[-1].decompose()
            src2 = self.input_proj(src2)
            pos.append(pos1[-1])
            if Config.TPE_type == "HW_NC":
                mask = mask1
                src = torch.cat((src1, src2), dim=1)
                src = src + pos_channel[-1]
            elif Config.TPE_type == "NHW_C":
                mask = torch.cat((mask1.unsqueeze(-1), mask1.unsqueeze(-1)), dim=-1)
                src = torch.cat((src1.unsqueeze(-1), src2.unsqueeze(-1)), dim=-1)  # [b, C, H, W, N]
                # TODO: Eslam Check the 3D Encoding and remove this work around.
                pos = [torch.cat((pos1[-1].unsqueeze(-1), pos1[-1].unsqueeze(-1)), dim=-1)]

        elif Config.exp_type == "depth_pos_enc":
            features1, pos1 = self.backbone(samples[0])
            src1, mask1 = features1[-1].decompose()
            src1 = self.input_proj(src1)
            # position from the second image is not used
            features2, _ = self.backbone(samples[1])
            src2, mask2 = features2[-1].decompose()
            src2 = self.input_proj(src2)
            src = torch.cat((src1, src2), dim=1)
            mask = mask1
            pos.append(pos1[-1])
        elif Config.exp_type == "shared_backbone":
            if Config.concate:
                features1, pos1 = self.backbone(samples[0])
                src1, mask1 = features1[-1].decompose()
                features2, pos2 = self.backbone(samples[1])
                src2, mask2 = features2[-1].decompose()
                features.append(NestedTensor(torch.cat((src1, src2), dim=1), mask1))
            else:
                features1, pos1 = self.backbone(samples[0])
                src1, mask1 = features1[-1].decompose()
                features2, pos2 = self.backbone(samples[1])
                src2, mask2 = features2[-1].decompose()
                features3, pos3 = self.backbone(samples[2])
                src3, mask3 = features3[-1].decompose()
                features4, pos4 = self.backbone(samples[3])
                src4, mask4 = features4[-1].decompose()

                # features.append(NestedTensor(torch.cat((src1, src2), dim=1), mask1))
                # pos.append(torch.cat((pos1[-1], pos2[-1]), dim=1))
                features.append(NestedTensor(src1+src2+src3+src4, mask1))
            pos.append(pos1[-1])
        elif Config.exp_type == "shared_rgb_of":
            features1, pos1 = self.backbone(samples[0])
            src1, mask1 = features1[-1].decompose()
            src1 = self.input_proj(src1)

            features2, pos2 = self.backbone2(samples[1])
            src2, mask2 = features2[-1].decompose()
            src2 = self.input_proj(src2)
            src = torch.cat((src1, src2), dim=1)
            mask = mask1
            # features.append(NestedTensor(src1+src2, mask1))
            pos.append(pos1[-1])
        elif Config.exp_type == "shared_rgb_of_N":
            if Config.pre_training_coco:
                if not isinstance(samples, NestedTensor):
                    samples = nested_tensor_from_tensor_list(samples)
                rgb_feature, pos1 = self.backbone(samples)
                src1, mask1 = rgb_feature[-1].decompose()
                src1 = self.input_proj(src1)
                src = torch.cat((src1, src1), dim=1)
                # TODO: Eslam make it generic
                #features = [src]
                features = [src, src]
                #features = [src, src, src, src]
            else:
                features = []
                for i in range(Config.num_of_repeated_blocks):
                    rgb_feature, pos1 = self.backbone(samples[2*i])
                    src1, mask1 = rgb_feature[-1].decompose()
                    src1 = self.input_proj(src1)
                    features2, _ = self.backbone2(samples[(2*i)+1])
                    src2, _ = features2[-1].decompose()
                    src2 = self.input_proj(src2)
                    src = torch.cat((src1, src2), dim=1)
                    features.append(src)

            mask = mask1
            pos.append(pos1[-1])
        elif Config.exp_type == "cat_4frames_res34":
            features1, pos1 = self.backbone(samples[0])
            src1, mask1 = features1[-1].decompose()
            src1 = self.input_proj(src1)
            features2, pos2 = self.backbone(samples[1])
            src2, mask2 = features2[-1].decompose()
            src2 = self.input_proj(src2)
            features3, pos3 = self.backbone(samples[2])
            src3, mask3 = features3[-1].decompose()
            src3 = self.input_proj(src3)
            features4, pos4 = self.backbone(samples[3])
            src4, mask4 = features4[-1].decompose()
            src4 = self.input_proj(src4)

            src = torch.cat((src1, src2, src3, src4), dim=1)
            mask = mask1
            pos.append(pos1[-1])
        else:
            if not isinstance(samples, NestedTensor):
                samples = nested_tensor_from_tensor_list(samples)
            features, pos = self.backbone(samples)
            # features shape is [batch_size, 2048, 5, 15]
            # pos shape is [batch_size, 256, 5, 15]

        if Config.exp_type != "depth_pos_enc" and Config.exp_type != "shared_rgb_of"\
                and Config.exp_type != "cat_4frames_res34" and Config.exp_type != "depth_pos_enc_arch2"\
                and Config.exp_type != "depth_pos_enc_arch4" and Config.exp_type != "shared_rgb_of_N"\
                and Config.backbone_type == "DETR":
            src, mask = features[-1].decompose()
            # src shape is [batch_size, 2048, 5, 15]
            # mask shape is [batch_size, 5, 15]

        if Config.backbone_type == "DETR":
            assert mask is not None
        if Config.exp_type == "depth_pos_enc" or Config.exp_type == "shared_rgb_of"\
                or Config.exp_type == "cat_4frames_res34":
            hs = self.transformer(src, mask, self.query_embed.weight, pos[-1])[0]
        elif Config.exp_type == "shared_rgb_of_N":
            if Config.variant == "N_B_N_T":
                trans_features = []
                for i in range(Config.num_of_repeated_blocks):
                    trans_features.append(self.transformer(features[i], mask, self.query_embed[i].weight, pos[-1])[0])

                if Config.sub_variant == "concat":
                    hs = torch.cat(trans_features, dim=-1)
                elif Config.sub_variant == "FC":
                    hs = torch.cat(trans_features, dim=-1)  # [6,b,Q,NC]
                    hs = self.fuse_queries(hs)
                elif Config.sub_variant == "NQ_C":
                    if Config.aux_q:
                        # Intermediate queries Aux. Loss
                        out_aux_intermediate_q = []
                        for i in range(len(trans_features)):
                            outputs_class = self.class_embed_aux_Q(trans_features[i])
                            outputs_coord = self.bbox_embed_aux_Q(trans_features[i]).sigmoid()
                            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
                            out_aux_intermediate_q.append(out)

                    hs = torch.cat(trans_features, dim=2)  # concat on the Queries dimension. [6,b,NQ,C]
                    # Add Positional Encoding
                    _, b, NQ, C = hs.size()
                    channel_embed = torch.ones(size=(Config.num_of_repeated_blocks, b), dtype=torch.float32, device=hs.device)
                    channel_embed = channel_embed.cumsum(1, dtype=torch.float32)
                    channel_embed = channel_embed / (channel_embed[:, -1:] + self.eps) * self.scale
                    dim_t = torch.arange(256, dtype=torch.float32, device=hs.device)
                    dim_t = self.temperature ** (2 * (dim_t // 2) / 256)
                    pos_ch = channel_embed[:, :, None] / dim_t
                    pos_ch = torch.stack((pos_ch[:, :, 0::2].sin(), pos_ch[:, :, 1::2].cos()), dim=3).flatten(2)  # [N,b,256]
                    pos = []
                    for i in range(Config.num_of_repeated_blocks):
                        pos.append(pos_ch[i].unsqueeze(0).repeat(100, 1, 1))
                    pos_ch = torch.cat(pos, dim=0)  # [NQ,b,256]

                    hs_fusion = []
                    for i, enc_layer in enumerate(self.fusion_enc_layer):
                        hs_fusion.append(enc_layer(hs[i].permute(1, 0, 2),                     # [NQ,b,C]
                                                   pos=pos_ch).permute(1, 0, 2).unsqueeze(0))  # [b, Q,C]
                    hs = torch.cat(hs_fusion, dim=0)
                elif Config.sub_variant == "NQ_C_Decoder":
                    if Config.aux_q:
                        # Intermediate queries Aux. Loss
                        out_aux_intermediate_q = []
                        for i in range(len(trans_features)):
                            outputs_class = self.class_embed(trans_features[i])
                            outputs_coord = self.bbox_embed(trans_features[i]).sigmoid()
                            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
                            out_aux_intermediate_q.append(out)

                    hs = torch.cat(trans_features, dim=2)  # concat on the Queries dimension. [6,b,NQ,C]
                    # Add Positional Encoding for Queries
                    _, b, NQ, C = hs.size()
                    channel_embed = torch.ones(size=(Config.num_of_repeated_blocks, b), dtype=torch.float32, device=hs.device)
                    channel_embed = channel_embed.cumsum(1, dtype=torch.float32)
                    channel_embed = channel_embed / (channel_embed[:, -1:] + self.eps) * self.scale
                    dim_t = torch.arange(256, dtype=torch.float32, device=hs.device)
                    dim_t = self.temperature ** (2 * (dim_t // 2) / 256)
                    pos_ch = channel_embed[:, :, None] / dim_t
                    pos_ch = torch.stack((pos_ch[:, :, 0::2].sin(), pos_ch[:, :, 1::2].cos()), dim=3).flatten(2)  # [N,b,256]
                    pos = []
                    for i in range(Config.num_of_repeated_blocks):
                        pos.append(pos_ch[i].unsqueeze(0).repeat(100, 1, 1))
                    pos_queries = torch.cat(pos, dim=0)  # [NQ,b,256]

                    hs_fusion = []
                    memory = hs[0].permute(1, 0, 2)
                    bs = memory.size(1)
                    query_embed = self.fusion_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
                    tgt = torch.zeros_like(query_embed)
                    for i, dec_layer in enumerate(self.fusion_dec_layer):
                        memory = hs[i].permute(1, 0, 2)
                        mask = torch.zeros((memory.size(1), memory.size(0)),
                                           dtype=torch.bool, device=memory.device)
                        hs_fusion.append(dec_layer(tgt=tgt, memory=memory,  # [NQ,b,C]
                                                   memory_key_padding_mask=mask, pos=pos_queries,
                                                   query_pos=query_embed).permute(1, 0, 2).unsqueeze(0))  # [b, Q,C]
                    hs = torch.cat(hs_fusion, dim=0)

                elif Config.sub_variant == "add":
                    hs = trans_features[0]
                    for i in range(1, Config.num_of_repeated_blocks):
                        hs += trans_features[i]
                else:
                    hs = trans_features[0]
                    for i in range(1, Config.num_of_repeated_blocks):
                        hs += trans_features[i]
            elif Config.variant == "N_B_1_T":
                if Config.concate:
                    # TODO: Add Temporal Encoding
                    src = torch.cat(features, dim=1)
                else:
                    src = features[0]
                    for i in range(1, Config.num_of_repeated_blocks):
                        src += features[i]
                hs = self.transformer(src, mask, self.query_embed.weight, pos[-1])[0]
            elif Config.variant == "N_B_proj_1_T" and Config.sub_variant == "decouple_curr_prev":
                src = features[0]  # current frame
                src2 = torch.cat(features, dim=1)  # current+prev. frames
                src2 = self.fuse_proj(src2)
                hs = self.transformer(src, mask, self.query_embed.weight, pos[-1], src2=src2)[0]
            elif Config.variant == "N_B_proj_1_T":
                src = torch.cat(features, dim=1)
                if Config.encode_channels:
                    # Channel Encoding
                    # TODO: Eslam should be generic according to "num_of_repeated_blocks"
                    channel_embed = torch.cat((torch.zeros(size=(src.size(0), src.size(2), self.hidden_dim),
                                                           dtype=torch.float32, device=src.device),
                                               torch.ones(size=(src.size(0), src.size(2), self.hidden_dim),
                                                          dtype=torch.float32, device=src.device),
                                               torch.ones(size=(src.size(0), src.size(2), self.hidden_dim),
                                                          dtype=torch.float32, device=src.device) * 2,
                                               torch.ones(size=(src.size(0), src.size(2), self.hidden_dim),
                                                          dtype=torch.float32, device=src.device) * 3,
                                               torch.ones(size=(src.size(0), src.size(2), self.hidden_dim),
                                                          dtype=torch.float32, device=src.device) * 4,
                                               torch.ones(size=(src.size(0), src.size(2), self.hidden_dim),
                                                          dtype=torch.float32, device=src.device) * 5,
                                               torch.ones(size=(src.size(0), src.size(2), self.hidden_dim),
                                                          dtype=torch.float32, device=src.device) * 6,
                                               torch.ones(size=(src.size(0), src.size(2), self.hidden_dim),
                                                          dtype=torch.float32, device=src.device) * 7,
                                               ), dim=2)
                    """
                    channel_embed = torch.cat((torch.zeros(size=(src.size(0), src.size(2), self.hidden_dim),
                                                           dtype=torch.float32, device=src.device),
                                               torch.ones(size=(src.size(0), src.size(2), self.hidden_dim),
                                                          dtype=torch.float32, device=src.device),
                                               torch.ones(size=(src.size(0), src.size(2), self.hidden_dim),
                                                          dtype=torch.float32, device=src.device) * 2,
                                               torch.ones(size=(src.size(0), src.size(2), self.hidden_dim),
                                                          dtype=torch.float32, device=src.device) * 3), dim=2)
                    """
                    scale = 6.283185307179586
                    eps = 1e-06
                    temperature = 10000
                    channel_embed = channel_embed / (channel_embed[:, :, -1:] + eps) * scale
                    dim_t = torch.arange(src.size(3), dtype=torch.float32, device=src.device)
                    dim_t = temperature ** (2 * (dim_t // 2) / src.size(3))
                    pos_ch = channel_embed[:, :, :, None] / dim_t
                    pos_ch = torch.stack((pos_ch[:, :, :, 0::2].sin(), pos_ch[:, :, :, 1::2].cos()), dim=4).flatten(3)
                    pos_ch = pos_ch.permute((0, 2, 1, 3))
                    src += pos_ch

                src = self.fuse_proj(src)
                hs = self.transformer(src, mask, self.query_embed.weight, pos[-1])[0]
            elif Config.variant == "N_B_N_proj_1_T" and Config.sub_variant == "decouple_curr_repeated_prev":
                for i, feat in enumerate(features):
                    features[i] = self.fuse_proj(feat)
                src2 = torch.cat(features, dim=1)  # current+prev. frames
                src = src2[:, 0:int(src2.size(1)/Config.num_of_repeated_blocks), :, :].repeat(1, Config.num_of_repeated_blocks, 1, 1)  # current frame
                hs = self.transformer(src, mask, self.query_embed.weight, pos[-1], src2=src2)[0]
            elif Config.variant == "N_B_N_proj_1_T":
                for i, feat in enumerate(features):
                    features[i] = self.fuse_proj(feat)
                src = torch.cat(features, dim=1)
                if Config.encode_channels:
                    # Channel Encoding
                    # TODO: Eslam should be generic according to "num_of_repeated_blocks"

                    channel_embed = torch.cat((torch.zeros(size=(src.size(0), src.size(2),
                                                                 int(src.size(1)/Config.num_of_repeated_blocks)),
                                                           dtype=torch.float32, device=src.device),
                                               torch.ones(size=(src.size(0), src.size(2),
                                                                int(src.size(1)/Config.num_of_repeated_blocks)),
                                                          dtype=torch.float32, device=src.device)), dim=2)
                    """
                    channel_embed = torch.cat((torch.zeros(size=(src.size(0), src.size(2), 64),
                                                           dtype=torch.float32, device=src.device),
                                               torch.ones(size=(src.size(0), src.size(2), 64),
                                                          dtype=torch.float32, device=src.device),
                                               torch.ones(size=(src.size(0), src.size(2), 64),
                                                          dtype=torch.float32, device=src.device)*2,
                                               torch.ones(size=(src.size(0), src.size(2), 64),
                                                          dtype=torch.float32, device=src.device)*3), dim=2)
                    """

                    scale = 6.283185307179586
                    eps = 1e-06
                    temperature = 10000
                    channel_embed = channel_embed / (channel_embed[:, :, -1:] + eps) * scale
                    dim_t = torch.arange(src.size(3), dtype=torch.float32, device=src.device)
                    dim_t = temperature ** (2 * (dim_t // 2) / src.size(3))
                    pos_ch = channel_embed[:, :, :, None] / dim_t
                    pos_ch = torch.stack((pos_ch[:, :, :, 0::2].sin(), pos_ch[:, :, :, 1::2].cos()), dim=4).flatten(3)
                    pos_ch = pos_ch.permute((0, 2, 1, 3))
                    src += pos_ch

                if Config.seg_only:
                    hs, trans_enc_feat = self.transformer(src, mask, self.query_embed.weight, pos[-1])
                    outputs_seg = self.seg_head(c1=features[0], feat_enc=trans_enc_feat,
                                                attns=hs[-1].unsqueeze(-1).permute(0, 1, 3, 2))
                    out = {'pred_masks': outputs_seg}
                    # TODO: Eslam: Add Aux loss for semantic segmentation
                    return out
                elif Config.MTL:
                    hs, trans_enc_feat = self.transformer(src, mask,
                                                          [self.query_embed[0], self.query_embed[1]],
                                                          pos[-1])
                    if Config.shared_dec_concat_q:
                        outputs_seg = self.seg_head(c1=features[0], feat_enc=trans_enc_feat,
                                                    attns=hs[:, :, -2:, :][-1].unsqueeze(-1).permute(0, 1, 3, 2))
                        hs = hs[:, :, :-2, :]  # detection output
                    elif Config.shared_dec_shared_q:
                        self.trans_out = hs  # detection & segmentation output
                    else:
                        if Config.seg_task_status:
                            outputs_seg = self.seg_head(c1=features[0], feat_enc=trans_enc_feat,
                                                        attns=hs[1][-1].unsqueeze(-1).permute(0, 1, 3, 2))
                        hs = hs[0]  # detection output
                        # TODO: Eslam: Add Aux loss for semantic segmentation
                else:
                    hs = self.transformer(src, mask, self.query_embed.weight, pos[-1])[0]
            elif Config.variant == "N_B_N_T_proj_1_T":
                trans_features = []
                for i in range(Config.num_of_repeated_blocks):
                    if Config.sharing:
                        trans_features.append(self.transformer(features[i], mask, self.query_embed.weight, pos[-1])[0])
                    else:
                        trans_features.append(self.transformer[i](features[i], mask,
                                                                  self.query_embed[i].weight, pos[-1])[0])
                if Config.concate:
                    hs = torch.cat(trans_features, dim=-1)
                    hs = self.fuse_proj(hs.permute(1, 3, 0, 2))
                    hs = hs.permute(2, 0, 3, 1)
                else:
                    hs = trans_features[0]
                    for i in range(1, Config.num_of_repeated_blocks):
                        hs += trans_features[i]
                if Config.encode_channels:
                    # TODO: Eslam make it generic w.r.t "num_of_repeated_blocks"
                    channel_embed = torch.cat((torch.zeros(size=(hs.size(1), int(hs.size(3) / 4)),
                                                           dtype=torch.float32, device=hs.device),
                                               torch.ones(size=(hs.size(1), int(hs.size(3) / 4)),
                                                          dtype=torch.float32, device=hs.device),
                                               torch.ones(size=(hs.size(1), int(hs.size(3) / 4)),
                                                          dtype=torch.float32, device=hs.device)*2,
                                               torch.ones(size=(hs.size(1), int(hs.size(3) / 4)),
                                                          dtype=torch.float32, device=hs.device)*3), dim=1)
                    scale = 6.283185307179586
                    eps = 1e-06
                    temperature = 10000
                    channel_embed = channel_embed / (channel_embed[:, -1:] + eps) * scale
                    dim_t = torch.arange(hs.size(2), dtype=torch.float32, device=hs.device)
                    dim_t = temperature ** (2 * (dim_t // 2) / hs.size(2))
                    pos_ch = channel_embed[:, :, None] / dim_t
                    pos_ch = torch.stack((pos_ch[:, :, 0::2].sin(), pos_ch[:, :, 1::2].cos()), dim=3).flatten(2)
                else:
                    pos_ch = None
                hs = hs[-1]
                hs = self.transformer2(hs, mask=torch.zeros((hs.size(0), hs.size(2)),
                                                            dtype=torch.bool, device=hs.device)
                                       , query_embed=self.query_embed2.weight,
                                       pos_embed=pos_ch, fusion_transformer=True)[0]

        elif Config.exp_type == "depth_pos_enc_arch2":
            hs = self.transformer(src1, mask, self.query_embed.weight, pos[-1],
                                  src2=src2, channel_embed=pos_channel[-1])[0]
        elif Config.exp_type != "depth_pos_enc_arch4" and Config.backbone_type == "DETR":
            if Config.seg_only:
                hs, trans_enc_feat = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])
                outputs_seg = self.seg_head(c1=features[0].decompose()[0], feat_enc=trans_enc_feat,
                                            attns=hs[-1].unsqueeze(-1).permute(0,1,3,2))
                out = {'pred_masks': outputs_seg}
                # TODO: Eslam: Add Aux loss for semantic segmentation
                return out
            elif Config.MTL:
                features = [self.input_proj(src)]
                hs, trans_enc_feat = self.transformer(features[0], mask,
                                                      [self.query_embed[0], self.query_embed[1]],
                                                      pos[-1])
                if Config.shared_dec_concat_q:
                    outputs_seg = self.seg_head(c1=features[0], feat_enc=trans_enc_feat,
                                                attns=hs[:, :, -2:, :][-1].unsqueeze(-1).permute(0, 1, 3, 2))
                    hs = hs[:, :, :-2, :]  # detection output
                elif Config.shared_dec_shared_q:
                    self.trans_out = hs  # detection & segmentation output
                else:
                    if Config.seg_task_status:
                        outputs_seg = self.seg_head(c1=features[0], feat_enc=trans_enc_feat,
                                                    attns=hs[1][-1].unsqueeze(-1).permute(0, 1, 3, 2))
                    hs = hs[0]  # detection output
                    # TODO: Eslam: Add Aux loss for semantic segmentation
            else:
                hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        if Config.backbone_type == "ViT":
            hs = self.transformer(features, mask=None, query_embed=self.query_embed.weight, pos_embed=pos)
            hs = hs[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if Config.shared_dec_shared_q:
            indices = self.matcher(out, targets)  # get the best quires that matches the target
            # get the selected indices and their assigned labels:
            list_of_dict = []
            for i, ind in enumerate(indices):  # Loop on batch_size
                selected_idx_for_each_class_dict = defaultdict(list)  # create dict of lists.
                # Key represents the class id and the list represents the selected indices for this class.
                for j, c in enumerate(targets[i]["labels"][ind[1]]):
                    selected_idx_for_each_class_dict[c.item()].append(ind[0][j])
                list_of_dict.append(selected_idx_for_each_class_dict)

            # print("hs = ", hs.shape)  # torch.Size([6, 16, 100, 256])
            # print("indices = ", indices[0].shape)  # torch.Size([2]) or torch.Size([3]) or any shape
            # Use the selected indices to generate foregrounds:
            foregrounds = torch.zeros_like(hs[:, :, 0, :]).unsqueeze(0).repeat((Config.num_classes-1, 1, 1, 1, 1))
            foregrounds = foregrounds.permute(0, 2, 3, 1, 4)  # torch.Size([80, 6, 8, 1, 256])
            for i in range(hs.shape[1]):  # loop on the batch_size
                for k, v in list_of_dict[i].items():
                    selected_indices = torch.tensor(v, device=hs.device)
                    k -= 1
                    foregrounds[k, :, i, :, :] = torch.sum(torch.index_select(hs[:, i, :, :].unsqueeze(1),
                                                                              2, selected_indices), dim=2)

            # sum all foreground classes
            foreground = torch.sum(foregrounds, dim=0, keepdim=False)  # torch.Size([6, b, 1, 256])
            background = torch.sum(hs, dim=2, keepdim=True) - foreground
            foregrounds = foregrounds.squeeze(3).permute(1, 2, 0, 3)  # torch.Size([6, b, num_classes, 256])
            hs = torch.cat([foregrounds, background], dim=2)  # torch.Size([6, b, num_classes, 256])
            outputs_seg = self.seg_head(c1=features[0], feat_enc=trans_enc_feat,
                                        attns=hs[-1].unsqueeze(-1).permute(0, 1, 3, 2))

        if Config.seg_task_status:
            out['pred_masks'] = outputs_seg
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        if Config.aux_q:
            return out, out_aux_intermediate_q
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        """
        Usage example:
        indices =  [(array([59, 71, 92]), array([0, 1, 2])),
         (array([ 4, 10, 16, 27, 35, 48, 49, 51, 52, 53, 55, 68, 71, 78, 81, 83, 90, 94, 98]),
          array([10, 15, 12,  1,  4,  9,  2, 14, 17, 11,  0,  5, 18,  3,  6, 16,  8, 13,  7]))]
        batch_idx =  tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        src_idx =  tensor([59, 71, 92,  4, 10, 16, 27, 35, 48, 49, 51, 52, 53, 55, 68, 71, 78, 81, 83, 90, 94, 98])
        """
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # Eslam
        # print("outputs_without_aux = ", outputs_without_aux['pred_boxes'].shape)
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    elif args.dataset_file == "cityscapes":
        num_classes = 2
    elif args.dataset_file == "kitti_tracking":
        num_classes = 2
    elif args.dataset_file == "kitti_old":
        num_classes = 2
    elif Config.convert_coco_to1class and args.dataset_file == "coco":
        num_classes = 2
    # TODO: Eslam--> Should check it
    # "You should always use num_classes = max_id + 1 where max_id is the highest class ID that you have in your dataset."
    # Reference: https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    elif args.dataset_file == "coco":
        num_classes = 81
    Config.num_classes = num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)
    if Config.exp_type == "shared_rgb_of" or (Config.exp_type == "depth_pos_enc_arch2" and not Config.sharing)\
            or Config.exp_type == "shared_rgb_of_N" or Config.exp_type == "depth_pos_enc_arch4":
        backbone2 = build_backbone(args)
    else:
        backbone2 = None

    transformer = build_transformer(args)
    if Config.exp_type == "depth_pos_enc_arch4":
        transformer2 = build_transformer(args)
        transformer3 = build_transformer(args, fusion_transformer=True)

    elif Config.exp_type == "shared_rgb_of_N" and Config.variant == "N_B_N_T_proj_1_T":
        transformer = []
        if Config.sharing:
            transformer = build_transformer(args)
        else:
            for i in range(Config.num_of_repeated_blocks):
                transformer.append(build_transformer(args))
            transformer = nn.ModuleList(transformer)
        transformer2 = build_transformer(args, fusion_transformer=True)
        transformer3 = None
    else:
        transformer2 = None
        transformer3 = None

    if Config.seg_only:
        if (args.dataset_file == "coco" and Config.convert_coco_to1class) or args.dataset_file == "kitti_old":
            num_queries = 2
        elif args.dataset_file == "coco":
            num_queries = 81
    elif Config.MTL:
        # Determine transformer decoder's number of queries for each task.
        num_queries = []
        # Detection queries are fixed 100.
        if Config.det_task_status:
            num_queries.append(100)
        else:
            num_queries.append(0)
        if Config.seg_task_status:
            if (args.dataset_file == "coco" and Config.convert_coco_to1class) or args.dataset_file == "kitti_old":
                num_queries.append(2)
            elif args.dataset_file == "coco":
                num_queries.append(81)
        else:
            num_queries.append(0)
    else:
        num_queries = args.num_queries

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=num_queries,
        aux_loss=args.aux_loss,
        backbone2=backbone2,
        transformer2=transformer2,
        transformer3=transformer3,
        args=args,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]

    criterion_mtl = []
    if Config.det_task_status:
        criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=args.eos_coef, losses=losses)
        criterion.to(device)
        criterion_mtl.append(criterion)
    if Config.seg_task_status:
        from models.semantic_seg.seg_loss import MixSoftmaxCrossEntropyLoss
        criterion = MixSoftmaxCrossEntropyLoss(aux=False)
        criterion.to(device)
        criterion_mtl.append(criterion)

    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    if Config.MTL:
        criterion = criterion_mtl
    return model, criterion, postprocessors
