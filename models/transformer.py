# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from config import Config


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        # Define transformer encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # Define transformer decoder
        if Config.MTL and not Config.shared_dec_concat_q and not Config.shared_dec_shared_q:
            self.d_model = d_model
            if Config.det_task_status:
                decoder_layer_det = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                            dropout, activation, normalize_before)
                if Config.exp_type == "shared_rgb_of_N" and Config.variant == "N_B_1_T":
                    decoder_norm_det = nn.LayerNorm(d_model * Config.num_of_repeated_blocks)
                else:
                    decoder_norm_det = nn.LayerNorm(d_model)
                self.decoder_det = TransformerDecoder(decoder_layer_det, num_decoder_layers, decoder_norm_det,
                                                      return_intermediate=return_intermediate_dec)
            if Config.seg_task_status:
                decoder_layer_seg = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                            dropout, activation, normalize_before, semantic_seg=True)
                if Config.exp_type == "shared_rgb_of_N" and Config.variant == "N_B_1_T":
                    decoder_norm_seg = nn.LayerNorm(d_model * Config.num_of_repeated_blocks)
                else:
                    decoder_norm_seg = nn.LayerNorm(d_model)
                self.decoder_seg = TransformerDecoder(decoder_layer_seg, num_decoder_layers, decoder_norm_seg,
                                                      return_intermediate=return_intermediate_dec)
        else:
            if Config.seg_only:
                semantic_seg = True
            else:
                semantic_seg = False
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before, semantic_seg=semantic_seg)
            if Config.exp_type == "shared_rgb_of_N" and Config.variant == "N_B_1_T":
                decoder_norm = nn.LayerNorm(d_model*Config.num_of_repeated_blocks)
            else:
                decoder_norm = nn.LayerNorm(d_model)
            if Config.exp_type == "depth_pos_enc_arch2":
                if not Config.sharing:
                    self.encoder2 = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
                if Config.variant == "NB_NTE_1T":
                    if Config.TPE_type == "NC_HW":
                        # TODO :Eslam make it generic.
                        self.hw_TO_dmodel_proj = nn.Conv1d(300, d_model, kernel_size=1)
                        """
                        encoder_layer = TransformerEncoderLayer(300, 10, dim_feedforward,
                                                                dropout, activation, normalize_before)
                        decoder_layer = TransformerDecoderLayer(300, 10, dim_feedforward,
                                                                dropout, activation, normalize_before)
                        decoder_norm = nn.LayerNorm(300)
                        """
                    elif Config.TPE_type == "N_CHW":
                        # TODO :Eslam make it generic.
                        encoder_layer = TransformerEncoderLayer(d_model*300, 10, dim_feedforward,
                                                                dropout, activation, normalize_before)
                        decoder_layer = TransformerDecoderLayer(d_model*300, 10, dim_feedforward,
                                                                dropout, activation, normalize_before)
                        decoder_norm = nn.LayerNorm(d_model*300)
                    self.encoder3 = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

            self.d_model = d_model
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                              return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.nhead = nhead

        if Config.exp_type == "depth_pos_enc_arch2":
            self.input_proj_transformer = nn.Conv1d(256, 128, kernel_size=1)
            if not Config.sharing:
                self.input_proj_transformer2 = nn.Conv1d(256, 128, kernel_size=1)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, src2=None, channel_embed=None, fusion_transformer=False):
        # flatten NxCxHxW to HWxNxC
        # print("src.shape = ", src.shape)
        # torch.Size([2, 256, 28, 38])
        if fusion_transformer:
            bs, q_shape, c = src.shape
            src = src.permute(2, 0, 1)
            if pos_embed is not None:
                pos_embed = pos_embed.permute(1, 0, 2)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            tgt = torch.zeros_like(query_embed)
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
            hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                              pos=pos_embed, query_pos=query_embed)
            return hs.transpose(1, 2), 0
        else:
            if Config.backbone_type == "DETR":
                bs, c, h, w = src.shape[0:4]  # (32, 256, 10, 30)
                src = src.flatten(2).permute(2, 0, 1)  # (300, 32, 256)
                if Config.sub_variant == "decouple_curr_repeated_prev" or\
                        Config.sub_variant == "decouple_curr_prev":
                    src2 = src2.flatten(2).permute(2, 0, 1)
                if pos_embed is not None:
                    pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # (300,32,256)
                    if Config.TPE_type == "NHW_C" and False:
                        pos_embed = torch.cat((pos_embed[0::2, :, :], pos_embed[1::2, :, :]), dim=0)
                    """
                    eslam_h = pos_embed.cpu().numpy()[:, 0, :]
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    # plt.imshow(eslam_h, cmap='hot', interpolation='nearest')
                    fig, ax = plt.subplots(figsize=(20, 12))
                    # ax = sns.heatmap(eslam_h, linewidth=0.5)
                    ax = sns.heatmap(eslam_h)
                    fig = ax.get_figure()
                    fig.savefig("total_3.png")
                    """
                mask = mask.flatten(1)  # (32,300)
            elif Config.backbone_type == "ViT":
                l, bs, c = src.shape
                mask = torch.zeros((bs, l), device=src.device, dtype=torch.bool)
            if Config.MTL:
                if Config.det_task_status:
                    query_embed_det = query_embed[0].weight.unsqueeze(1).repeat(1, bs, 1)  # (100,32,256)
                    tgt_det = torch.zeros_like(query_embed_det)
                    if Config.shared_dec_shared_q:
                        tgt = tgt_det
                        query_embed = query_embed_det
                if Config.seg_task_status and not Config.shared_dec_shared_q:
                    query_embed_seg = query_embed[1].weight.unsqueeze(1).repeat(1, bs, 1)  # (#of_classes,32,256)
                    tgt_seg = torch.zeros_like(query_embed_seg)
                if Config.shared_dec_concat_q:
                    tgt = torch.cat([tgt_det, tgt_seg], dim=0)
                    query_embed = torch.cat([query_embed_det, query_embed_seg], dim=0)
            else:
                query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)   # (100,32,256)
                tgt = torch.zeros_like(query_embed)

        if Config.sub_variant == "decouple_curr_repeated_prev" or \
                Config.sub_variant == "decouple_curr_prev":
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed, src2=src2)
        else:
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        if Config.exp_type == "depth_pos_enc_arch2":
            channel_embed = channel_embed.flatten(2).permute(2, 0, 1)
            src2 = src2.flatten(2).permute(2, 0, 1)
            memory = self.input_proj_transformer(memory.permute(1, 2, 0)).permute(2, 0, 1)
            if Config.sharing:
                memory2 = self.encoder(src2, src_key_padding_mask=mask, pos=pos_embed)
                memory2 = self.input_proj_transformer(memory2.permute(1, 2, 0)).permute(2, 0, 1)
            else:
                memory2 = self.encoder2(src2, src_key_padding_mask=mask, pos=pos_embed)
                memory2 = self.input_proj_transformer2(memory2.permute(1, 2, 0)).permute(2, 0, 1)

            if Config.variant == "NB_NTE_1T":
                if Config.TPE_type == "HW_NC":
                    memory = torch.cat((memory, memory2), dim=2)  # HW * b * NC
                    #memory = memory + channel_embed
                    pos_embed = pos_embed + channel_embed
                elif Config.TPE_type == "NC_HW":
                    # TODO: Eslam check this again.
                    memory = torch.cat((memory, memory2), dim=2)  # HW * b * NC
                    memory = memory.permute(2, 1, 0)  # NC * b * HW
                    memory = self.hw_TO_dmodel_proj(memory.permute(1, 2, 0)).permute(2, 0, 1)
                    mask = torch.zeros((memory.size(1), memory.size(0)), dtype=torch.bool, device=memory.device)  # b*HW
                    # Encoding Channels
                    channel_embed = torch.cat((torch.zeros(size=(memory.size(1), int(memory.size(0) / 2)),
                                                           dtype=torch.float32, device=memory.device),
                                               torch.ones(size=(memory.size(1), int(memory.size(0) / 2)),
                                                          dtype=torch.float32, device=memory.device)), dim=1)
                    scale = 6.283185307179586
                    eps = 1e-06
                    temperature = 10000
                    channel_embed = channel_embed / (channel_embed[:, -1:] + eps) * scale
                    dim_t = torch.arange(memory.size(2), dtype=torch.float32, device=memory.device)
                    dim_t = temperature ** (2 * (dim_t // 2) / memory.size(2))
                    pos_ch = channel_embed[:, :, None] / dim_t
                    pos_ch = torch.stack((pos_ch[:, :, 0::2].sin(), pos_ch[:, :, 1::2].cos()), dim=3).flatten(2)
                    pos_embed = pos_ch.permute((1, 0, 2))

                elif Config.TPE_type == "N_CHW":
                    # TODO: Eslam check this again.
                    memory = torch.cat((memory, memory2), dim=0)  # NHW * b * C
                    memory = memory.permute(2, 1, 0)  # NC * b * HW
                    mask = torch.zeros((memory.size(1), memory.size(0)), dtype=torch.bool, device=memory.device)  # b*HW
                    # Encoding Channels
                    channel_embed = torch.cat((torch.zeros(size=(memory.size(1), int(memory.size(0) / 2)),
                                                           dtype=torch.float32, device=memory.device),
                                               torch.ones(size=(memory.size(1), int(memory.size(0) / 2)),
                                                          dtype=torch.float32, device=memory.device)), dim=1)
                    scale = 6.283185307179586
                    eps = 1e-06
                    temperature = 10000
                    channel_embed = channel_embed / (channel_embed[:, -1:] + eps) * scale
                    dim_t = torch.arange(memory.size(2), dtype=torch.float32, device=memory.device)
                    dim_t = temperature ** (2 * (dim_t // 2) / memory.size(2))
                    pos_ch = channel_embed[:, :, None] / dim_t
                    pos_embed = torch.stack((pos_ch[:, :, 0::2].sin(), pos_ch[:, :, 1::2].cos()), dim=3).flatten(2)
                    pos_embed = pos_embed.permute((1,0,2))

                memory = self.encoder3(memory, src_key_padding_mask=mask, pos=pos_embed)
            else:
                memory = torch.cat((memory, memory2), dim=2)  # HW * b * NC
                # TODO: Eslam check this, make it configurable or remove addition operation.
                memory = memory + channel_embed

        """
        print(tgt.shape)  # [100, 2, 256]
        print(memory.shape)  # [300, 2, 256]
        print(mask.max())  # [2, 300]
        print(mask.min())  # [2, 300]
        print(pos_embed.shape)  # [300, 2, 256]
        print(query_embed.shape)  # [100, 2, 256]
        """
        if Config.MTL and not Config.shared_dec_concat_q and not Config.shared_dec_shared_q:
            hs_mtl = []
            if Config.det_task_status:
                hs_det = self.decoder_det(tgt_det, memory, memory_key_padding_mask=mask,
                                          pos=pos_embed, query_pos=query_embed_det)
                hs_mtl.append(hs_det.transpose(1, 2))
            else:
                hs_mtl.append(0)
            if Config.seg_task_status:
                hs_seg = self.decoder_seg(tgt_seg, memory, memory_key_padding_mask=mask,
                                          pos=pos_embed, query_pos=query_embed_seg)
                hs_mtl.append(hs_seg.transpose(1, 2))
            else:
                hs_mtl.append(0)

            return hs_mtl, memory.permute(1, 2, 0).view(bs, c, h, w)
        else:
            hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                              pos=pos_embed, query_pos=query_embed)
            if Config.shared_dec_concat_q and False:
                return [hs[0].transpose(1, 2), hs[1].transpose(1, 2)], memory.permute(1, 2, 0).view(bs, c, h, w)
        if Config.exp_type == "depth_pos_enc_arch2" \
                or (Config.exp_type == "depth_pos_enc" and Config.TPE_type == "NHW_C"):
            return hs.transpose(1, 2), 0
        if Config.backbone_type == "DETR":
            return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
        elif Config.backbone_type == "ViT":
            return hs.transpose(1, 2), memory


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, fuse=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.fuse = fuse

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, src2=None):
        output = src

        for i, layer in enumerate(self.layers):
            if i > 0 and self.fuse:
                pos = None
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, src2=src2)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        if Config.MTL and Config.shared_dec_concat_q and False:
            self.norm_w = norm
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        if Config.MTL and Config.shared_dec_concat_q and False:
            intermediate_w = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                if Config.MTL and Config.shared_dec_concat_q and False:
                    intermediate.append(self.norm(output[0]))
                    intermediate_w.append(self.norm_w(output[1]))
                else:
                    intermediate.append(self.norm(output))

        if Config.MTL and Config.shared_dec_concat_q and False:
            if self.norm is not None:
                output[0] = self.norm(output[0])
                output[1] = self.norm_w(output[1])
                if self.return_intermediate:
                    intermediate.pop()
                    intermediate.append(output[0])
                    intermediate_w.pop()
                    intermediate_w.append(output[1])

            if self.return_intermediate:
                return [torch.stack(intermediate), torch.stack(intermediate_w)]
        else:
            if self.norm is not None:
                output = self.norm(output)
                if self.return_intermediate:
                    intermediate.pop()
                    intermediate.append(output)

            if self.return_intermediate:
                return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, fuse=False):
        super().__init__()
        if Config.exp_type == "shared_rgb_of_N" and Config.variant == "N_B_1_T":
            d_model *= Config.num_of_repeated_blocks
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.fuse = fuse

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, src2=None):
        if Config.sub_variant == "decouple_curr_repeated_prev" or \
                Config.sub_variant == "decouple_curr_prev":
            q = self.with_pos_embed(src, pos)
            k = self.with_pos_embed(src2, pos)
            src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        elif self.fuse:
            k = self.with_pos_embed(src, pos)
            q = self.with_pos_embed(src, pos)[:100]
            v = src
            src2 = self.self_attn(q, k, value=v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
            src = src[:100]
        else:
            q = k = self.with_pos_embed(src, pos)
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None, src2=None):
        src = self.norm1(src)
        if Config.sub_variant == "decouple_curr_repeated_prev" or \
                Config.sub_variant == "decouple_curr_prev":
            q = self.with_pos_embed(src, pos)
            k = self.with_pos_embed(src2, pos)
            src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        else:
            q = k = self.with_pos_embed(src, pos)

            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, src2=None):
        if self.normalize_before:
            if Config.sub_variant == "decouple_curr_repeated_prev" or \
                    Config.sub_variant == "decouple_curr_prev":
                return self.forward_pre(src, src_mask, src_key_padding_mask, pos, src2=src2)
            else:
                return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        if Config.sub_variant == "decouple_curr_repeated_prev" or \
                Config.sub_variant == "decouple_curr_prev":
            return self.forward_post(src, src_mask, src_key_padding_mask, pos, src2=src2)
        else:
            return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, semantic_seg=False):
        super().__init__()
        if Config.exp_type == "shared_rgb_of_N" and Config.variant == "N_B_1_T":
            d_model *= Config.num_of_repeated_blocks
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        if Config.MTL and Config.shared_dec_concat_q and False:
            self.linear1_w = nn.Linear(d_model, dim_feedforward)
            self.linear2_w = nn.Linear(dim_feedforward, d_model)
            self.dropout_w = nn.Dropout(dropout)
            self.dropout2_w = nn.Dropout(dropout)
            self.dropout3_w = nn.Dropout(dropout)
            self.norm2_w = nn.LayerNorm(d_model)
            self.norm3_w = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.semantic_seg = semantic_seg

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if self.semantic_seg:
            index_of_interest = 1
        else:
            index_of_interest = 0
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[index_of_interest]
        if self.semantic_seg:
            tgt2 = tgt2.permute(1, 0, 2)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    # TODO:Eslam should clean this function, I think I have to remove it and use forward_post.
    def forward_shared_dec(self, tgt, memory,
                           tgt_mask: Optional[Tensor] = None,
                           memory_mask: Optional[Tensor] = None,
                           tgt_key_padding_mask: Optional[Tensor] = None,
                           memory_key_padding_mask: Optional[Tensor] = None,
                           pos: Optional[Tensor] = None,
                           query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt_org = self.norm1(tgt)
        tgt2, tgt2_w = self.multihead_attn(query=self.with_pos_embed(tgt_org, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)

        tgt2_w = tgt2_w.permute(1, 0, 2)
        # 1- Attention output
        tgt = tgt_org + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

        # TODO:Eslam: I will try to share this part too for both decoders
        """
        # 2- Attention output weights
        tgt_w = tgt_org + self.dropout2_w(tgt2_w)
        tgt_w = self.norm2_w(tgt_w)
        tgt2_w = self.linear2_w(self.dropout_w(self.activation(self.linear1_w(tgt_w))))
        tgt_w = tgt_w + self.dropout3_w(tgt2_w)
        tgt_w = self.norm3_w(tgt_w)
        return [tgt, tgt_w]
        """

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if Config.MTL and Config.shared_dec_concat_q:
            return self.forward_shared_dec(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask,
                                           memory_key_padding_mask, pos, query_pos)
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args, fusion_transformer=False):
    if fusion_transformer:
        #d_model = args.hidden_dim * 2
        d_model = 100
        nheads = 4
    else:
        d_model = args.hidden_dim
        nheads = args.nheads
    return Transformer(
        d_model=d_model,
        dropout=args.dropout,
        nhead=nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
