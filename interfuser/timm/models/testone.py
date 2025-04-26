import math
import copy
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
import numpy
import numpy as np
import logging
from typing import Optional, List
from collections import OrderedDict
from .registry import register_model
from .resnet import resnet26d, resnet50d, resnet18d, resnet26, resnet50, resnet101d
from .layers import StdConv2dSame, StdConv2d, to_2tuple

_logger = logging.getLogger(__name__)



class HybridEmbed(nn.Module):
    def __init__(self, backbone: nn.Module, img_size=224, patch_size=1,
                 feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                backbone.eval()
                o = backbone(torch.zeros(1, in_chans, *img_size))
                if isinstance(o, (list, tuple)):
                    o = o[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = (backbone.feature_info.channels()[-1]
                           if hasattr(backbone, "feature_info")
                           else backbone.num_features)
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]
        x = self.proj(x)
        global_x = x.mean(dim=[2, 3], keepdim=False)[:, :, None]
        return x, global_x


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000,
                 normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = scale or 2 * math.pi

    def forward(self, tensor: Tensor):
        b, _, h, w = tensor.shape
        mask = torch.ones((b, h, w), device=tensor.device)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = (self.temperature **
                 (2 * (torch.arange(self.num_pos_feats,
                                     device=tensor.device) // 2)
                  / self.num_pos_feats))
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(),
                             pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(),
                             pos_y[..., 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation=nn.ReLU, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead,
                                               dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_mask=None, src_key_padding_mask=None,
                pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation=nn.ReLU, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead,
                                               dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead,
                                                    dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        # self-attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # cross-attention
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # feed-forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask,
                           pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        output = tgt
        intermediates = []
        for layer in self.layers:
            output = layer(output, memory,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediates.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
        if self.return_intermediate:
            intermediates.append(output)
            return torch.stack(intermediates)
        return output.unsqueeze(0)


class GRUWaypointsPredictor(nn.Module):
    def __init__(self, input_dim, waypoints=10):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=64,
                          batch_first=True)
        self.encoder = nn.Linear(1, 64)  # velocity only
        self.decoder = nn.Linear(64, 2)
        self.waypoints = waypoints

    def forward(self, x, measurements):
        bs = x.size(0)
        vel = measurements[:, 6:7]  # velocity scalar per batch
        z = self.encoder(vel).unsqueeze(0)
        out, _ = self.gru(x, z)
        out = out.reshape(bs * self.waypoints, -1)
        out = self.decoder(out).reshape(bs, self.waypoints, 2)
        return torch.cumsum(out, dim=1)


@register_model
class InterfuserTwoCam(nn.Module):
    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3,
                 embed_dim=256, enc_depth=6, dec_depth=6,
                 num_heads=8, dropout=0.1,
                 waypoints_pred_head="gru"):  
        super().__init__()
        # backbone and embedding
        backbone = resnet50d(pretrained=True,
                             in_chans=in_chans,
                             features_only=True,
                             out_indices=[4])
        self.patch_embed = partial(HybridEmbed,
                                   backbone=backbone,
                                   img_size=img_size,
                                   patch_size=patch_size,
                                   in_chans=in_chans,
                                   embed_dim=embed_dim)
        # embeddings per view (left=0, right=1)
        self.global_embed = nn.Parameter(torch.zeros(1, embed_dim, 2))
        self.view_embed   = nn.Parameter(torch.zeros(1, embed_dim, 2, 1))
        self.pos_enc = PositionEmbeddingSine(embed_dim//2,
                                             normalize=True)
        # transformer
        enc_layer = TransformerEncoderLayer(embed_dim, num_heads,
                                            dropout=dropout)
        dec_layer = TransformerDecoderLayer(embed_dim, num_heads,
                                            dropout=dropout)
        self.encoder = TransformerEncoder(enc_layer, enc_depth)
        self.decoder = TransformerDecoder(dec_layer, dec_depth,
                                          norm=nn.LayerNorm(embed_dim))
        # queries: 400 spatial + 13 non-spatial
        self.query_pos = nn.Parameter(torch.zeros(1, embed_dim, 13))
        self.query_tok = nn.Parameter(torch.zeros(400+13, 1, embed_dim))
        # heads
        self.wayp_head = GRUWaypointsPredictor(embed_dim, waypoints=10)
        self.junc_head = nn.Linear(embed_dim, 2)
        self.light_head = nn.Linear(embed_dim, 2)
        self.stop_head = nn.Linear(embed_dim, 2)
        self.traffic_head = nn.Sequential(
            nn.Linear(embed_dim+1, 64), nn.ReLU(),
            nn.Linear(64, 7), nn.Sigmoid()
        )

    def forward_features(self, left: Tensor, right: Tensor):
        feats = []
        for idx, img in enumerate((left, right)):
            x, xg = self.patch_embed(img)
            x = x + self.view_embed[:, :, idx:idx+1, :] + self.pos_enc(x)
            seq = x.flatten(2).permute(2, 0, 1)
            xg = xg + self.view_embed[:, :, idx, :] + self.global_embed[:, :, idx:idx+1]
            seqg = xg.permute(2, 0, 1)
            feats += [seq, seqg]
        return torch.cat(feats, dim=0)

    def forward(self, x: dict):
        left  = x['rgb_left']      # (bs, C, H, W)
        right = x['rgb_right']
        meas  = x['measurements']   # (bs, >=7)
        bs = left.size(0)
        # encoder
        mem = self.encoder(self.forward_features(left, right))
        # prepare decoder tgt
        spatial = self.pos_enc(torch.ones((bs,1,20,20), device=left.device))
        spatial = spatial.flatten(2)
        tgt = torch.cat([spatial, self.query_pos.repeat(bs,1,1)], dim=2)
        tgt = tgt.permute(2,0,1)
        # decode
        hs = self.decoder(self.query_tok.repeat(1,bs,1), mem,
                          query_pos=tgt)[0]
        hs = hs.permute(1,0,2)
        # split
        traffic_feat = hs[:, :400]
        junc_feat    = hs[:, 400]
        light_feat   = hs[:, 401]
        stop_feat    = hs[:, 402]
        wayp_feat    = hs[:, 403:413]
        # heads
        waypoints = self.wayp_head(wayp_feat, meas)
        is_junc   = self.junc_head(junc_feat)
        light     = self.light_head(light_feat)
        stop      = self.stop_head(stop_feat)
        vel = meas[:,6:7]  # (bs,1)
        vel_rep = vel.unsqueeze(-1).repeat(1,400,1)
        traffic = self.traffic_head(torch.cat([traffic_feat, vel_rep], dim=2))
        return traffic, waypoints, is_junc, light, stop, traffic_feat

@register_model
def interfuser_baseline(**kwargs):
    model = Interfuser(
        enc_depth=6,
        dec_depth=6,
        embed_dim=256,
        rgb_backbone_name="r50",
        lidar_backbone_name="r18",
        waypoints_pred_head="gru",
        use_different_backbone=True,
    )
    return model
