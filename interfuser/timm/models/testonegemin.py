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
    def __init__(
        self,
        backbone,
        img_size=224,
        patch_size=1,
        feature_size=None,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, "feature_info"):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features

        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x)
        global_x = torch.mean(x, [2, 3], keepdim=False)[:, :, None]
        return x, global_x


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        x = tensor
        bs, _, h, w = x.shape
        not_mask = torch.ones((bs, h, w), device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format="NCHW"):
        super().__init__()

        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.0

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self.height), np.linspace(-1.0, 1.0, self.width)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...

        if self.data_format == "NHWC":
            feature = (
                feature.transpose(1, 3)
                .tranpose(2, 3)
                .view(-1, self.height * self.width)
            )
        else:
            feature = feature.view(-1, self.height * self.width)

        weight = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(
            torch.autograd.Variable(self.pos_x) * weight, dim=1, keepdim=True
        )
        expected_y = torch.sum(
            torch.autograd.Variable(self.pos_y) * weight, dim=1, keepdim=True
        )
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel, 2)
        feature_keypoints[:, :, 1] = (feature_keypoints[:, :, 1] - 1) * 12
        feature_keypoints[:, :, 0] = feature_keypoints[:, :, 0] * 12
        return feature_keypoints


class MultiPath_Generator(nn.Module):
    def __init__(self, in_channel, embed_dim, out_channel):
        super().__init__()
        self.spatial_softmax = SpatialSoftmax(100, 100, out_channel)
        self.tconv0 = nn.Sequential(
            nn.ConvTranspose2d(in_channel, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 192, 4, 2, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.tconv4_list = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(64, out_channel, 8, 2, 3, bias=False),
                    nn.Tanh(),
                )
                for _ in range(6)
            ]
        )

        self.upsample = nn.Upsample(size=(50, 50), mode="bilinear")

    def forward(self, x, measurements):
        mask = measurements[:, :6]
        mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 100, 100)
        velocity = measurements[:, 6:7].unsqueeze(-1).unsqueeze(-1)
        velocity = velocity.repeat(1, 32, 2, 2)

        n, d, c = x.shape
        x = x.transpose(1, 2)
        x = x.view(n, -1, 2, 2)
        x = torch.cat([x, velocity], dim=1)
        x = self.tconv0(x)
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.upsample(x)
        xs = []
        for i in range(6):
            xt = self.tconv4_list[i](x)
            xs.append(xt)
        xs = torch.stack(xs, dim=1)
        x = torch.sum(xs * mask, dim=1)
        x = self.spatial_softmax(x)
        return x


class LinearWaypointsPredictor(nn.Module):
    def __init__(self, input_dim, cumsum=True):
        super().__init__()
        self.cumsum = cumsum
        self.rank_embed = nn.Parameter(torch.zeros(1, 10, input_dim))
        self.head_fc1_list = nn.ModuleList([nn.Linear(input_dim, 64) for _ in range(6)])
        self.head_relu = nn.ReLU(inplace=True)
        self.head_fc2_list = nn.ModuleList([nn.Linear(64, 2) for _ in range(6)])

    def forward(self, x, measurements):
        # input shape: n 10 embed_dim
        bs, n, dim = x.shape
        x = x + self.rank_embed
        x = x.reshape(-1, dim)

        mask = measurements[:, :6]
        mask = torch.unsqueeze(mask, -1).repeat(n, 1, 2)

        rs = []
        for i in range(6):
            res = self.head_fc1_list[i](x)
            res = self.head_relu(res)
            res = self.head_fc2_list[i](res)
            rs.append(res)
        rs = torch.stack(rs, 1)
        x = torch.sum(rs * mask, dim=1)

        x = x.view(bs, n, 2)
        if self.cumsum:
            x = torch.cumsum(x, 1)
        return x


class GRUWaypointsPredictor(nn.Module):
    def __init__(self, input_dim, waypoints=10):
        super().__init__()
        # self.gru = torch.nn.GRUCell(input_size=input_dim, hidden_size=64)
        self.gru = torch.nn.GRU(input_size=input_dim, hidden_size=64, batch_first=True)
        self.encoder = nn.Linear(2, 64)
        self.decoder = nn.Linear(64, 2)
        self.waypoints = waypoints

    def forward(self, x, target_point):
        bs = x.shape[0]
        z = self.encoder(target_point).unsqueeze(0)
        output, _ = self.gru(x, z)
        output = output.reshape(bs * self.waypoints, -1)
        output = self.decoder(output).reshape(bs, self.waypoints, 2)
        output = torch.cumsum(output, 1)
        return output

class GRUWaypointsPredictorWithCommand(nn.Module):
    def __init__(self, input_dim, waypoints=10):
        super().__init__()
        # self.gru = torch.nn.GRUCell(input_size=input_dim, hidden_size=64)
        self.grus = nn.ModuleList([torch.nn.GRU(input_size=input_dim, hidden_size=64, batch_first=True) for _ in range(6)])
        self.encoder = nn.Linear(2, 64)
        self.decoders = nn.ModuleList([nn.Linear(64, 2) for _ in range(6)])
        self.waypoints = waypoints

    def forward(self, x, target_point, measurements):
        bs, n, dim = x.shape
        mask = measurements[:, :6, None, None]
        mask = mask.repeat(1, 1, self.waypoints, 2)

        z = self.encoder(target_point).unsqueeze(0)
        outputs = []
        for i in range(6):
            output, _ = self.grus[i](x, z)
            output = output.reshape(bs * self.waypoints, -1)
            output = self.decoders[i](output).reshape(bs, self.waypoints, 2)
            output = torch.cumsum(output, 1)
            outputs.append(output)
        outputs = torch.stack(outputs, 1)
        output = torch.sum(outputs * mask, dim=1)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.ReLU(),
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.ReLU(),
        normalize_before=False,
    ):
        super().__init__()
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

        self.activation = activation()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def build_attn_mask(mask_type):
    # Original mask was built for 5 views (front, left, right, center, lidar) and global tokens.
    # With only left and right cameras, the structure changes significantly.
    # A new mask logic would be needed if separate view attention is still desired.
    # For simplicity and to avoid potential errors with misaligned masks,
    # returning None will result in full self-attention across all tokens.
    _logger.warning(f"Attention mask building for mask_type='{mask_type}' might not be suitable for only two camera inputs. Using no mask.")
    return None


class Interfuser(nn.Module):
    def __init__(
        self,
        img_size=224,
        multi_view_img_size=112,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        enc_depth=6,
        dec_depth=6,
        dim_feedforward=2048,
        normalize_before=False,
        rgb_backbone_name="r26",
        lidar_backbone_name="r26",
        num_heads=8,
        norm_layer=None,
        dropout=0.1,
        end2end=False,
        direct_concat=False,
        separate_view_attention=False,
        separate_all_attention=False,
        act_layer=None,
        weight_init="",
        freeze_num=-1,
        with_lidar=False, # Modified to False
        with_right_left_sensors=True, # Keep as True
        with_center_sensor=False, # Modified to False
        traffic_pred_head_type="det",
        waypoints_pred_head="heatmap",
        reverse_pos=True,
        use_different_backbone=False,
        use_view_embed=True,
        use_mmad_pretrain=None,
    ):
        super().__init__()
        self.traffic_pred_head_type = traffic_pred_head_type
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.reverse_pos = reverse_pos
        self.waypoints_pred_head = waypoints_pred_head
        self.with_lidar = with_lidar
        self.with_right_left_sensors = with_right_left_sensors
        self.with_center_sensor = with_center_sensor

        self.direct_concat = direct_concat
        self.separate_view_attention = separate_view_attention
        self.separate_all_attention = separate_all_attention
        self.end2end = end2end
        self.use_view_embed = use_view_embed

        if self.direct_concat:
             # This case is not intended with the current modification (only left/right views)
            in_chans = in_chans * 4
            self.with_center_sensor = False
            self.with_right_left_sensors = False # This would make the model use no cameras if direct_concat is True

        if self.separate_view_attention or self.separate_all_attention:
             # Call build_attn_mask but it will return None for safety with two views
            self.attn_mask = build_attn_mask("seperate_view" if self.separate_view_attention else "seperate_all")
        else:
            self.attn_mask = None


        if use_different_backbone:
            if rgb_backbone_name == "r50":
                self.rgb_backbone = resnet50d(
                    pretrained=True,
                    in_chans=in_chans,
                    features_only=True,
                    out_indices=[4],
                )
            elif rgb_backbone_name == "r26":
                self.rgb_backbone = resnet26d(
                    pretrained=True,
                    in_chans=in_chans,
                    features_only=True,
                    out_indices=[4],
                )
            elif rgb_backbone_name == "r18":
                self.rgb_backbone = resnet18d(
                    pretrained=True,
                    in_chans=in_chans,
                    features_only=True,
                    out_indices=[4],
                )
            # Removed lidar_backbone as lidar is not used
            rgb_embed_layer = partial(HybridEmbed, backbone=self.rgb_backbone)


            if use_mmad_pretrain:
                params = torch.load(use_mmad_pretrain)["state_dict"]
                updated_params = OrderedDict()
                for key in params:
                    if "backbone" in key:
                        updated_params[key.replace("backbone.", "")] = params[key]
                self.rgb_backbone.load_state_dict(updated_params)

            self.rgb_patch_embed = rgb_embed_layer(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            # Removed lidar_patch_embed as lidar is not used
        else:
            if rgb_backbone_name == "r50":
                self.rgb_backbone = resnet50d(
                    pretrained=True, in_chans=3, features_only=True, out_indices=[4]
                )
            elif rgb_backbone_name == "r101":
                self.rgb_backbone = resnet101d(
                    pretrained=True, in_chans=3, features_only=True, out_indices=[4]
                )
            elif rgb_backbone_name == "r26":
                self.rgb_backbone = resnet26d(
                    pretrained=True, in_chans=3, features_only=True, out_indices=[4]
                )
            elif rgb_backbone_name == "r18":
                self.rgb_backbone = resnet18d(
                    pretrained=True, in_chans=3, features_only=True, out_indices=[4]
                )
            embed_layer = partial(HybridEmbed, backbone=self.rgb_backbone)

            self.rgb_patch_embed = embed_layer(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            # Removed lidar_patch_embed as lidar is not used


        # Adjusted for 2 views (left and right)
        self.global_embed = nn.Parameter(torch.zeros(1, embed_dim, 2))
        self.view_embed = nn.Parameter(torch.zeros(1, embed_dim, 2, 1))

        # Adjusted query embed sizes for logical consistency with prediction heads and 2 views
        if self.end2end:
             # Assuming end-to-end still predicts 4 waypoints based on GRUWaypointsPredictor init
            self.query_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 4)) # 4 query positions for waypoints
            self.query_embed = nn.Parameter(torch.zeros(4, 1, embed_dim)) # 4 query tokens for waypoints
        elif self.waypoints_pred_head == "heatmap":
            # Heatmap head expects 400 spatial tokens and 5 non-spatial tokens based on original query_embed size 405.
            # Let's keep this structure.
            self.query_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 5)) # 5 query positions for non-spatial (global/heatmap related)
            self.query_embed = nn.Parameter(torch.zeros(400 + 5, 1, embed_dim)) # 400 spatial + 5 non-spatial tokens
        else: # non-heatmap heads (gru, gru-command, linear, linear-sum)
             # Assuming 400 spatial tokens for traffic and 13 non-spatial tokens for other heads (1 junction, 1 light, 1 stop, 10 waypoints)
            self.query_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 13)) # 13 query positions for non-spatial
            self.query_embed = nn.Parameter(torch.zeros(400 + 13, 1, embed_dim)) # 400 spatial + 13 non-spatial tokens


        if self.end2end:
            self.waypoints_generator = GRUWaypointsPredictor(embed_dim, 4) # 4 waypoints
        elif self.waypoints_pred_head == "heatmap":
            self.waypoints_generator = MultiPath_Generator(
                embed_dim + 32, embed_dim, 10 # Generates 10 waypoints using spatial softmax
            )
        elif self.waypoints_pred_head == "gru":
            self.waypoints_generator = GRUWaypointsPredictor(embed_dim, 10) # 10 waypoints
        elif self.waypoints_pred_head == "gru-command":
            self.waypoints_generator = GRUWaypointsPredictorWithCommand(embed_dim, 10) # 10 waypoints
        elif self.waypoints_pred_head == "linear":
            self.waypoints_generator = LinearWaypointsPredictor(embed_dim, cumsum=False) # Assuming linear predicts delta waypoints, 10 waypoints
        elif self.waypoints_pred_head == "linear-sum":
            self.waypoints_generator = LinearWaypointsPredictor(embed_dim, cumsum=True) # Assuming linear-sum predicts cumulative waypoints, 10 waypoints

        self.junction_pred_head = nn.Linear(embed_dim, 2)
        self.traffic_light_pred_head = nn.Linear(embed_dim, 2)
        self.stop_sign_head = nn.Linear(embed_dim, 2)

        if self.traffic_pred_head_type == "det":
             # Input dimension depends on the traffic_feature size (400 tokens * embed_dim per token if flattened, or just embed_dim if applied per token)
             # and velocity (32). Original code concatenated after slicing 400 tokens.
             # Assuming traffic_feature is shape (bs, 400, embed_dim) and velocity is (bs, 400, 32) after repeat.
             # Concatenation is along dim 2, resulting in (bs, 400, embed_dim + 32).
             # The linear layer expects (bs * 400, embed_dim + 32). So input dim is embed_dim + 32. This seems correct.
            self.traffic_pred_head = nn.Sequential(
                *[
                    nn.Linear(embed_dim + 32, 64),
                    nn.ReLU(),
                    nn.Linear(64, 7), # Outputting 7 values per spatial location (e.g., probabilities for different traffic elements/states)
                    nn.Sigmoid(), # Sigmoid for probabilities
                ]
            )
        elif self.traffic_pred_head_type == "seg":
             # Segmentation head likely predicts a single value per spatial token. Input is embed_dim per token.
            self.traffic_pred_head = nn.Sequential(
                *[nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()]
            )


        self.position_encoding = PositionEmbeddingSine(embed_dim // 2, normalize=True)

        encoder_layer = TransformerEncoderLayer(
            embed_dim, num_heads, dim_feedforward, dropout, act_layer, normalize_before
        )
        self.encoder = TransformerEncoder(encoder_layer, enc_depth, None)

        decoder_layer = TransformerDecoderLayer(
            embed_dim, num_heads, dim_feedforward, dropout, act_layer, normalize_before
        )
        decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder = TransformerDecoder(
            decoder_layer, dec_depth, decoder_norm, return_intermediate=False
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.global_embed)
        nn.init.uniform_(self.view_embed)
        nn.init.uniform_(self.query_embed)
        nn.init.uniform_(self.query_pos_embed)

    def forward_features(
        self,
        left_image,
        right_image,
    ):
        features = []

        # Left view processing
        left_image_token, left_image_token_global = self.rgb_patch_embed(left_image)
        if self.use_view_embed:
            # Use index 0 for left camera
            left_image_token = (
                left_image_token
                + self.view_embed[:, :, 0:1, :]
                + self.position_encoding(left_image_token)
            )
        else:
            left_image_token = left_image_token + self.position_encoding(
                left_image_token
            )
        left_image_token = left_image_token.flatten(2).permute(2, 0, 1)
        left_image_token_global = (
            left_image_token_global
            + self.view_embed[:, :, 0, :]
            + self.global_embed[:, :, 0:1] # Use index 0 for left global
        )
        left_image_token_global = left_image_token_global.permute(2, 0, 1)
        features.extend([left_image_token, left_image_token_global])

        # Right view processing
        right_image_token, right_image_token_global = self.rgb_patch_embed(
            right_image
        )
        if self.use_view_embed:
            # Use index 1 for right camera
            right_image_token = (
                right_image_token
                + self.view_embed[:, :, 1:2, :]
                + self.position_encoding(right_image_token)
            )
        else:
            right_image_token = right_image_token + self.position_encoding(
                right_image_token
            )
        right_image_token = right_image_token.flatten(2).permute(2, 0, 1)
        right_image_token_global = (
            right_image_token_global
            + self.view_embed[:, :, 1, :]
            + self.global_embed[:, :, 1:2] # Use index 1 for right global
        )
        right_image_token_global = right_image_token_global.permute(2, 0, 1)
        features.extend([right_image_token, right_image_token_global])

        # features concatenation now only includes left and right camera features
        features = torch.cat(features, 0)
        return features

    def forward(self, x):
        # Expecting 'rgb_left', 'rgb_right', 'measurements', and 'target_point' in x
        left_image = x["rgb_left"]
        right_image = x["rgb_right"]
        measurements = x["measurements"]
        target_point = x["target_point"]

        # direct_concat is set to False in the example baseline, so this block is likely inactive.
        if self.direct_concat:
             _logger.warning("Direct concat logic might need adjustment for only two camera inputs if used.")
             pass # Keeping original logic but with warning

        # Modified call to forward_features with only required inputs
        features = self.forward_features(
            left_image,
            right_image,
        )

        bs = left_image.shape[0] # Use batch size from one of the input images

        # Constructing the target sequence for the decoder
        if self.end2end:
             # Target for end-to-end is based on query_pos_embed for waypoints
            tgt = self.query_pos_embed.repeat(bs, 1, 1)
        else:
             # Target includes spatial position encoding (400 tokens) and non-spatial query positions
            tgt = self.position_encoding(
                torch.ones((bs, 1, 20, 20), device=left_image.device) # Assuming 20x20 spatial grid
            )
            tgt = tgt.flatten(2) # Flatten spatial dimensions -> 400 tokens

            # Concatenate spatial position encoding with non-spatial query position embeddings
            tgt = torch.cat([tgt, self.query_pos_embed.repeat(bs, 1, 1)], 2)

        tgt = tgt.permute(2, 0, 1) # Permute to (sequence_length, batch_size, embed_dim)

        # Encoder processes fused features
        memory = self.encoder(features, mask=self.attn_mask)

        # Decoder takes query embeddings and encoder memory to produce output tokens (hs)
        # The number of output tokens (sequence_length of hs) is determined by the size of query_embed
        hs = self.decoder(self.query_embed.repeat(1, bs, 1), memory, query_pos=tgt)[0]

        hs = hs.permute(1, 0, 2)  # Permute to (batch_size, sequence_length, embed_dim)

        # Slicing of hs depends on the waypoints prediction head type and the corresponding query_embed size

        if self.end2end:
             # For end-to-end, hs contains the waypoint tokens directly
            waypoints = self.waypoints_generator(hs, target_point)
            # The end-to-end head only returns waypoints
            return waypoints
        elif self.waypoints_pred_head == "heatmap":
            # Heatmap case: query_embed size is 400 + 5 = 405
            # Slicing: 400 spatial tokens + 5 non-spatial tokens
            traffic_feature = hs[:, :400] # First 400 tokens for traffic prediction
            # Assuming the 5 non-spatial tokens are used for heatmap-related waypoints and other predictions
            # Original slicing used hs[:, 400] for junction, light, stop, and hs[:, 401:405] for waypoints_feature
            # Let's adhere to the original slicing for heatmap case as it aligns with the query_embed size
            is_junction_feature = hs[:, 400]
            traffic_light_state_feature = hs[:, 400] # Original code used the same token
            stop_sign_feature = hs[:, 400] # Original code used the same token
            waypoints_feature = hs[:, 401:405] # 4 tokens for heatmap waypoint generator input

            waypoints = self.waypoints_generator(waypoints_feature, measurements)

            is_junction = self.junction_pred_head(is_junction_feature)
            traffic_light_state = self.traffic_light_pred_head(traffic_light_state_feature)
            stop_sign = self.stop_sign_head(stop_sign_feature)

            # Traffic prediction input
            velocity = measurements[:, 6:7].unsqueeze(-1)
            velocity = velocity.repeat(1, 400, 32)
            traffic_feature_with_vel = torch.cat([traffic_feature, velocity], dim=2)
            traffic = self.traffic_pred_head(traffic_feature_with_vel)

            return traffic, waypoints, is_junction, traffic_light_state, stop_sign, traffic_feature

        else: # Non-heatmap cases (gru, gru-command, linear, linear-sum)
             # Non-heatmap case: query_embed size is 400 + 13 = 413
             # Slicing: 400 spatial tokens + 13 non-spatial tokens
            traffic_feature = hs[:, :400] # First 400 tokens for traffic prediction
            # Allocate distinct tokens for junction, light, stop, and waypoints (10 tokens)
            is_junction_feature = hs[:, 400] # Token 400 for junction
            traffic_light_state_feature = hs[:, 401] # Token 401 for traffic light
            stop_sign_feature = hs[:, 402] # Token 402 for stop sign
            waypoints_feature = hs[:, 403:413] # Tokens 403-412 (10 tokens) for waypoint generator input

            if self.waypoints_pred_head == "gru":
                waypoints = self.waypoints_generator(waypoints_feature, target_point)
            elif self.waypoints_pred_head == "gru-command":
                waypoints = self.waypoints_generator(waypoints_feature, target_point, measurements)
            elif self.waypoints_pred_head == "linear" or self.waypoints_pred_head == "linear-sum":
                 # Linear generators expect input shape (bs, 10, embed_dim) which matches waypoints_feature
                waypoints = self.waypoints_generator(waypoints_feature, measurements) # Linear generators also take measurements

            is_junction = self.junction_pred_head(is_junction_feature)
            traffic_light_state = self.traffic_light_pred_head(traffic_light_state_feature)
            stop_sign = self.stop_sign_head(stop_sign_feature)

            # Traffic prediction input remains the same
            velocity = measurements[:, 6:7].unsqueeze(-1)
            velocity = velocity.repeat(1, 400, 32)
            traffic_feature_with_vel = torch.cat([traffic_feature, velocity], dim=2)
            traffic = self.traffic_pred_head(traffic_feature_with_vel)

            return traffic, waypoints, is_junction, traffic_light_state, stop_sign, traffic_feature


@register_model
def interfuser_baseline(**kwargs):
    model = Interfuser(
        enc_depth=6,
        dec_depth=6,
        embed_dim=256,
        rgb_backbone_name="r50",
        lidar_backbone_name="r18", # This parameter is effectively ignored when with_lidar=False
        waypoints_pred_head="gru", # Example: using GRU waypoints predictor
        use_different_backbone=True,
        with_lidar=False, # Explicitly set to False
        with_center_sensor=False, # Explicitly set to False
        with_right_left_sensors=True, # Explicitly set to True
    )
    return model