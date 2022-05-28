import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from config import Config


def _expand(x, nclass):
    return x.unsqueeze(1).repeat(1, nclass, 1, 1, 1).flatten(0, 1)


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True,
                 bias=False, norm_layer=nn.BatchNorm2d):
        super().__init__()
        depthwise = nn.Conv2d(inplanes, inplanes, kernel_size,
                              stride=stride, padding=dilation,
                              dilation=dilation, groups=inplanes, bias=bias)
        bn_depth = norm_layer(inplanes)
        pointwise = nn.Conv2d(inplanes, planes, 1, bias=bias)
        bn_point = norm_layer(planes)

        if relu_first:
            self.block = nn.Sequential(OrderedDict([('relu', nn.ReLU()),
                                                    ('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point)
                                                    ]))
        else:
            self.block = nn.Sequential(OrderedDict([('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('relu1', nn.ReLU(inplace=True)),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point),
                                                    ('relu2', nn.ReLU(inplace=True))
                                                    ]))

    def forward(self, x):
        return self.block(x)


class TransformerHead(nn.Module):
    def __init__(self, c1_channels=256, c4_channels=2048, hid_dim=64, norm_layer=nn.BatchNorm2d, args=None):
        super().__init__()

        # TODO Eslam: make this generic using args.
        last_channels = args.hidden_dim
        nhead = 1

        self.conv_c1 = _ConvBNReLU(c1_channels, hid_dim, 1, norm_layer=norm_layer)

        self.lay1 = SeparableConv2d(last_channels+nhead, hid_dim, 3, norm_layer=norm_layer, relu_first=False)
        self.lay2 = SeparableConv2d(hid_dim, hid_dim, 3, norm_layer=norm_layer, relu_first=False)
        self.lay3 = SeparableConv2d(hid_dim, hid_dim, 3, norm_layer=norm_layer, relu_first=False)

        self.pred = nn.Conv2d(hid_dim, 1, 1)

    def forward(self, c1, feat_enc, attns):
        B, nclass, nhead, _ = attns.shape  # [batch, nclass, 1, d]
        _, _, H, W = feat_enc.shape  # [batch, d, H, W]
        # TODO: Eslam: I should investigate why/how Trans2seg output nhead in the shape?!!!
        attns = attns.reshape(B*nclass, nhead, H, W)

        x = torch.cat([_expand(feat_enc, nclass), attns], 1)

        x = self.lay1(x)
        x = self.lay2(x)

        size = c1.size()[2:]
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        c1 = self.conv_c1(c1)
        x = x + _expand(c1, nclass)

        x = self.lay3(x)
        x = self.pred(x).reshape(B, nclass, size[0], size[1])
        x = F.interpolate(x, (Config.input_size, Config.input_size), mode='bilinear', align_corners=True)
        # [b, #classes, Config.input_size, Config.input_size]

        return x
