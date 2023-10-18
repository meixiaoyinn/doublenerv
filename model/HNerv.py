from datetime import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from math import pi, sqrt, ceil
import numpy as np

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class HNerv(nn.Module):
    def __init__(self,args):
        super().__init__()
        strds = [5, 4, 2]
        self.encoder = ConvNeXt(stage_blocks=1, strds=[5,4,2], dims=[64,64,16],
                                drop_path_rate=0)
        hnerv_hw = np.prod(strds) // np.prod(strds)
        self.fc_h, self.fc_w = hnerv_hw, hnerv_hw

        self.conv1to3=nn.Conv2d(1,3,(1,1))
        decoder_layer1 = NeRVBlock(dec_block=False, conv_type='conv', ngf=16, new_ngf=68, ks=0, strd=1,
                                   bias=True, norm=args.norm, act=args.act)

        decoder_layers = []
        decoder_layers.append(decoder_layer1)
        ngf=68
        for i, strd in enumerate(strds):
            # reduction = sqrt(strd) if args.reduce == -1 else args.reduce
            reduction=1.2
            new_ngf = int(max(round(ngf / reduction), 32))
            for j in range(1):
                cur_blk = NeRVBlock(dec_block=True, conv_type='pshuffel', ngf=ngf, new_ngf=new_ngf,
                                    ks=min(1 + 2 * i, 5), strd=1 if j else strd, bias=True, norm=args.norm,
                                    act=args.act)
                decoder_layers.append(cur_blk)
                ngf = new_ngf

        self.decoder = nn.ModuleList(decoder_layers)
        self.head_layer = nn.Conv2d(ngf, 3, 3, 1, 1)
        self.out_bias='tanh'
        self.conv3to1 = nn.Conv2d(3, 1, (1, 1))


    def forward(self,x):
        x=self.conv1to3(x)
        img_embed=self.encoder(x)

        embed_list = [img_embed]
        output = self.decoder[0](img_embed)
        n, c, h, w = output.shape
        output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0, 1, 4, 2, 5, 3).reshape(n, -1, self.fc_h * h,
                                                                                                  self.fc_w * w)
        embed_list.append(output)
        for layer in self.decoder[1:]:
            output = layer(output)
            embed_list.append(output)
        img_out = OutImg(self.head_layer(output), self.out_bias)
        img_out=self.conv3to1(img_out)
        return img_out


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, stage_blocks=0, strds=[2,2,2,2], dims=[96, 192, 384, 768],
            in_chans=3, drop_path_rate=0., layer_scale_init_value=1e-6,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stage_num = len(dims)
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, stage_blocks*self.stage_num)]
        cur = 0
        for i in range(self.stage_num):
            # Build downsample layers
            if i > 0:
                downsample_layer = nn.Sequential(
                        LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
                        nn.Conv2d(dims[i-1], dims[i], kernel_size=strds[i], stride=strds[i]),
                )
            else:
                downsample_layer = nn.Sequential(
                    nn.Conv2d(in_chans, dims[0], kernel_size=strds[i], stride=strds[i]),
                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
                )
            self.downsample_layers.append(downsample_layer)

            # Build more blocks
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(stage_blocks)]
            )
            self.stages.append(stage)
            cur += stage_blocks

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_list = []
        for i in range(self.stage_num):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            out_list.append(x)
        return out_list[-1]


def OutImg(x, out_bias='tanh'):
    if out_bias == 'sigmoid':
        return torch.sigmoid(x)
    elif out_bias == 'tanh':
        return (torch.tanh(x) * 0.5) + 0.5
    else:
        return x + float(out_bias)

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class NeRVBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        conv = UpConv if kargs['dec_block'] else DownConv
        self.conv = conv(ngf=kargs['ngf'], new_ngf=kargs['new_ngf'], strd=kargs['strd'], ks=kargs['ks'],
            conv_type=kargs['conv_type'], bias=kargs['bias'])
        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class UpConv(nn.Module):
    def __init__(self, **kargs):
        super(UpConv, self).__init__()
        ks, ngf, new_ngf, strd = kargs['ks'], kargs['ngf'], kargs['new_ngf'], kargs['strd']
        if  kargs['conv_type']  == 'pshuffel':
            self.upconv = nn.Sequential(
                nn.Conv2d(ngf, new_ngf * strd * strd, ks, 1, ceil((ks - 1) // 2), bias=kargs['bias']),
                nn.PixelShuffle(strd) if strd !=1 else nn.Identity(),
            )
        elif  kargs['conv_type']  == 'conv':
            self.upconv = nn.ConvTranspose2d(ngf, new_ngf, ks+strd, strd, ceil(ks / 2))
        elif  kargs['conv_type']  == 'interpolate':
            self.upconv = nn.Sequential(
                nn.Upsample(scale_factor=strd, mode='bilinear',),
                nn.Conv2d(ngf, new_ngf, strd + ks, 1, ceil((ks + strd -1) / 2), bias=kargs['bias'])
            )

    def forward(self, x):
        return self.upconv(x)


class DownConv(nn.Module):
    def __init__(self, **kargs):
        super(DownConv, self).__init__()
        ks, ngf, new_ngf, strd = kargs['ks'], kargs['ngf'], kargs['new_ngf'], kargs['strd']
        if kargs['conv_type'] == 'pshuffel':
            self.downconv = nn.Sequential(
                nn.PixelUnshuffle(strd) if strd != 1 else nn.Identity(),
                nn.Conv2d(ngf * strd ** 2, new_ngf, ks, 1, ceil((ks - 1) // 2), bias=kargs['bias'])
            )
        elif kargs['conv_type'] == 'conv':
            self.downconv = nn.Conv2d(ngf, new_ngf, ks + strd, strd, ceil(ks / 2), bias=kargs['bias'])
        elif kargs['conv_type'] == 'interpolate':
            self.downconv = nn.Sequential(
                nn.Upsample(scale_factor=1. / strd, mode='bilinear', ),
                nn.Conv2d(ngf, new_ngf, ks + strd, 1, ceil((ks + strd - 1) / 2), bias=kargs['bias'])
            )

    def forward(self, x):
        return self.downconv(x)


def NormLayer(norm_type, ch_width):
    if norm_type == 'none':
        norm_layer = nn.Identity()
    elif norm_type == 'bn':
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == 'in':
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer


def ActivationLayer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    # elif act_type == 'sin':
    #     act_layer = Sin
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer