import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import scipy.io as sio

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


class ConvMod(nn.Module):
    def __init__(self, dim,outdim):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
            nn.Conv2d(dim, outdim, 1),
            nn.GELU(),
            nn.Conv2d(outdim, outdim, 11, padding=5, groups=outdim)
        )
        self.v = nn.Conv2d(dim, outdim, 1)
        self.proj = nn.Conv2d(outdim, outdim, 1)
    def forward(self, x):
        # B, C, H, W = x.shape

        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)
        return x

class Spatial_block(nn.Module):
    def __init__(self, L_spatial, out_dim,activation='GELU'):
        super(Spatial_block, self).__init__()
        self.L = L_spatial

        act_func=self.ActivationLayer(activation)
        self.layers1 = nn.Sequential(
            nn.Conv2d(4 * self.L +2+ out_dim*4, 4 * self.L , (1, 1)),
            nn.InstanceNorm2d(4 * self.L , affine=True),
            act_func, )

        self.layers2 = nn.Sequential(nn.Conv2d(4 * self.L , 3 * self.L , (1, 1)),
                                     nn.InstanceNorm2d(3 * self.L , affine=True),
                                     act_func)

        self.layers3 = nn.Sequential(nn.Conv2d(3 * self.L, 2 * self.L , (1, 1)),
                                     nn.InstanceNorm2d(2 * self.L , affine=True),
                                     act_func, )
        self.layers4 = nn.Sequential(nn.Conv2d(2 * self.L, 1 * self.L , (1, 1)),
                                     nn.InstanceNorm2d(1 * self.L , affine=True),
                                     act_func,
                                     nn.Conv2d(1 * self.L, out_dim, (1, 1))
                                     )


    def ActivationLayer(self,act_type):
        if act_type=='relu':
            act_layer=nn.ReLU(inplace=True)
        elif act_type=='leaky_relu':
            act_layer=nn.LeakyReLU(inplace=True)
        elif act_type=='GELU':
            act_layer=nn.GELU()
        else:act_layer=None
        # elif act_type=='swish':
        #     act_layer=Swish()
        return act_layer


    def add_noise(self,inputs, std_dev):
        # 创建一个与输入形状相同的正态分布噪声张量
        noise = torch.randn_like(inputs) * std_dev
        # 将噪声张量添加到输入层
        noisy_input = inputs + noise
        return noisy_input


    def forward(self, x,attachedvec):
        attachedvec=self.add_noise(attachedvec,0.1)
        out1 = self.layers1(torch.cat([x,attachedvec], 1))
        out1 = self.add_noise(out1, 0.1)
        out2=self.layers2(out1)
        out2 = self.add_noise(out2, 0.1)
        out3 = self.layers3(out2)
        out3 = self.add_noise(out3, 0.1)
        out = self.layers4(out3)
        return out



# class block(nn.Module):
#     def __init__(self,input_channels,out_channels):
#         super(block,self).__init__()
#         self.local_list=nn.Sequential(nn.Conv2d(input_channels,input_channels+input_channels//2,3,1,padding=1),
#                                       nn.InstanceNorm2d(input_channels+input_channels//2 , affine=True),nn.GELU(),
#         nn.Conv2d(input_channels+input_channels//2,out_channels,3,1,padding=1),nn.InstanceNorm2d(out_channels, affine=True),nn.GELU())
#
#     def forward(self,x):
#         return self.local_list(x)


class rec_net(nn.Module):
    def __init__(self,args):
        super(rec_net,self).__init__()
        self.spatial_block=[]
        # input_channels=[args.out_dim,args.out_dim+args.out_dim//2,]
        for i in range(1,4):
            self.spatial_block.append(ConvMod(args.out_dim*i,args.out_dim*(i+1)))
        self.spatial_block=nn.ModuleList(self.spatial_block)
        self.spatial_mlp=Spatial_block(args.L_spatial, args.out_dim)
        # self.att = ConvMod(args.out_dim)
        self.add_dim = []
        for i in range(1,4):
            self.add_dim.append(nn.Conv2d(args.out_dim*i,args.out_dim*(i+1),(1,1)))
        self.conv1=nn.ModuleList(self.add_dim)
        self.conv_final = nn.Conv2d(args.out_dim,args.out_dim*4,(1,1))
        # self.temp_block=Temp_block()


    def forward(self,x,pos):
        x_l=x.clone()
        x_g=x.clone()
        for i in range(3):
            x_l=self.spatial_block[i](x_l)
            x_g=self.conv1[i](x_g)
            x_l+=x_g
        x=self.conv_final(x)
        x+=x_l
        out=self.spatial_mlp(pos,x)

        return out