import os

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from model.HNerv import *
from model.Hinerv import *
from torch.utils.data import Dataset


class Video_dataset(Dataset):
    def __init__(self,args):
        self.frames=args.frame_num
        datapath = os.path.join('datasets', args.dataset_dir, args.dataset_name)
        self.data = sio.loadmat(datapath)
        self.meas = self.data['meas'] / 255
        self.gt = self.data['orig'] / 255
        self.mask = self.data['mask']
        self.frame_all=self.gt.shape[2]
        frame_idx=np.arange(0,self.frame_all)
        self.frame_idx = [float(x) / self.frame_all for x in frame_idx]

    def __len__(self):
        return int(self.frame_all/self.frames)

    def __getitem__(self, idx):
        frame_idx = torch.tensor(self.frame_idx[:(idx+1)*self.frames])
        meas=torch.from_numpy(self.meas).permute(2,0,1).float()
        gt=torch.from_numpy(self.gt[:,:,:(idx+1)*self.frames]).permute(2,0,1).float()
        mask=torch.from_numpy(self.mask).float().permute(2,0,1)
        mask_s = torch.sum(mask, 0) + 1
        meas_y = torch.div(meas, mask_s)
        x = At(meas_y[idx], mask)
        yb = A(x, mask)
        x = x + At(torch.div(meas_y[idx] - yb.squeeze(), mask_s), mask)
        meas=meas[idx]
        return [x.cuda(),meas.float().cuda(),mask.cuda(),gt.cuda()],frame_idx.cuda()


class Einet(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.explicit_net=HNerv(args)
        # self.implict_net=HiNerv(args)
        self.implict_net=HiNerv(args,video_resolution=[8,256,256],feature_dim=64, num_grids=[4, 8])

    def forward(self,x,norm_index):
        _, _, h, w = x.shape
        x = x.view(-1, 1, h, w)
        # x = F.pad(x, (32, 32, 32, 32))
        x=F.pad(x,(12,12,12,12))
        exp_out=self.explicit_net(x)
        exp_out=exp_out[:,:,12:268,12:268]
        exp_out=exp_out.view(1,-1,h,w)

        #implict reconstruction
        patch_indices = self.generate_patch_indices(8, 256, 256)
        imp_out=self.implict_net(patch_indices)
        return exp_out

    def generate_patch_indices(self, T, H, W):
        """
        Generate patch indices for a given video resolution.

        Args:
            - T (int): spatial resolution .
            - H (int): height .
            - W (int): width .

        Returns:
        - patch_indices (torch.Tensor): A tensor of shape (T, H, W, 3) containing the patch indices.
        """

        # Generate coordinates for each dimension
        t_coords = torch.arange(T).view(T, 1, 1).float() / T
        h_coords = torch.arange(H).view(1, H, 1).float() / H
        w_coords = torch.arange(W).view(1, 1, W).float() / W

        # Expand dimensions to make them broadcastable
        t_coords = t_coords.expand(T, H, W)
        h_coords = h_coords.expand(T, H, W)
        w_coords = w_coords.expand(T, H, W)

        # Stack the coordinates along the last dimension
        patch_indices = torch.stack([t_coords, h_coords, w_coords], dim=-1)

        return patch_indices