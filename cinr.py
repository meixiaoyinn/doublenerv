import os
import torch
import platform
import numpy as np

from tqdm import tqdm
from time import time
import torch.nn as nn
from argparse import ArgumentParser
import scipy.io as sio
from PositionEncoding import *
from net import *
from doublenerv import *
from utils import *
from skimage.metrics import structural_similarity as sk_cpt_ssim
import cv2

def data_process(data_dir,frame_name):
    datapath = os.path.join('datasets',data_dir, frame_name)
    kobe = sio.loadmat(datapath)
    meas = torch.from_numpy(kobe['meas'])/255
    gt = torch.from_numpy(kobe['orig'])/255
    mask = torch.from_numpy(kobe['mask'])
    return meas.permute(2,0,1).float().cuda(),gt.permute(2,0,1).unsqueeze(0).float().cuda(),mask.float().permute(2,0,1).unsqueeze(0).cuda()

def psnr(img1, img2):         #计算信噪比
    l2_loss = F.mse_loss(img1, img2, reduction='mean')
    psnr = -10 * torch.log10(l2_loss)
    psnr = psnr.view(1, 1).expand(output.size(0), -1)
    return psnr
    # img1.astype(np.float32)
    # img2.astype(np.float32)
    # mse = np.mean((img1 - img2) ** 2)
    # if mse == 0:
    #     return 100
    # PIXEL_MAX = 1
    # return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

parser=ArgumentParser(description='VSS')
'''dir setting'''
parser.add_argument('--dataset_dir',type=str,default='cacti',help='video dataset dir')
parser.add_argument('--dataset_name',type=str,default='kobe_cacti',help='dataset_name')
# parser.add_argument('--model_dir', type=str, default='E:/data/net', help='trained or pre-trained model directory')
# parser.add_argument('--model_name', type=str, default='VSS_3.pkl', help='trained or pre-trained model name')
# parser.add_argument('--video_transform',type=bool,default=False,help='decide if transform video to frame')
# parser.add_argument('--frame_dir',type=str,default='E:/data/Datasets/ss-video/video_frames',help='frame dir')
'''net parameter set'''
parser.add_argument('--frame_num',type=int,default=8,help='cs frames')
parser.add_argument('--train_epoch',type=int,default=20,help='epoch number of testing')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate of model')
parser.add_argument('--blocksize', type=int, default=0, help='image block size of each train cycle')
parser.add_argument('--norm', default='none', type=str, help='norm layer for generator', choices=['none', 'bn', 'in'])
parser.add_argument('--act', type=str, default='gelu', help='activation to use',
                    choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])

parser.add_argument('--t_dim', nargs='+', default=[64,128,256], type=int, help='temporal resolution of grids')
parser.add_argument('--wbit', default=32, type=int, help='QAT weight bit width')
parser.add_argument('--M', default=16, type=int, help='patch nums')
parser.add_argument('--sf', default=2, type=int, help='scalling factor')

parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--L_spatial', type=str, default=32, help='position encoding at spatial')
parser.add_argument('--L_temporal', type=str, default=8, help='position encoding at temper')
parser.add_argument('--embed', type=str, default='1.25_80', help='base value/embed length for position encoding')
parser.add_argument('--height', type=int, default=256, help='image height')
parser.add_argument('--width', type=int, default=256, help='image width')
parser.add_argument('--out_dim', type=int, default=8, help='output dims')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--lr', type=int, default=0.004, help='learning rate')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--end_epoch', type=int, default=1000000, help='end epoch')

args=parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
video_dataset=Video_dataset(args)
dataloader=torch.utils.data.DataLoader(video_dataset,batch_size=1, shuffle=False)
# meas,gt,mask=data_process(args.dataset_dir,'kobe_cacti.mat')
# mask_s=torch.sum(mask,1)+1
# mask_s_loss=torch.sum(1-mask,1)
# meas_y=torch.div(meas,mask_s)
pos = np.transpose(positionencoding2D(args.height, args.width, args.L_spatial, 'sin_cos',0), (2, 0, 1))
pos = torch.from_numpy(pos).cuda().unsqueeze(0)

start_epoch = args.start_epoch
end_epoch=args.end_epoch
# pe_t=PositionalEncoding_temp(args.embed)
model=Einet(args).cuda()
# model=rec_net(args).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# x = At(meas_y[0], mask)
# yb = A(x, mask)
# x = x + At(torch.div(meas_y[0]-yb.squeeze(), mask_s), mask)
max_psnr=np.zeros(args.out_dim)
for i ,(data,frame_idx) in enumerate(dataloader):
    x_input=data[0].clone()
    for epoch_i in range(start_epoch+1, end_epoch):
        # gt = gt[:,:8]
        # psnr_ = 0
        # ssim_ = 0
        # model.train()
        # optimizer.zero_grad()
        # output = model(x, pos)
        # pred_meas = torch.sum(output * mask, 1).squeeze(0)
        # loss_mlp = nn.SmoothL1Loss()(output, meas_y[0])
        gt=data[3]
        mask=data[2]
        psnr_ = 0
        ssim_ = 0
        model.train()
        optimizer.zero_grad()

        # output=model(x_input,pos)
        output = model(x_input, frame_idx)
        pred_meas=torch.sum(output*mask,1).squeeze(0)
        loss_mlp = nn.SmoothL1Loss()(pred_meas, data[1])
        loss_mlp=loss_mlp.mean()
        loss_mlp.backward(retain_graph=True)
        optimizer.step()
        x_input = output.detach()

        for i in range(output.shape[1]):
            X_rec = output[0, i, :, :]
            X_gt = gt[0, i, :, :]
            X_gt = X_gt.cpu().detach()
            X_rec = X_rec.cpu().detach()
            rec_PSNR = psnr(X_rec, X_gt)
            if max_psnr[i] < psnr_:
                max_psnr[i]=psnr_
                cv2.imwrite('out/kobe_{}.jpg'.format(i), X_rec.numpy() * 255)
            psnr_ += rec_PSNR
            rec_SSIM = sk_cpt_ssim(X_rec.numpy(), X_gt.numpy())
            ssim_ += rec_SSIM
        output_data = "[%02d/%02d] AVG Loss: %.7f, PSNR: %.7f,ssim: %.7f,  l_rate: %.8f" \
                      % (epoch_i, end_epoch, loss_mlp, psnr_/output.shape[1],ssim_/output.shape[1],optimizer.state_dict()['param_groups'][0]['lr'])
        if epoch_i % 10 == 0:
            print(output_data)


