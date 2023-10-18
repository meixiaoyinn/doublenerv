import torch

def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,dim=0,keepdim=True)
    return y

def At(y,Phi):
    x = y*Phi
    return x