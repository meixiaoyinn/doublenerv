import torch
import torch.nn as nn
import torch.nn.functional as F

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

class stem(nn.Module):
    def __init__(self,input_channerls,out_channerls):
        super().__init__()
        self.stem=nn.Conv2d(input_channerls,out_channerls,(1,1))
        self.norm = LayerNorm(out_channerls, eps=1e-6, data_format="channels_first")
        self.relu=nn.GELU()

    def forward(self,x):
        return self.relu(self.norm(self.stem(x)))


class HiNerv(nn.Module):
    def __init__(self,args,video_resolution,feature_dim, num_grids):
        super().__init__()
        self.quantize_fn = weight_quantize_fn(args.wbit)
        # Multi-resolution temporal grids
        self.video_grids = nn.ParameterList()
        self.T,self.H,self.W=video_resolution
        for t in num_grids:
            self.video_grids.append(nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.T, feature_dim, self.H, self.W))))

        # for t in args.t_dim:
        #     self.video_grid.append(nn.Parameter(nn.init.xavier_uniform_(torch.empty(t, args.out_dim, args.height//args.M, args.width//args.M))))

        self.stem=stem(args.out_dim,args.out_dim*2)
        self.sf=args.sf


    def forward(self, patch_indices):
        out_list = []

        for grid in self.video_grids:
            # Interpolate grid features along the time dimension
            t_coords = patch_indices[:, :, :, 0] * grid.size(0)
            left = torch.floor(t_coords).long()
            right = torch.clamp(left + 1, 0, grid.size(0) - 1)
            d_left = (t_coords - left).unsqueeze(-1)
            d_right = (right - t_coords).unsqueeze(-1)
            interpolated_features = d_right * grid[left] + d_left * grid[right]
            out_list.append(interpolated_features)

        # Combine the interpolated features (e.g., by concatenation, addition, etc.)
        # This is a placeholder and might need to be adjusted based on the exact requirements
        output = torch.cat(out_list, dim=1)


        return output


class weight_quantize_fn(nn.Module):
    def __init__(self, bit):
        super(weight_quantize_fn, self).__init__()
        self.wbit = bit
        assert self.wbit <= 8 or self.wbit == 32

    def forward(self, x):
        if self.wbit == 32:
            weight_q = x
        else:
            weight = torch.tanh(x)
            weight_q = qfn.apply(weight, self.wbit)
        return weight_q


class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        n = float(2**(k-1) - 1)
        out = torch.floor(torch.abs(input) * n) / n
        out = out*torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

