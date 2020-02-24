import torch.nn as nn
import torch.nn.functional as F


# 1. conv_ws_2d = weight norm + conv2d
#    TODOs: investigate why we need this kind of conv?
# 2. nn.Module: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html
#    Base class for all network modules:
#    a. tree structure, top down control, change top level module can influence the submodules
#       self._modules[name] = module, 
#                 a                2 relations: run order, occupancy
#         /    /  |   \  \
#        b -> c-> d <- e -> f
#                     /|\
#    
#    b. __init__ forward.
def conv_ws_2d(input,
               weight,  # weight is changed in the function
               bias=None,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               eps=1e-5):
    c_in = weight.size(0)
    weight_flat = weight.view(c_in, -1)
    mean = weight_flat.mean(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    std = weight_flat.std(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    weight = (weight - mean) / (std + eps)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


class ConvWS2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 eps=1e-5):
        super(ConvWS2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.eps = eps

    def forward(self, x):
        return conv_ws_2d(x, self.weight, self.bias, self.stride, self.padding,
                          self.dilation, self.groups, self.eps)
