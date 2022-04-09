import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import warnings

from mmcv.cnn import constant_init, kaiming_init



def get_activation(name="silu", inplace=False):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=False)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "noact":
        module = nn.Identity()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def bias_init_with_prob(prior_prob):
    """ initialize conv/fc bias value according to giving probablity"""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

class PositionEncoder(nn.Module):
    def __init__(self, device):
        super(PositionEncoder, self).__init__()
        self.device = device

    def forward(self, feature):
        B, C, H, W = feature.shape #B, C, H, W
        x = torch.linspace(-1, 1, W).reshape(1, 1, 1, W).to(self.device)
        y = torch.linspace(-1, 1, H).reshape(1, 1, H, 1).to(self.device)
        x_encode = x.repeat(B, 1, H, 1)
        y_encode = y.repeat(B, 1, 1, W)
        encoded_feature = torch.cat([feature, x_encode, y_encode], dim = 1)
        
        return encoded_feature

def get_norm(out_channels, name=None):
    assert name is not None
    if name == "BN":
        module = nn.BatchNorm2d(out_channels)
    elif name == "GN":
        module = nn.GroupNorm(num_groups=32, num_channels=out_channels)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride,padding = None, groups=1, bias=False, act="silu", norm="BN"
    ):
        super().__init__()
        # same padding
        if padding is None:
            pad = (ksize - 1) // 2
        else:
            pad = padding
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.norm = get_norm(out_channels, norm) if norm is not None else None
        self.act = get_activation(act, inplace=False)

        self.init_weights()

    def init_weights(self):
        kaiming_init(self.conv, nonlinearity='relu')
        if self.norm is not None:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x):
        if self.norm is not None:
            return self.act(self.norm(self.conv(x)))
        else:
            return self.act(self.conv(x))

    def fuseforward(self, x):
        return self.act(self.conv(x))

    
class OnlyConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride,padding = None, groups=1, bias=False
    ):
        super().__init__()
        # same padding
        if padding is None:
            pad = (ksize - 1) // 2
        else:
            pad = padding
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)

    # def fuseforward(self, x):
    #     return self.act(self.conv(x))
    
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = BaseConv(in_channels, c_, 1, 1)
        self.cv2 = BaseConv(in_channels, c_, 1, 1)
        self.cv3 = BaseConv(2 * c_, out_channels, 1, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = BaseConv(in_channels, c_, 1, 1)
        self.cv2 = BaseConv(c_, out_channels, 3, 1, groups=g)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, in_channels, out_channels, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = BaseConv(in_channels, c_, 1, 1)
        self.cv2 = BaseConv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class YoloSegTower(nn.Module):
# Concatenate a list of tensors along dimension
    def __init__(self, kdim=(320,640,1280), midim=320, outdim=320, dimension=1):
        super().__init__()
        self.d = dimension
        self.m = nn.ModuleList([BaseConv(x,midim,1,1) for x in kdim])
        self.seg = nn.Sequential(
                BaseConv(midim*len(kdim), outdim, 1,1),
                Ctx4(outdim, red=outdim//2))

    def forward(self, x):
        B,C,H,W = x[0].shape
        outs = torch.cat([F.interpolate(m(feat), (H,W),mode='bilinear',align_corners=True) 
                            for m, feat in zip(self.m,x)], dim=1)
        # seg_feats, ctx = self.seg(outs)
        return self.seg(outs) #self.cls(seg_feats), ctx

class SegHead(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2,1, bias=True)

    def forward(self, x):
        return self.conv(x)

class KernelHead(nn.Module):
    def __init__(self, kdim=(320, 640, 1280), outdim = 64):
        super().__init__()
        self.outdim = outdim
        self.anchor_num = 3
        self.m = nn.ModuleList([OnlyConv(x, self.outdim * self.anchor_num, 1 ,1) for x in kdim]) #outdim * anchor num (64 * 3)

    def forward(self, x):
        output = []
        for i in range(len(x)):
            feat = self.m[i](x[i])

            B, C, H, W = feat.shape
            #x[i] = x[i].view(B, self.anchor_num, self.outdim, H, W).permute(0, 1, 3, 4, 2).contiguous()
            feat = feat.permute(0, 2, 3, 1).contiguous().view(B, -1, self.outdim)
            output.append(feat)
        output = torch.cat(output, dim = 1)
        
        return output
        

class Ctx4(nn.Module):
    def __init__(self, inplanes, red, nlayer=1):
        '''
            2 residual addition, CAT 3 features. 
        '''
        super().__init__()
        self.red_inputs = BaseConv(inplanes, inplanes//2,1,1)
        self.block1 = nn.Sequential(*[CtxBottle13(inplanes, inplanes//2) for _ in range(nlayer)])
        self.block2 = nn.Sequential(*[CtxBottle13(inplanes//2, inplanes//2) for _ in range(nlayer)])
        self.ctx_reduce = BaseConv(2*inplanes,red,1,1)
        self.ctx_fuse = BaseConv((inplanes//2+red), (inplanes//2+red),1,1)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        ctx = self.ctx_reduce(torch.cat((x,x1,x2), dim=1))
        out = self.ctx_fuse(torch.cat((x2,ctx), dim=1))
        return out


class CtxBottle13(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=False, g=1, e=1, act=nn.SiLU):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = BaseConv(c1, c_, 1, 1)
        self.cv2 = OnlyConv(c_, c2, 3, 1, groups=g)
        self.act = act(inplace=True)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return self.act(x + self.cv2(self.cv1(x))) if self.add else self.act(self.cv2(self.cv1(x)))
