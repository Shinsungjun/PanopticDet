from copy import deepcopy
from operator import mod
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import BaseConv, C3, SPPF
import pkg_resources as pkg
import numpy as np

class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=None, inplace=True, size = 'x'):  # detection layer
        super().__init__()
        if size == 'x':
            depth_multiple = 1.33 # layer 내 모듈 반복 수 변경
            width_multiple = 1.25 # channel 두께 변경
        elif size == 'l':
            depth_multiple = 1
            width_multiple = 1
        elif size == 'm':
            depth_multiple = 0.67
            width_multiple = 0.75
        else :
            raise ValueError('choose head model size m OR l OR x')
            
        anchors = [[10, 13, 16,30, 33,23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)

        ch = np.array([256, 512, 1024]) * width_multiple
        ch = ch.astype(int)

        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
        self.stride = torch.tensor([8, 16, 32]).cuda()

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            #x[i] = self.m[i](x[i])  # conv
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid



def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    if hard:  # assert min requirements met
        assert result, f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'
    else:
        return result

def main():
    backbone = CSPDarkNet()
    neck = YoloNeck()
    head = Detect()
    data = torch.ones((2, 3, 512, 512))
    x = backbone(data)
    print(len(x))
    x = neck(x)
    print(len(x))
    print(x[0].shape, x[1].shape, x[2].shape)
    x = head(x)
    print(len(x))
    print(x[0].shape, x[1].shape, x[2].shape)

    
if __name__ == '__main__':
    main()