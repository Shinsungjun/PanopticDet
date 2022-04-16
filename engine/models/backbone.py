from copy import deepcopy
from operator import mod
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import BaseConv, C3, SPPF


class CSPDarkNet(nn.Module):
    def __init__(self, size = 'x'):
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
            raise ValueError('choose backbone model size m OR l OR x')
        
        #! -- stem (1/2) -- #
        in_channels = 3
        out_channels = int(round(64 * width_multiple)) # m : 48 l : 64 x : 80
        self.stem = nn.Sequential(
            BaseConv(in_channels=in_channels, out_channels=out_channels, ksize=6, stride=2, padding=2)
        )

        #! -- 1/4 -- #
        in_channels = out_channels
        out_channels = int(round(128 * width_multiple)) # m : 96 l : 128 x : 160
        module_num = int(round(3 * depth_multiple)) # m : 2 l : 3 x : 4
        self.B4 = nn.Sequential(
            BaseConv(in_channels=in_channels, out_channels=out_channels, ksize=3, stride=2),
            C3(in_channels=out_channels, out_channels=out_channels, n=module_num)
        )

        #! -- 1/8 -- #
        in_channels = out_channels
        out_channels = int(round(256 * width_multiple)) # m : 192 l : 256 x : 320
        module_num = int(round(6 * depth_multiple)) # m : 6 l : 9 x : 12

        self.B8 = nn.Sequential(
            BaseConv(in_channels=in_channels, out_channels=out_channels, ksize=3, stride=2),
            C3(in_channels=out_channels, out_channels=out_channels, n = module_num)
        )
        
        self.B8_conv = BaseConv(in_channels=in_channels, out_channels=out_channels, ksize=3, stride=2)
        self.B8_C3 = C3(in_channels=out_channels, out_channels=out_channels, n = module_num)

        #! -- 1/16 -- #
        in_channels = out_channels
        out_channels = int(round(512 * width_multiple)) #m : 384 l : 512 x : 640
        module_num = int(round(9 * depth_multiple)) # m : 6 l : 9 x : 12
        self.B16 = nn.Sequential(
            BaseConv(in_channels=in_channels, out_channels=out_channels, ksize=3, stride=2),
            C3(in_channels=out_channels, out_channels = out_channels, n = module_num)
        )
        
        #! -- 1/32 -- #
        in_channels = out_channels
        out_channels = int(round(1024 * width_multiple)) #m : 768 l : 1024 x : 1280
        module_num = int(round(3 * depth_multiple)) # m : 2 l : 3 x : 4
        self.B32 = nn.Sequential(
            BaseConv(in_channels=in_channels, out_channels=out_channels, ksize=3, stride=2),
            C3(in_channels = out_channels, out_channels = out_channels, n = module_num)
        )
        
        in_channels = out_channels
        out_channels = int(round(1024 * width_multiple)) #m : 768 l : 1024 x : 1280
        self.SPPF = nn.Sequential(
            SPPF(in_channels=in_channels, out_channels=out_channels, k = 5) #! input size에 따라 6x6이 의미가 있는가 ?!
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self,x):
        
        feats = []

        f2 = self.stem(x)

        f4 = self.B4(f2) #1/4
        b,c, h,w = f4.shape
        f8 = self.B8(f4) #1/8
        feats.append(f8)

        f8 = self.B8_conv(f4)
        f8 = self.B8_C3(f8)

        f16 = self.B16(f8) #1/16
        feats.append(f16)

        f32 = self.B32(f16) #1/32
        feats.append(f32)
        
        spp = self.SPPF(f32)
        feats.append(spp)
        
        return feats, (h,w)

def main():
    backbone = CSPDarkNet()
    data = torch.ones((5, 3, 512, 512))
    x = backbone(data)

if __name__ == '__main__':
    main()