from copy import deepcopy
from operator import mod
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import BaseConv, C3, SPPF

class YoloNeck(nn.Module):
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
            raise ValueError('choose neck model size m OR l OR x')
        
        self.up_sample = nn.Upsample(None, 2, 'nearest')

        #! -- up resolution (32 -> 16 -> 8) -- #
        in_channels = int(round(1024 * width_multiple))
        out_channels = int(round(512 * width_multiple))
        self.conv_spp = BaseConv(in_channels= in_channels, out_channels=out_channels , ksize=1, stride=1)

        in_channels = out_channels * 2 #after concat
        out_channels = int(round(512 * width_multiple))
        module_num = int(round(3 * depth_multiple))
        self.up_c3_16cat = C3(in_channels=in_channels, out_channels=out_channels, n = module_num, shortcut=False)

        in_channels = out_channels
        out_channels = int(round(256 * width_multiple))
        self.up_conv_16c3 = BaseConv(in_channels=in_channels, out_channels = out_channels, ksize=1, stride=1)

        in_channels = out_channels * 2 #after concat [backbone8, up16, logit8]
        out_channels = int(round(256 * width_multiple))
        module_num = int(round(3 * depth_multiple))
        self.c3_8cat = C3(in_channels=in_channels, out_channels=out_channels, n = module_num, shortcut=False)
        #feats to head 1/8

        #! -- down resolution (8 -> 16 -> 32) -- #
        in_channels = out_channels
        out_channels = int(round(256 * width_multiple))
        module_num = int(round(3 * depth_multiple))
        self.down_conv_8c3 = BaseConv(in_channels=in_channels, out_channels = out_channels, ksize=3, stride=2)
        #1/8 -> 1/16 conv

        in_channels = out_channels * 2 #[down8, f16, logit16]
        out_channels = int(round(512 * width_multiple))
        module_num = int(round(3 * depth_multiple))
        self.down_c3_16cat = C3(in_channels=in_channels, out_channels=out_channels, n = module_num, shortcut=False)

        in_channels = out_channels
        out_channels = int(round(512 * width_multiple))
        module_num = int(round(3 * depth_multiple))
        self.down_conv_16c3 = BaseConv(in_channels=in_channels, out_channels = out_channels, ksize=3, stride=2)
        #1/16 -> 1/32 conv

        in_channels = out_channels * 2 #[down16, f32, logit32]
        out_channels = int(round(1024 * width_multiple))
        module_num = int(round(3 * depth_multiple))
        self.down_c3_32cat = C3(in_channels=in_channels, out_channels=out_channels, n = module_num, shortcut=False)


        # self.logit_to_feat8 = BaseConv(in_channels=320, out_channels=320, ksize=1, stride=1)
        # self.logit_to_feat16 = BaseConv(in_channels=320, out_channels=320, ksize=3, stride=2, padding=1)
        # self.logit_to_feat32 = BaseConv(in_channels=320, out_channels=320, ksize=3, stride=2, padding=1)
    def forward(self, feats):
        '''
        inputs:
            feats = list of tensors [1/8, 1/16, 1/32, spp]
            logits = tensor [B, 320, H/8, W/8]

        output:
            feats_to_head = list of tensors [1/8, 1/16, 1/32]
        '''
        feats_to_head = []
        feats8 = feats[0]
        feats16 = feats[1]
        spp = feats[3]

        f_32 = self.conv_spp(spp)

        f_16 = torch.cat([feats16, self.up_sample(f_32)], dim=1)
        f_16 = self.up_c3_16cat(f_16)
        f_16 = self.up_conv_16c3(f_16)
        f_8 = torch.cat([feats8, self.up_sample(f_16)], dim=1)
        f_8_to_head = self.c3_8cat(f_8)
        feats_to_head.append(f_8_to_head)

        f_8_to_16 = self.down_conv_8c3(f_8_to_head) 

        f_16 = torch.cat([f_16, f_8_to_16], dim = 1)
        f_16_to_head = self.down_c3_16cat(f_16)
        feats_to_head.append(f_16_to_head)

        f_16_to_32 = self.down_conv_16c3(f_16_to_head)

        f_32 = torch.cat([f_32, f_16_to_32], dim=1)
        f_32_to_head = self.down_c3_32cat(f_32)
        feats_to_head.append(f_32_to_head)
        
        return feats_to_head


def main():
    backbone = CSPDarkNet()
    neck = YoloNeck()
    data = torch.ones((2, 3, 512, 512))
    x = backbone(data)
    print(len(x))
    x = neck(x)
    print(len(x))
    print(x[0].shape, x[1].shape, x[2].shape)

if __name__ == '__main__':
    main()