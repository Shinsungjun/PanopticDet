import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from engine.utils.torch_utils import model_info
# from .modules import YoloSegTower, SegHead, BaseConv, OnlyConv, KernelHead
# from .ctx import Ctxnet
# from .head import Head

class Fullmodel(nn.Module):
    '''
        Fullmodel
    '''
    def __init__(self, backbone=None, ob_neck = None, ob_head=None, depthneck = None, depthhead=None):
        super().__init__()

        #*YOLOv5
        self.backbone = backbone
        self.ob_neck = ob_neck
        self.ob_head = ob_head
        
        #*Seg
        # self.aux_seg_tower = YoloSegTower(kdim=(320, 640, 1280, 1280), midim=320, outdim=320)
        # self.aux_seg_head = SegHead(320, 81)

        # self.seg_tower = YoloSegTower(kdim=(320, 640, 1280), midim=320, outdim=320)
        # self.seg_head = SegHead(320, 81)

        #*Instance Seg
        # self.mask_head = nn.Sequential(
        #                     BaseConv(in_channels=320, out_channels=320, ksize=3, stride=1, padding=1),
        #                     OnlyConv(in_channels=320, out_channels=64, ksize=1, stride=1)
        #                 )

        # self.kernel_head = KernelHead(kdim=(320,640,1280), outdim = 64)

        # Build strides, anchors
        m = self.ob_head  # Detect()
    
        s = 256  # 2x min stride
        m.inplace = True
        ch = 3

        """
        m.stride tensor([ 8., 16., 32.]) 3
        m.anchors tensor([[[ 10.,  13.],
         [ 16.,  30.],
         [ 33.,  23.]],

        [[ 30.,  61.],
         [ 62.,  45.],
         [ 59., 119.]],

        [[116.,  90.],
         [156., 198.],
         [373., 326.]]])"""
        # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(2, ch, s, s))[0]])  # forward
        m.stride = torch.tensor([ 8., 16., 32.])
        # print("m.anchors", m.anchors)
        """
            m.anchors tensor([[[ 10.,  13.],
         [ 16.,  30.],
         [ 33.,  23.]],

        [[ 30.,  61.],
         [ 62.,  45.],
         [ 59., 119.]],

        [[116.,  90.],
         [156., 198.],
         [373., 326.]]])

            """
            
        m.anchors /= m.stride.view(-1, 1, 1)
        # print("m.anchors", m.anchors)
        """
            m.anchors tensor([[[ 1.25000,  1.62500],
         [ 2.00000,  3.75000],
         [ 4.12500,  2.87500]],

        [[ 1.87500,  3.81250],
         [ 3.87500,  2.81250],
         [ 3.68750,  7.43750]],

        [[ 3.62500,  2.81250],
         [ 4.87500,  6.18750],
         [11.65625, 10.18750]]])
            """

        # check_anchor_order(m)
        self.stride = m.stride
        # self._initialize_biases()  # only run once
        self.info()

    def forward(self, images, info=None):
        '''
        images : [B, C, H, W] input image
        output : Dict
            'aux_seg' : [B, 81, H/8, W/8] = aux seg for auxiliary segmentation loss Use only training
            'box_pred' :
                training : list of tensors [B, anchor_num, H, W, 85 (80 + 4 + 1)]
                inference : 
            'kernel_pred' : list of tensors [B, anchor_num, H, W, 64]
            'mask_pred' : tensor [B, 64, H/4, W/4]
            'seg_pred' : tensor [B, 81, H/8, W/8]
        '''
        seg = []
        output = {}
        feats = self.backbone(images) #[1/8, 1/16, 1/32, spp]

        aux_seg_feats = self.aux_seg_tower(feats)

        if self.training:
            aux_seg_pred = self.aux_seg_head(aux_seg_feats)
            output['aux_seg_pred'] = aux_seg_pred

        feats_to_head = self.ob_neck(feats) #[1/8, 1/16, 1/32]
        seg_feats = self.seg_tower(feats_to_head)

        # if self.training:
        box_pred = self.ob_head(feats_to_head)
            
        kernel_pred = self.kernel_head(feats_to_head)
        mask_pred = self.mask_head(F.interpolate(seg_feats, feat_4_size,mode='bilinear',align_corners=True))
        seg_pred = self.seg_head(seg_feats)

        output['box_pred'] = box_pred #
        output['kernel_pred'] = kernel_pred #[B, N, 64]
        output['mask_pred'] = mask_pred #[B, 64, H/4, W/4]
        output['seg_pred'] = seg_pred #

        return output, seg

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

