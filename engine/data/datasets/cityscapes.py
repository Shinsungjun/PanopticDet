import os
from loguru import logger

import cv2
import numpy as np



import glob

import torch
from torch.utils import data
import torch.distributed as dist

from PIL import Image
import torch.nn.functional as F

'''
data loader for cityscapes dataset with panoptic annotation

'''
class Cityscapes(data.Dataset):
    def __init__(self, data_root, split='train', ignore_label=255, transforms=False, cache=False):
        '''
        path:
            Image: /cityscapes/leftImg8bits/train/
                    bochum/bochum_000000_000600_leftImg8bit.png

            ground truth: /cityscapes/gtFine_trainvaltest/gtFine/train/
                            bochum/bochum_000000_000313_gtFine_color.png

            root: ~/workspace/dataset/cityscapes
        '''
        self.root = data_root
        self.split = split
        self.image_base = os.path.join(self.root, 'leftImg8bit',self.split)
        self.files = glob.glob(self.image_base+'/*/*.png')
        assert len(self.files) is not 0 , ' cannot find datas!{}'.format(self.image_base)
        
        self.transforms = transforms

        self.nClasses = 19
        self.imgs = None


        self.label_mapping = {-1: ignore_label, 0: ignore_label, 
                              1: ignore_label, 2: ignore_label, 
                              3: ignore_label, 4: ignore_label, 
                              5: ignore_label, 6: ignore_label, 
                              7: 0, 8: 1,
                              9: ignore_label, 10: ignore_label,
                              11: 2, 12: 3,
                              13: 4, 14: ignore_label, 
                              15: ignore_label, 16: ignore_label, 
                              17: 5, 18: ignore_label, 
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 
                              24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15, 
                              29: ignore_label, 30: ignore_label, 
                              31: 16, 32: 17, 33: 18}
        self.class_names = ['road', 'sidewalk', 'building', 'wall', 'fence',\
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',\
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle','unlabelled'] # 6,5,7,2

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img, label, img_path = self.pull_item(index)

        if self.transforms is not None:
            img, label= self.transforms(img,label)
        label = self.convert_label(label)

        return img, label

    def pull_item(self, index):

        img_path = self.files[index] # .../cityscapes/leftImg8bit/train/00000.jpg
        right_img_path = img_path.replace('leftImg8bit', 'rightImg8bit', 1)
        gt_path = img_path.replace('leftImg8bit','gtFine',1)
        # gt_path = gt_path.replace('leftImg8bit','gtFine_labelTrainIds',1)
        gt_path = gt_path.replace('leftImg8bit','gtFine_labelIds',1)
        label = Image.open(gt_path)
        label = np.asarray(label)
    
        if self.imgs is not None:
            pad_img = self.imgs[index]
        else:
            img = Image.open(img_path)
            img = np.asarray(img)

        return img, label.copy(), img_path


    def convert_label(self, label, inverse=False):
        temp = label.clone()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label
    
    def get_stereo_T(self, do_flip, side):
        stereo_T = np.eye(4, dtype=np.float32)
        baseline_sign = -1 if do_flip else 1
        side_sign = -1 if side == "l" else 1
        stereo_T[0, 3] = side_sign * baseline_sign * 0.1
        
        return stereo_T
            
    def get_file_path(self):
        return self.files

