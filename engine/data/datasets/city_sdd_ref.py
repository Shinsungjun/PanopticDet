'''
class mapping
origin_id  |   seg_id    |  class_name    |  detection_id
    25          12:         rider               1 -> 0
    26          13:         car                 2 -> 1
    33          18:         bicycle             3 -> 2
    24          11:         person              4 -> 3
    28          15:         bus                 5 -> 4
    32          17:         motorcycle          6 -> 5
    27          14:         truck               7 -> 6
    31          16:         train               8 -> 7
    07          00:         road                   08
    08          01:         sidewalk               09
    11          02:         building               10
    12          03:         wall                   11
    13          04:         fence                  12
    17          05:         pole                   13
    19          06:         traffic_light          14
    20          07:         traffic_sign           15
    21          08:         vegetation             16
    22          09:         terrain                17
    23          10:         sky                    18

'''

import os
import json
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils import data

from PIL import Image

# from .augmentation import DataAugmentor
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

class Cityscapes_test(data.Dataset):
    def __init__(self, data_root, split='train', ignore_label=255, transforms=False, cache=False):
        '''
            root: cityscapes/gtFine_trainvaltest/
            annotation: root/annotations/instancesonly_filtered_gtFine_val.json
            Image: root/leftImg8bits/...city_name/...image_id_leftImg8bit.png
            ground truch: root/gtFine/train/...city_name/...image_id_gtFine_color.png
        '''
        #% cfg
        self.root = data_root
        self.split = split
        
        self.imgRoot = os.path.join(self.root, 'leftImg8bit', self.split)
        self.segRoot = os.path.join(self.root, 'gtFine', self.split)
        
        self.anno_path = os.path.join(self.root,
                                 'annotations/instancesonly_filtered_gtFine_' + \
                                  self.split+'.json')

        with open(os.path.join(self.anno_path), 'r') as j:
            files = json.load(j)
            # self.images = files['images']
            temp = files['annotations']
        self.anno = [item for item in temp if item['box_info']]

        self.transforms = transforms

        self.nClasses = 19

        self.label_mapping = {-1: ignore_label, 0: ignore_label, 
                              1: ignore_label, 2: ignore_label, 
                              3: ignore_label, 4: ignore_label, 
                              5: ignore_label, 6: ignore_label, 
                              7: 8, 8: 9,
                              9: ignore_label, 10: ignore_label,
                              11: 10, 12: 11,
                              13: 12, 14: ignore_label, 
                              15: ignore_label, 16: ignore_label, 
                              17: 13, 18: ignore_label, 
                              19: 14, 20: 15, 21: 16, 22: 17, 23: 18, 
                              24: 3,
                              25: 0, 26: 1, 27: 6, 28: 4, 
                              29: ignore_label, 30: ignore_label, 
                              31: 7, 32: 5, 33: 2}
        self.class_names = ['road', 'sidewalk', 'building', 'wall', 'fence',\
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',\
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle','unlabelled'] # 5,5,6,3


    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        name = self.anno[index]['image_id']
        imgPath = os.path.join(self.imgRoot, name.split('_')[0],
                                name+'_leftImg8bit.png')

        rightimgPath = os.path.join(self.imgRoot, name.split('_')[0],
                                name+'_rightImg8bit.png')

        images = np.asarray(Image.open(imgPath, mode='r'))
        rightimages = np.asarray(Image.open(rightimgPath, mode='r'))

        segPath = os.path.join(self.segRoot, name.split('_')[0],
                                name+'_gtFine_labelIds.png')
        seg = np.asarray(Image.open(segPath))

        box_info = self.anno[index]['box_info']
        bbox = []
        classId = []
        for boxData in box_info:
            bbox.append(boxData['bbox'])
            classId.append(boxData['category_id'])
        boxes = torch.as_tensor(bbox, dtype=torch.float32).reshape(-1, 4)
        boxes = box_cxcywh_to_xyxy(boxes)
        classIds = torch.tensor(classId, dtype=torch.int64)

        det = {'boxes': boxes,
                'labels': classIds,}

        # while True:
        #     tf_img, tf_seg, tf_target = self._transforms(img, seg, target)
        # if len(tf_target['boxes']) != 0:
        #     break
        # else:
        #     print(data_dict['path']) 
        tf_img, tf_rightimg, tf_seg, tf_target = self.transforms(images, rightimages, seg, det)

        inputs = {
            'left': tf_img,
            'right': tf_rightimg
        }

        targets = {
            'seg':  self.convert_label(tf_seg),
            'classIds': tf_target['labels']-1,
            'boxes': tf_target['boxes'] }

        return inputs, targets

    def convert_label(self, label, inverse=False):
        temp = label.clone()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label
        
    def get_file_path(self):
        return self.files



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.colors as clr
    import matplotlib.patches as patches

    data = Cityscapes('nono')

    inputs = data.__getitem__(3)


    # print(img)
    print(inputs)
    print(inputs['images'].shape)
    print(inputs['boxes'])

 

    fig_in = plt.figure()
    ax = fig_in.add_subplot(1,2,1)
    ax.imshow(inputs['images'])
    for box in inputs['boxes']:
        x,y,x2,y2 = box
        w = x2-x
        h = y2-y
        rect = patches.Rectangle((x,y),w, h, linewidth=1, edgecolor='r', facecolor='none') # x,y,w,h
        ax.add_patch(rect)       

    ax = fig_in.add_subplot(1,2,2)
    ax.imshow(inputs['seg'])
    plt.show()