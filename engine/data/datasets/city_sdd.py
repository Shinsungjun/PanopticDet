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


from pycocotools.coco import COCO
import cv2


class Cityscapes(data.Dataset):
    def __init__(self, data_root, split='train', ignore_label=255,
            transforms=False, cache=False, preproc=None, img_size=None):
        '''
        '''
        self.root = data_root # "/usr/src/EXP_template/cityscapes"
        self.split = split
        self.json =  data_root + '/annotations/cityscapes_to_coco_' + split + '.json'

        self.city = COCO(self.json)
        self.ids = self.city.getImgIds()
        self.class_ids = sorted(self.city.getCatIds())  # [1, 2, 3, 4, 5, 6, 7, 8]
        cats = self.city.loadCats(self.city.getCatIds()) 
        self._classes = tuple([c["name"] for c in cats]) # ('person', 'rider', 'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'train')

        self.img_size = img_size
        self.imgs = None
        # returns: annotation dict, img h,w, resized h,w, file_path(left)
        self.annotations = [self.load_anno_from_ids(_ids) for _ids in self.ids]

        #TODO: add cache fuction
        if cache:
            self._cache_images()

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
                            'motorcycle', 'bicycle','unlabelled'] # 5,5,6,3 #TODO: change sequence


        #! 1/4에서 predict하게 하면 2048, 1024로 나누는게 아니고 (0.5도 마찬가지) 1/4사이즈로 나눠야하나?
        self.K = np.array([[2262.52 / 2048, 0, 0.5, 0],
                            [0, 1096.98 / 1024, 0.5, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32)

    def load_anno_from_ids(self, id_):
        im_ann = self.city.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.city.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.city.loadAnns(anno_ids)
        
        objs = []
        #TODO: check annotation. 아래는 annotation이 x1,y1,w,h로 되어있는 경우를 가정하고 작성한 것.
        for obj in annotations: 
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["boxes"] = [x1, y1, x2, y2] #TODO: dict 키가 clean box인지, bbox인지 확인.
                objs.append(obj)

        num_objs = len(objs)

        #% yolo style [n,5] type
        # res = np.zeros((num_objs, 5))
        # for ix, obj in enumerate(objs):
        #     cls = self.class_ids.index(obj["category_id"])
        #     res[ix, 0:4] = obj["boxes"] #TODO: dict 키가 clean box인지, bbox인지 확인.
        #     res[ix, 4] = cls

        bbox, labels, area = list(), list(), list()
        for obj in objs:
            bbox.append(obj['bbox'])
            labels.append(obj['category_id'])
            area.append(obj['area'])
        res = {'boxes': torch.as_tensor(bbox, dtype=torch.float32).reshape(-1, 4),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'area': torch.tensor(area, dtype=torch.float32),}

        img_info = (height, width)
        resized_info = None
        if self.img_size is not None:
            r = min(self.img_size[0] / height, self.img_size[1] / width)
            res['boxes'] *= r
            resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 136G available disk space for training COCO.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = self.root + "/img_resized_cache_" + self.name + ".array"
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 20 minutes for COCO"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )


    def __len__(self):
        return len(self.ids)

    def load_image(self, index, is_right=False):
        '''load left image with index'''
        file_path = self.annotations[index][3]

        img_file = os.path.join(self.root, file_path)

        if is_right:
            right_img_file = file_path.replace('leftImg8bit', 'rightImg8bit')
            img_file = os.path.join(self.root, right_img_file)

        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.open(img_file)
        # img = np.asarray(img)
        assert img is not None, "no images"
        return img

    def load_resized_img(self, index):
        assert self.img_size is not None
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_seg(self, index, is_right=False):
        file_path = self.annotations[index][3]

        keyword = 'leftImg8bit'
        if is_right:
            keyword = 'rightImg8bit'

        seg_file = os.path.join(self.root, file_path)
        seg_path = seg_file.replace(keyword,'gtFine',1)
        seg_path = seg_path.replace(keyword,'gtFine_labelIds',1)

        seg = cv2.imread(seg_path,cv2.IMREAD_GRAYSCALE)                           
        # seg = Image.open(seg_file)
        # seg = np.asarray(seg)
        # assert img is not None
        assert seg is not None, "no seg label"
        return seg

    def pull_item(self, index):
        id_ = self.ids[index]

        #% load annotation:
        res, img_info, resized_info, _ = self.annotations[index]

        #% load images:
        if self.imgs is not None:  # this is for cached images #TODO: not working yet
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            #TODO: resized img
            limg = self.load_image(index) 
            rimg = self.load_image(index,is_right=True)

        #% load segmentation annotation:
        lseg = self.load_seg(index)
        
        return limg, rimg, lseg.copy(), res.copy(), img_info, np.array([id_])

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

    def get_stereo_T(self, do_flip, side):
        stereo_T = np.eye(4, dtype=np.float32)
        baseline_sign = -1 if do_flip else 1
        side_sign = -1 if side == "l" else 1
        stereo_T[0, 3] = side_sign * baseline_sign * 0.1
        
        return stereo_T

    def __getitem__(self, index):

        limg, rimg, lseg, ldet, img_info, img_id = self.pull_item(index)

        if self.transforms is not None:
            limg_tf, rimg_tf, lseg_tf, ldet_tf = self.transforms(limg, rimg, lseg, ldet)

        lseg_tf = self.convert_label(lseg_tf)
        K = self.K.copy()
        inv_K = np.linalg.pinv(K)
        stereo_T = self.get_stereo_T(do_flip= False, side = 'l')
        targets = {
            'limg': torch.from_numpy(limg).float().permute(2,0,1), 
            'rimg': torch.from_numpy(rimg).float().permute(2,0,1),
            'rimg_tf': rimg_tf,
            'lseg': lseg_tf,
            'ldet': ldet_tf,
            # 'img_size': img_info,
            # 'img_id': img_id,
            'K': torch.from_numpy(K).float(),
            'inv_K': torch.from_numpy(inv_K).float(),
            'stereo_T' : torch.from_numpy(stereo_T).float()
        }

        return limg_tf, targets


def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
    This describes how to combine these tensors of different sizes. We use lists.
    :params 
        batch: an iterable of N sets from __getitem__() // list of N  from __getitem__()
                list of N samples ( tuples( getitem[0], getitem[1], ...) )

    :return: 
        a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """
    batch = list(zip(*batch))
    batch[0] = torch.stack(batch[0], dim=0)
    # batch[1] = [b for b in batch[1]]

    
    return tuple(batch)


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