#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

import numpy as np

import torch

from engine.data.datasets import COCO_CLASSES
from engine.utils.metrics import ap_per_class, ConfusionMatrix, box_iou
from engine.utils.general import coco80_to_coco91_class, non_max_suppression, xyxy2xywh, xywh2xyxy, scale_coords, non_max_suppression2, crop
from engine.utils.torch_utils import time_sync
from engine.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')


class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    Note: I type annotated this because why not.
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score:float, is_true:bool):
        self.data_points.append((score, is_true))
    
    def add_gt_positives(self, num_positives:int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls    = []
        num_true  = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]: num_true += 1
            else: num_false += 1
            
            precision = num_true / (num_true + num_false)
            recall    = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions)-1, 0, -1):
            if precisions[i] > precisions[i-1]:
                precisions[i-1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101 # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)

class AverageMeter(object):
    """
        code is from pytorch imagenet examples
        Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # print(val, n)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def per_class_mAP_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
    per_class_mAP = {}
    precisions = coco_eval.eval["precision"]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_mAP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_mAP) * len(headers))
    result_pair = [x for pair in per_class_mAP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    '''
        'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.

        params:
            output: BxHxW size tensor filled with predicted class labels.
            target: BxHxW size tensor filled with ground truth class labels.
            K:      number of classes

        returns:
            area of intersection
            area of union
            area of target
    '''
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape

    output = output.view(2,-1)
    target = target.view(2,-1)

    output[target == ignore_index] = ignore_index

    # mask of intersection where predict==target
    intersection = output[output == target] 
    
    # compute histogram of tensor. shape: [19]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1) 
    area_output = torch.histc(output, bins=K, min=0, max=K-1) 
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection

    return area_intersection, area_union, area_target

def calc_map(ap_data):
    print('Calculating mAP...')
    aps = [{'mask': []} for _ in iou_thresholds]
    #aps = [{'box': []} for _ in iou_thresholds]

    for _class in range(len(COCO_CLASSES)):
        for iou_idx in range(len(iou_thresholds)):
            # for iou_type in ('box', 'mask'):
            ap_obj = ap_data['mask'][iou_idx][_class]

            if not ap_obj.is_empty():
                aps[iou_idx]['mask'].append(ap_obj.get_ap())

    all_maps = {'mask': OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    # for iou_type in ('box', 'mask'):
    all_maps['mask']['all'] = 0 # Make this first in the ordereddict
    for i, threshold in enumerate(iou_thresholds):
        mAP = sum(aps[i]['mask']) / len(aps[i]['mask']) * 100 if len(aps[i]['mask']) > 0 else 0
        all_maps['mask'][int(threshold*100)] = mAP
    all_maps['mask']['all'] = (sum(all_maps['mask'].values()) / (len(all_maps['mask'].values())-1))
    
    print_maps(all_maps)
    
    # Put in a prettier format so we can serialize it to json during training
    all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
    return all_maps

def print_maps(all_maps):
    # Warning: hacky 
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n:  ('-------+' * n)
    print()
    print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['mask'].keys()]))
    print(make_sep(len(all_maps['mask']) + 1))
    # for iou_type in ('box', 'mask'):
    for iou_type in (['mask']):
        print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['mask']) + 1))
    print()


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct
def _mask_iou(mask1, mask2, iscrowd=False):
    ret = mask_iou(mask1, mask2, iscrowd)
    return ret.cpu()

def mask_iou(masks_a, masks_b, iscrowd=False):
    """
    Computes the pariwise mask IoU between two sets of masks of size [a, h, w] and [b, h, w].
    The output is of size [a, b].

    Wait I thought this was "box_utils", why am I putting this in here?
    """

    masks_a = masks_a.view(masks_a.size(0), -1)
    masks_b = masks_b.view(masks_b.size(0), -1)

    intersection = masks_a @ masks_b.t()
    area_a = masks_a.sum(dim=1).unsqueeze(1)
    area_b = masks_b.sum(dim=1).unsqueeze(0)

    return intersection / (area_a + area_b - intersection) if not iscrowd else intersection / area_a

iou_thresholds = [x / 100 for x in range(50, 100, 5)]

class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        testdev: bool = False,
        per_class_mAP: bool = False,):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_mAP: Show per class mAP during evalution or not. Default to False.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_mAP = per_class_mAP

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        compute_loss = False,
        LOGGER = None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        training = model is not None
        if training:  # called by train.py
            device = next(model.parameters()).device  # get model device

        # else:  # called directly
        #     device = select_device(device, batch_size=batch_size)

        #     # Directories
        #     save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        #     (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        #     # Load model
        #     check_suffix(weights, '.pt')
        #     model = attempt_load(weights, map_location=device)  # load FP32 model
        #     gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        #     imgsz = check_img_size(imgsz, s=gs)  # check image size

        #     # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        #     # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     #     model = nn.DataParallel(model)

        #     # Data
        #     data = check_dataset(data)  # check

        # Half
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()

        # Configure
        model.eval()
        is_coco = True  # COCO dataset
        # is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
        nc = 80  # number of classes
        # nc = 1 if single_cls else int(data['nc'])  # number of classes
        iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()

        # Dataloader
        # if not training:
        #     if device.type != 'cpu':
        #         model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        #     pad = 0.0 if task == 'speed' else 0.5
        #     task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        #     dataloader = create_dataloader(data[task], imgsz, batch_size, gs, single_cls, pad=pad, rect=True,
        #                                 prefix=colorstr(f'{task}: '))[0]

        seen = 0
        confusion_matrix = ConfusionMatrix(nc=nc)
        #names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
        class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
        s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        #! add seg loss -> torch.zeros(3, device = device) -> torch.zeros(4, device = device)
        loss = torch.zeros(4, device=device)
        jdict, stats, ap, ap_class = [], [], [], []


        intersection_meter = AverageMeter('IoU')
        union_meter = AverageMeter('Union')
        target_meter = AverageMeter('Target')
        ap_data = {
                    'mask': [[APDataObject() for _ in COCO_CLASSES] for _ in iou_thresholds]
                }
        for batch_i, (img, segs, ins_masks, targets, paths, shapes) in enumerate(tqdm(self.dataloader, desc=s)):

            if batch_i > 50:
                break

            segs = segs.to(device)
            t1 = time_sync()
            img = img.to(device, non_blocking=True)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255  # 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)
            for i in range(len(ins_masks)):
                ins_masks[i] = ins_masks[i].to(device)

            nb, _, height, width = img.shape  # batch size, channels, height, width
            t2 = time_sync()
            dt[0] += t2 - t1

            # Run model
            #! add segmentation
            output = model(img)  # inference and training outputs
            out = output['box_pred'][0]

            # print("out shape : ", out.shape)
            # print("out shape : ", out[0][0:5])
            pred_seg = output['seg_pred']
            kernel_pred = output['kernel_pred']
            mask_pred = output['mask_pred'].permute(0, 2,3, 1)
            dt[1] += time_sync() - t2


            # Compute loss
            if compute_loss:
                #! add segmentation
                loss += compute_loss([[x.float() for x in output['box_pred'][1]], [y.float() for y in pred_seg]], targets, segs)[1]  # box, obj, cls
            #! add segmentation metrics
            # intersection, union, target = intersectionAndUnionGPU(pred_seg[-1].max(1)[1], segs, 81, 255)
            intersection, union, target = intersectionAndUnionGPU(pred_seg.max(1)[1], segs, 81, 255)
            # intersection, union, target = intersectionAndUnionGPU(pred_seg[-1].max(1)[1], segs, 80, 80)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            # lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            lb = []
            t3 = time_sync()
            # out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
            out, kernel_out = non_max_suppression2(out, kernel_pred, 0.001, 0.6, labels=lb, multi_label=True, agnostic=False)
            dt[2] += time_sync() - t3

            #print('kernel_out len : ', len(kernel_out)) #([300, 64])
            # if isinstance(kernel_out, list):
            #     for t in kernel_out:
            #         print(t.shape)
            #print("mask shape :", mask_pred.shape) # torch.Size([1, 64, 48, 168])

            #print('box shape : ', out[0].shape)
            #print('box : ', out[0][0:5, :])
            

            #! box -> full size image (not native space)
            #! mask -> 1/4
            #! crop -> relative point-form box (change 1/4 box in crop method) 

            # Statistics per image
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                path, shape = Path(paths[si]), shapes[si][0]
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                # if single_cls:
                #     pred[:, 5] = 0
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
                #print("pred box : ", predn[:5, :4])
                #print("shape[0] : ", shape[0])
                #print("shape[1] : ", shape[1])
                crop_box = predn[:, :4].clone()
                crop_box[:, :2] = crop_box[:, :2] / shape[0]
                crop_box[:, 2:4] = crop_box[:, 2:4] / shape[1]
                #print("crop_box[5, :4]", crop_box[:5, :4])
                
                masks = mask_pred[si] @ kernel_out[si].t()

                masks = masks.sigmoid()

                masks = crop(masks, crop_box)

                masks = masks.permute(2, 0, 1).contiguous()

                # Binarize the masks
                masks.gt_(0.5)
                #print("masks2", masks.shape) #torch.Size([300, 48, 168])
                #print("ins_masks[0]", ins_masks[0].shape)
                # Evaluate
                if nl:
                    c, h, w = masks.shape
                    masks = masks.view(-1, h*w)
                    scores = pred[:, 4]
                    scores = list(scores.cpu().numpy().astype(float))
                    mask_scores = scores
                    box_scores = scores
                    gt_masks = ins_masks[si].view(-1, h*w).float()
                    mask_iou_cache = _mask_iou(masks, gt_masks)
                    num_pred = len(pred)
                    box_indices = sorted(range(num_pred), key=lambda i: -box_scores[i])
                    mask_indices = sorted(box_indices, key=lambda i: -mask_scores[i])
                    crowd_mask_iou_cache = None
                    classes = list(pred[:, 5].cpu().numpy().astype(int))
                    gt_classes = list(targets[:, 1].to(int))
                    num_gt = len(gt_classes)
                    iou_types = [
                    ('mask', lambda i,j: mask_iou_cache[i, j].item(),
                            lambda i,j: crowd_mask_iou_cache[i,j].item(),
                            lambda i: mask_scores[i], mask_indices)
                    ]

                    for _class in set(classes + gt_classes):
                        ap_per_iou = []
                        num_gt_for_class = sum([1 for x in gt_classes if x == _class])
                        
                        for iouIdx in range(len(iou_thresholds)):
                            iou_threshold = iou_thresholds[iouIdx]

                            for iou_type, iou_func, crowd_func, score_func, indices in iou_types:
                                gt_used = [False] * len(gt_classes)
                                
                                ap_obj = ap_data[iou_type][iouIdx][_class]
                                ap_obj.add_gt_positives(num_gt_for_class)

                                for i in indices:
                                    if classes[i] != _class:
                                        continue
                                    
                                    max_iou_found = iou_threshold
                                    max_match_idx = -1
                                    for j in range(num_gt):
                                        if gt_used[j] or gt_classes[j] != _class:
                                            continue
                                            
                                        iou = iou_func(i, j)

                                        if iou > max_iou_found:
                                            max_iou_found = iou
                                            max_match_idx = j
                                    
                                    if max_match_idx >= 0:
                                        gt_used[max_match_idx] = True
                                        ap_obj.push(score_func(i), True)
                                    else:
                                        # If the detection matches a crowd, we can just ignore it
                                        matched_crowd = False

                                        # All this crowd code so that we can make sure that our eval code gives the
                                        # same result as COCOEval. There aren't even that many crowd annotations to
                                        # begin with, but accuracy is of the utmost importance.
                                        if not matched_crowd:
                                            ap_obj.push(score_func(i), False)



                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    correct = process_batch(predn, labelsn, iouv)
                    # if plots:
                    #     confusion_matrix.process_batch(predn, labelsn)
                else:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

                # Save/log
                # if save_txt:
                #     save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
                # if save_json:
                #     save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
                # callbacks.run('on_val_image_end', pred, predn, path, names, img[si])

            # Plot images
            # if plots and batch_i < 3:
            #     f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            #     Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            #     f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            #     Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

        #! Compute statistics for segmentation
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        #! instance masks
        all_map = calc_map(ap_data)
        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        # pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
        # LOGGER.info("               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95")
        # LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
        # LOGGER.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))


        # Print results per class
        # if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        #     for i, c in enumerate(ap_class):
        #         LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
        #     for i in range(81):
        #         LOGGER.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))

        # Print speeds
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        # if not training:
        #     shape = (batch_size, 3, imgsz, imgsz)
        #     LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

        # Plots
        # if plots:
        #     confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        #     callbacks.run('on_val_end')

        # Save JSON
        # if save_json and len(jdict):
        #     w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        #     anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        #     pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        #     LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        #     with open(pred_json, 'w') as f:
        #         json.dump(jdict, f)

        #     try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        #         check_requirements(['pycocotools'])
        #         from pycocotools.coco import COCO
        #         from pycocotools.cocoeval import COCOeval

        #         anno = COCO(anno_json)  # init annotations api
        #         pred = anno.loadRes(pred_json)  # init predictions api
        #         eval = COCOeval(anno, pred, 'bbox')
        #         if is_coco:
        #             eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
        #         eval.evaluate()
        #         eval.accumulate()
        #         eval.summarize()
        #         map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        #     except Exception as e:
        #         LOGGER.info(f'pycocotools unable to run: {e}')

        # Return results
        model.float()  # for training
        # if not training:
        #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        maps = np.zeros(nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]
        return (seen, nt.sum(), mp, mr, map50, map), (mIoU, mAcc, allAcc)
        # return (mp, mr, map50, map, *(loss.cpu() / len(self.dataloader)).tolist()), maps, t

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            try:
                from engine.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            if self.per_class_mAP:
                info += "per class mAP:\n" + per_class_mAP_table(cocoEval)
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
