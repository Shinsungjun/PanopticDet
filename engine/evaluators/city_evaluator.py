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

import numpy as np

import torch

from engine.utils import (
    gather,
    is_main_process,
    synchronize,
    time_synchronized,
)


class CITYEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        num_classes: int,
        testdev: bool = False,):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.num_classes = num_classes
        self.testdev = testdev

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        devide=None,
        trt_file=None,
        decoder=None,
        test_size=None,
        ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.
        #TODO: detailed time recording이 필요한 경우와 필요없는 경우를 나눠주면 좋을듯. 이대로 쓰면 학습시간 늘어남.
        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        #TODO(YOLOX) half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        progress_bar = tqdm if is_main_process() else iter

        #% time check
        inference_time = 0

        #% for metric
        all_inter, all_union, all_targets = torch.zeros(0, device=device), torch.zeros(0, device=device), torch.zeros(0, device=device)


        for cur_iter, (imgs, targets) in enumerate(progress_bar(self.dataloader)):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                losses, outputs = model(imgs)

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                intersection, union, target = intersectionAndUnionGPU(outputs, target, 
                                                        self.num_classes, self.ignore_index)
                all_inter += intersection
                all_union += union
                all_targets += target

        #TODO: use utils.dist functions
        if distributed:
            torch.distributed.all_reduce(all_inter)
            torch.distributed.all_reduce(all_union)
            torch.distributed.all_reduce(all_targets)

        #TODO: results type, device 결정해야 함.
        all_inter, all_union, all_targets = all_inter.cpu().numpy(), all_union.cpu().numpy(), all_targets.cpu().numpy()

        iou_class = all_inter / (all_union+ 1e-10)
        accuracy_class = all_inter / (all_targets + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(all_inter) / (sum(all_targets) + 1e-10)

        # statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        # if distributed:
        #     data_list = gather(data_list, dst=0)
        #     data_list = list(itertools.chain(*data_list))
        #     torch.distributed.reduce(statistics, dst=0)

        # eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        # return eval_results
        return mIoU, mAcc, allAcc

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

    output = output.view(-1)
    target = target.view(-1)

    output[target == ignore_index] = ignore_index

    # mask of intersection where predict==target
    intersection = output[output == target] 
    
    # compute histogram of tensor. shape: [19]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1) 
    area_output = torch.histc(output, bins=K, min=0, max=K-1) 
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection

    return area_intersection, area_union, area_target
