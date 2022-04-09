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

import numpy as np

import torch

from engine.data.datasets import COCO_CLASSES
from engine.utils.metrics import ap_per_class, ConfusionMatrix, box_iou
from engine.utils.general import coco80_to_coco91_class, non_max_suppression, xyxy2xywh, xywh2xyxy, scale_coords
from engine.utils.torch_utils import time_sync
from engine.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)

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
        names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
        class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
        s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        #! add seg loss -> torch.zeros(3, device = device) -> torch.zeros(4, device = device)
        loss = torch.zeros(4, device=device)
        jdict, stats, ap, ap_class = [], [], [], []


        intersection_meter = AverageMeter('IoU')
        union_meter = AverageMeter('Union')
        target_meter = AverageMeter('Target')

        for batch_i, (img, segs, targets, paths, shapes) in enumerate(tqdm(self.dataloader, desc=s)):
            segs = segs.to(device)
            t1 = time_sync()
            img = img.to(device, non_blocking=True)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255  # 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)
            nb, _, height, width = img.shape  # batch size, channels, height, width
            t2 = time_sync()
            dt[0] += t2 - t1

            # Run model
            #! add segmentation
            (out, train_out), pred_seg = model(img, augment=False)  # inference and training outputs
            dt[1] += time_sync() - t2
            # Compute loss
            if compute_loss:
                #! add segmentation
                loss += compute_loss([[x.float() for x in train_out], [y.float() for y in pred_seg]], targets, segs)[1]  # box, obj, cls
            #! add segmentation metrics
            # intersection, union, target = intersectionAndUnionGPU(pred_seg[-1].max(1)[1], segs, 81, 255)
            intersection, union, target = intersectionAndUnionGPU(pred_seg[0].max(1)[1], segs, 81, 255)
            # intersection, union, target = intersectionAndUnionGPU(pred_seg[-1].max(1)[1], segs, 80, 80)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            # lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            lb = []
            t3 = time_sync()
            # out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
            out = non_max_suppression(out, 0.001, 0.6, labels=lb, multi_label=True, agnostic=False)
            dt[2] += time_sync() - t3



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

                # Evaluate
                if nl:
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


        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, names=names)
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
