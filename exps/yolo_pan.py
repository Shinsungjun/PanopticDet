#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import random
import yaml

import torch
import torch.distributed as dist
import torch.nn as nn

from engine.exp import BaseExp
from engine.utils.general import colorstr
from engine.data.datasets.coco_yolo import InfiniteDataLoader
from engine.utils.dist import synchronize
from engine.utils.torch_utils import torch_distributed_zero_first
# from engine.data import get_yolox_datadir

class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- config ---------------- #
        self.seed = None
        self.output_dir = "./results"

        # ---------------- model config ---------------- #
        self.num_classes = 80
        self.act = 'relu'
        #!Need to fix model_yaml from args
        self.model_yaml = '/usr/src/PanopticDet/engine/models/reg32_ctx4_seg_3_v1.yaml'

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4 #yolov5 worker
        self.input_size = (640, 640)  # (height, width)
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)

        #% TODO: train
        self.train_path = "/usr/src/PanopticDet/coco/train2017.txt"
        self.val_path = "/usr/src/PanopticDet/coco/val2017.txt"
        self.ignore_label = 255
        # self.train_ann = "instances_train2017.json"
        # self.val_ann = "instances_val2017.json"

        # --------------- transform config ----------------- #
        # self.mosaic_prob = 1.0
        # self.mixup_prob = 1.0
        # self.hsv_prob = 1.0
        # self.flip_prob = 0.5
        # self.degrees = 10.0
        # self.translate = 0.1
        # self.mosaic_scale = (0.1, 2)
        # self.mixup_scale = (0.5, 1.5)
        # self.shear = 2.0
        # self.enable_mixup = True

        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.mean = [item * value_scale for item in mean]
        self.std = [item * value_scale for item in std]
        self.scale = (0.5,2)
        self.flip_prob = 0.5
        self.crop_size = (768,1536)
        self.imgsz = 640
        self.grid_size = 32
        self.pad = 0.0
        self.hyp = {
            'hsv_h' : 0.015,
            'hsv_s' : 0.7,
            'hsv_v' : 0.4,
            'degrees' : 0,
            'translate' : 0.1,
            'scale' : 0.5,
            'shear' : 0.0,
            'perspective' : 0.0,
            'flipud' : 0.0,
            'fliplr' : 0.5,
            'mosaic' : 0.0,
            'mixup' : 0.0,
            'copy_paste' : 0.0
        }
        # --------------  training config --------------------- #
        self.warmup_epochs = 0
        self.max_epoch = 300
        self.warmup_lr = 0
        self.basic_lr = 0.01
        # self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "poly"
        # self.no_aug_epochs = 15
        # self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 0.0005
        self.momentum = 0.937
        self.print_interval = 10
        self.eval_interval = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        #self.device = torch.device('cuda', LOCAL_RANK)
        # -----------------  testing config ------------------ #
        self.test_size = (1024, 2048)
        # self.test_conf = 0.01
        # self.nmsthre = 0.65

    def get_model(self):
        from engine.models import Model

        # init BN mode ????????? ??????, ????????? freezeBN ?????? ??????????????? ?????? ???
        def init_BN(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        # ?????? template??? fullmodel??? ?????? ??? ???
        if getattr(self, "model", None) is None:
            self.model = Model(self.model_yaml , ch=3, nc=self.num_classes, anchors=None)

        self.model.apply(init_BN)
        # self.model.head.initialize_biases(1e-2)
        return self.model


    def get_model_nonsequential(self):
        def init_BN(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        #if getattr(self, "model", None) is None:
        from engine.models.yolov5 import Fullmodel
        from engine.models.backbone import CSPDarkNet
        from engine.models.neck import YoloNeck
        from engine.models.head import Detect
        backbone = CSPDarkNet()

        neck = YoloNeck()
        
        head = Detect(nc=80, anchors=None) #!Need to put anchor
        
        self.model = Fullmodel(backbone=backbone, ob_neck = neck, ob_head=head)
            
        self.model.apply(init_BN)
        # self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from engine.data import (
            LoadImagesAndLabels
        )
        from engine.utils import (
            wait_for_the_master,
            get_local_rank,
            get_world_size
        )

        local_rank = get_local_rank()
        #% dataset init ?????? ?????? ???cd 
        if is_distributed:
            with torch_distributed_zero_first(local_rank):
                dataset = LoadImagesAndLabels(self.train_path, self.imgsz, batch_size // get_world_size(),
                                            augment=True,  # augment images
                                            hyp=self.hyp,  # augmentation hyperparameters
                                            rect=False,  # rectangular training
                                            cache_images=None,
                                            single_cls=False,
                                            stride=int(self.grid_size),
                                            pad=self.pad,
                                            image_weights=False,
                                            prefix=colorstr('train: '))

        else:
            dataset = LoadImagesAndLabels(self.train_path, self.imgsz, batch_size // get_world_size(),
                                            augment=True,  # augment images
                                            hyp=self.hyp,  # augmentation hyperparameters
                                            rect=False,  # rectangular training
                                            cache_images=None,
                                            single_cls=False,
                                            stride=int(self.grid_size),
                                            pad=self.pad,
                                            image_weights=False,
                                            prefix=colorstr('train: '))
        self.dataset = dataset

        #% dataloader args: batch_size, shuffle, drop_last, sampler
        if is_distributed:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(self.dataset)
        else:
            sampler =  None

        loader = InfiniteDataLoader

        dataloader_kwargs = {"num_workers": self.data_num_workers,
                            "pin_memory": True,
                            "batch_size": batch_size // get_world_size(),
                            "sampler": sampler,
                            'collate_fn': LoadImagesAndLabels.collate_fn}
        
        train_loader = loader(self.dataset, **dataloader_kwargs)


        return train_loader

        #% using custom batchSampler
        # sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        # batch_sampler = YoloBatchSampler(
        #     sampler=sampler,
        #     batch_size=batch_size,
        #     drop_last=False,
        #     mosaic=not no_aug,
        # )
        # dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        # dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        # dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        # train_loader = DataLoader(self.dataset, **dataloader_kwargs)



    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets

    def get_optimizer(self, batch_size):
        ''' #! warm-up??? weight,bias??? decay ????????? ???????????? + dict????????????
            yolo??? warm-up??????, weight, bias??? ??????, decay?????? ?????? ????????? ??????. 
            joint ??????, ????????? ??? ????????? ????????? ???.
        '''
        #TODO: adam version ??????
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                # lr = self.basic_lr_per_img * batch_size
                lr = self.basic_lr
            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True)          
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay     
            optimizer.add_param_group({"params": pg2})

            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        #from engine.utils.lr_scheduler import poly_lr #TODO: change Our class RLScheduler
        from engine.utils.lr_scheduler import LRScheduler
        scheduler = LRScheduler(
                self.scheduler, 
                lr,
                iters_per_epoch,
                self.max_epoch,
                warmup_epochs=self.warmup_epochs,
                warmup_lr_start=self.warmup_lr,
                optimizer = self.optimizer
                # no_aug_epochs=self.no_aug_epochs,
                # min_lr_ratio=self.min_lr_ratio, 
        ) #! ????????? argument??? ??????????????? self.__dict__.update ??? attribute??? init?????? ????????? (self.optimizer??? ????????? ???????????? ???)

        return scheduler

    #% eval dataset loader
    def get_eval_loader(self, batch_size, is_distributed=False, testdev=False, legacy=False):
        from engine.data import LoadImagesAndLabels
        from engine.utils import (
            wait_for_the_master,
            get_local_rank,
        )
        valdataset = LoadImagesAndLabels(self.val_path, self.imgsz, batch_size,
                                    augment=False,  # augment images
                                    hyp=self.hyp,  # augmentation hyperparameters
                                    rect=True,  # rectangular training
                                    cache_images=None,
                                    single_cls=False,
                                    stride=int(self.grid_size),
                                    pad=0.5,
                                    image_weights=False,
                                    prefix=colorstr('val: '))
            

        dataloader_kwargs = {"num_workers": self.data_num_workers,
                            "pin_memory": True,
                            "batch_size": batch_size,
                            "sampler": None,
                            'collate_fn': LoadImagesAndLabels.collate_fn}

        loader = InfiniteDataLoader

        val_loader = loader(valdataset, **dataloader_kwargs)

        return val_loader

    #% eval ???????????? ?????? ????????? ?????? ?????? ???????????? ?????? eval??? ?????? ????????? ???????????? ?????????.
    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from engine.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size = 640,
        confthre = 0.001,
        nmsthre = 0.6,
        num_classes=80,
        )
        return evaluator

    def eval(self, model, evaluator, is_distributed, half=False, logger = None):
        #TODO ?????? eval??? ????????? ???????????? ?????? ?????? ???????????? ??????.
        return evaluator.evaluate(model, is_distributed, half, LOGGER = logger)
