#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from engine.exp import BaseExp
# from engine.data import get_yolox_datadir


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- config ---------------- #
        self.seed = None
        self.output_dir = "./results"

        # ---------------- model config ---------------- #
        self.num_classes = 19
        self.act = 'relu'

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (1024, 2048)  # (height, width)
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)

        #% TODO: train
        self.data_dir = "/usr/src/EXP_template/cityscapes"
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

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (1024, 2048)
        # self.test_conf = 0.01
        # self.nmsthre = 0.65

    def get_model(self):
        from engine.models import SegModel, Ctxnet, SegHead

        # init BN mode 정도로 해서, 나중애 freezeBN 까지 추가해주면 좋을 듯
        def init_BN(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        # 이전 template의 fullmodel로 보면 될 듯
        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = Ctxnet()
            head = SegHead()
            self.model = SegModel(backbone, head)

        self.model.apply(init_BN)
        # self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from engine.data import (
            Cityscapes,
            transform,
            worker_init_reset_seed,
        )
        from engine.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        #% dataset init 으로 보면 됨cd 
        with wait_for_the_master(local_rank):
            dataset = Cityscapes(
                data_root=self.data_dir,
                split="train",
                # ignore_label=self.ignore_label,
                transforms=transform.Compose([ #% init에 parameter 추가
                    transform.RandScale(self.scale),
                    transform.RandomHorizontalFlip(p=self.flip_prob),
                    transform.Crop(self.crop_size, crop_type='rand', padding=self.mean),
                    transform.ToTensor(),
                    transform.Normalize(mean=self.mean, std=self.std)
                    ]),
                cache=cache_img,
            )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        #% dataloader args: batch_size, shuffle, drop_last, sampler
        if is_distributed:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(self.dataset)
        else:
            sampler =  None

        dataloader_kwargs = {"num_workers": self.data_num_workers,
                            "pin_memory": True,
                            "batch_size": batch_size,
                            "drop_last": True,
                            "shuffle": False,
                            "sampler": sampler}
        train_loader = torch.utils.data.DataLoader(self.dataset, **dataloader_kwargs)


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
        ''' #! warm-up과 weight,bias의 decay 구분과 관련하여 + dict관련하여
            yolo의 warm-up부분, weight, bias에 따라, decay다른 부분 등등이 있음. 
            joint 할때, 어떻게 할 것인지 정해야 함.
        '''
        #TODO: adam version 필요
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
                pg0, lr=lr, momentum=self.momentum, weight_decay=self.weight_decay)          
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
        ) #! 이렇게 argument로 넣어놓으면 self.__dict__.update 로 attribute로 init에서 변경됨 (self.optimizer가 알아서 생긴다는 말)

        return scheduler

    #% eval dataset loader
    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from engine.data import Cityscapes, transform
        valdataset = Cityscapes(
                data_root="/usr/src/EXP_template/cityscapes",
                split="val",
                ignore_label=self.ignore_label,
                transforms=transform.Compose([ #% init에 parameter 추가
                    transform.ToTensor(),
                    transform.Normalize(mean=self.mean, std=self.std)
                    ]),)
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset)
        else:
            sampler =  None

        dataloader_kwargs = {"num_workers": self.data_num_workers,
                            "pin_memory": True,
                            "batch_size": batch_size,
                            "drop_last": False,
                            "shuffle": False,
                            "sampler": sampler}
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    #% eval 부분인데 굳이 이렇게 따로 만들 필요없이 아래 eval에 그냥 구현해 버리는게 좋은듯.
    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from engine.evaluators import CITYEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = CITYEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator

    def eval(self, model, evaluator, is_distributed, half=False):
        #TODO 그냥 eval을 여기에 구현하는 것도 좋은 방법으로 보임.
        return evaluator.evaluate(model, is_distributed, half)
