#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger #수정
import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from engine.data import DataPrefetcher
from engine.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_local_rank,
    get_model_info, #TODO: model summary
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)
from engine.utils.general import one_cycle
from engine.utils.loss import ComputeLoss

#! check list
#! exp.output_dir 위치

class Trainer:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema

        self.exp.no_aug_epochs = 0 #TODO: chage to paramter
        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0
        self.best_mIoU = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()



    def before_train(self):
        logger.info("args: {}".format(self.args)) #TODO: 저장되는 것인지, 그냥 Logging인지 확인 --> rank0만 저장되는 듯.
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        # logger.info(
        #     "Model Summary: {}".format(get_model_info(model.eval(), self.exp.test_size))
        # )
        model.train()
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)
        warmup_epochs = 3
        self.num_warm = max(round(warmup_epochs * self.max_iter), 1000)

        # self.lr_scheduler = self.exp.get_lr_scheduler(
        #     # self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        #     self.exp.basic_lr * self.args.batch_size, self.max_iter
        # )
        self.lf = one_cycle(1, 0.1, self.max_epoch)
        self.lr_scheduler = lr_scheduler.LambdaLR(self.optimizer , lr_lambda=self.lf)
        self.lr_scheduler.last_epoch = -1

        self.accumulate = max(round(64 / self.args.batch_size), 1)

        if self.args.occupy:
            occupy_mem(self.local_rank)
        self.last_opt_step = -1
        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.model.train()

        self.compute_loss = ComputeLoss(model)
        if self.rank in [-1, 0]:
            self.evaluator = self.exp.get_evaluator(
                batch_size=1, is_distributed=False
            )
        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(self.file_name)

        logger.info("Training start...")
        logger.info("\n{}".format(model))
        synchronize()
    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()


    def before_epoch(self):
        logger.info("init prefetcher, this might take one minute or less...")
        #self.prefetcher = DataPrefetcher(self.train_loader) #! 우선 skip .. lisf of dict로 들어가면서 prefetcher가 힘듬
        self.prefetcher = iter(self.train_loader)

        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            # if self.iter != 0 and self.iter % 30 == 0:
            #     if self.rank in [-1, 0]:
            #         self.evaluate_and_save_model()
            # synchronize()
            self.train_one_iter()
            self.after_iter()


    def before_iter(self):
        pass

    def train_one_iter(self):
        iter_start_time = time.time()

        imgs, segs, targets, paths, _ = self.prefetcher.__next__() #! 우선 스킵
        data_end_time = time.time()

        ni = self.iter + self.max_iter * self.epoch  # number integrated batches (since train start)
        imgs = imgs.to(self.device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

        # Warmup
        if ni <= self.num_warm:
            xi = [0, self.num_warm]  # x interp
            # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
            self.accumulate = max(1, np.interp(ni, xi, [1, 64 / self.args.batch_size]).round())
            for j, x in enumerate(self.optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * self.lf(self.epoch)])
                if 'momentum' in x:
                    warmup_momentum = 0.8
                    momentum = 0.937
                    x['momentum'] = np.interp(ni, xi, [warmup_momentum, momentum])

        # Forward
        inf_start_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            pred = self.model(imgs)  # forward
            inf_end_time = time.time()

            loss, loss_items = self.compute_loss(pred, targets.to(self.device), segs.to(self.device))  # loss scaled by batch_size
            if self.rank != -1:
                loss *= get_world_size()  # gradient averaged between devices in DDP mode

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        if ni - self.last_opt_step >= self.accumulate:
            self.scaler.step(self.optimizer)  # optimizer.step
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.use_model_ema:
                self.ema_model.update(self.model)
            self.last_opt_step = ni

        # lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1) #TODO: add update_lr
        # for param_group in self.optimizer.param_groups:
        #     param_group["lr"] = lr
        lr = [x['lr'] for x in self.optimizer.param_groups][0]  # for loggers
        outputs = {}
        outputs['box_loss'] = loss_items[0]
        outputs['obj_loss'] = loss_items[1]
        outputs['cls_loss'] = loss_items[2]
        # outputs['seg_loss'] = loss_items[3]
        iter_end_time = time.time()

        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            inference_time = inf_end_time - inf_start_time,
            lr=lr,
            **outputs,
        )

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.5f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            self.meter.clear_meters()

        # random resizing
        # if (self.progress_in_iter + 1) % 10 == 0:
        #     self.input_size = self.exp.random_resize(
        #         self.train_loader, self.epoch, self.rank, self.is_distributed
        #     )


    def after_epoch(self):
        self.lr_scheduler.step()
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            if self.rank in [-1, 0]:
                self.evaluate_and_save_model()
            synchronize()

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_mIoU * 100)
        )



    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        det_results, seg_result = self.exp.eval(
            evalmodel, self.evaluator, self.is_distributed, logger=logger
        )
        self.model.train()
        ap50 = det_results[4]
        ap50_95 = det_results[5]
        pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
        logger.info("               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95")
        logger.info(pf % ('all', det_results[0], det_results[1], det_results[2], det_results[3], det_results[4], det_results[5]))
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(seg_result[0], seg_result[1], seg_result[2]))
        if self.rank in [-1, 0]:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)

        self.save_ckpt("last_epoch", ap50_95 > self.best_ap)
        self.best_ap = max(self.best_ap, ap50_95)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank in [-1, 0]:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )
