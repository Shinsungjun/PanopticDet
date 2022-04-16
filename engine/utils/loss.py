# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from engine.utils.general import xywh2xyxy, make_priors, center_size, crop, match

from engine.utils.metrics import bbox_iou
from engine.utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

from engine.utils.general import xywh2xyxy
import torch
def yolo2yolact_target(targets):
    '''
    yolo target -> yolact target
    yolo : tensors [img_num, class, x, y, w, h]
    yolact : list of tensors [img1:[num_objs, x1, y1, x2, y2, class], img2:[...]]
    '''
    #* convert xywh -> xyxy format
    convert_target = targets.clone().detach()
    xywhbox = convert_target[:, 2:]
    xyxybox = xywh2xyxy(xywhbox)
    t_cls = convert_target[:, 1:2].clone()
    convert_target[:, 1:5] = xyxybox
    convert_target[:, 5:6] = t_cls
    #* convert tensors -> list of tensors
    yolact_target = []
    batch_target = []
    img_num = 0
    for i, t in enumerate(convert_target):
        if t[0] == img_num:
            batch_target.append(t[1:])
        if i+1 < len(convert_target):
            if convert_target[i+1, 0] != img_num:

                yolact_target.append(torch.stack(batch_target))
                batch_target = []
                img_num += 1
        else:
            yolact_target.append(torch.stack(batch_target))
    
    return yolact_target

class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        self.pos_threshold = 0.5
        self.neg_threshold = 0.5


        device = next(model.parameters()).device  # get model device
        #h = model.hyp  # hyperparameters
        h = {'cls_pw' : 1.0,
            'obj_pw' : 1.0,
            'fl_gamma' : 0.0,
            'anchor_t' : 4.0,
            'box' : 0.05,
            'obj' : 1.0,
            'seg' : 1.0,
            'cls' : 0.5,
            'mask' : 1.0,
            'label_smoothing' : 0.0}
        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        det = model.module.ob_head if is_parallel(model) else model.ob_head
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

        # self.segloss = nn.CrossEntropyLoss()
        self.segBCE = nn.BCEWithLogitsLoss()
        self.segCE = nn.CrossEntropyLoss()
        self.priors = None

    def __call__(self, pred, targets, segs=False, gt_masks=None):  # predictions, targets, model
    
        assert len(segs.shape) == 3
        p = pred['box_pred']
        device = targets.device
        #! lseg add
        lmask, lseg, lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        #! make yolact target
        predictions_for_mask_loss, targets_for_mask_loss = self.make_mask_target(pred['box_pred'],  pred['kernel_pred'],pred['mask_pred'], targets, device)

        #! compute mask loss
        lmask += self.mask_loss(predictions_for_mask_loss, targets_for_mask_loss, gt_masks)
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE


                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        #! segmentation loss BCS
        aux_seg = pred['aux_seg_pred']
        lseg += self.segCE(aux_seg, segs)

        pred_seg = pred['seg_pred']
        bb,cc,hh,ww = pred_seg.shape
        segs = F.one_hot(segs, num_classes=81) # [nBox, 80]
        segs = segs.reshape(bb,-1,cc).type_as(pred_seg)
        seg_logits = pred_seg.permute(0,2,3,1)
        seg_logits = seg_logits.reshape(bb,-1,cc)
        lseg += self.segBCE(seg_logits, segs)
        # lseg = lseg/2.0
        # lseg += self.segloss(pred[1], segs)


        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lseg *= self.hyp['seg'] #! segmentation loss
        lmask *= self.hyp['mask']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls + lseg + lmask) * bs, torch.cat((lbox, lobj, lcls, lseg, lmask)).detach()
        # return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    def make_mask_target(self, box_pred, kernel_pred, mask_pred,  targets, device):
        """
        args:
            
            pred(list): [P8, P16, P32]
                P8 shape: [B x na x H/8 x W/8 x C]
                P16 shape: [B x na x H/16 x W/16 x C]
                P32 shape: [B x na x H/32 x W/32 x C]

                C = (# of class(80) + bbox(4) + objectness(1))
                na = # of anchors(3)
            len(pred) = 2
            pred = [P8, P16, P32]
            pred[0].shape = torch.Size([8, 3, 80, 80, 85])
            pred[1].shape = torch.Size([8, 3, 40, 40, 85]) 
            pred[2].shape = torch.Size([8, 3, 20, 20, 85])
            len(pred[0][2]) = 8
            pred[1]  =[]

            pred_mask = [P8, P16, P32]
                P8 shape: [B, na, H/8, W/8, 64]
                P16 shape: [B, na, H/16, W/16, 64]
                P32 shape: [B, na, H/32, W/32, 64]

            targets(tensor):  [nt, img_idxn + cls + bbox]

            masks = [B, # of True, 160, 160]
            masks(list<tensor>): 
                Ground truth masks for each object in each image,
                shape: [batch_size][num_objs,im_height,im_width]

            proto_data(tensor): 
                [B, H/4, W/4, K = 64]
        
        Out:
            mask_pred: 
                mask_pred['loc'] = torch.Size([2, 19248, 4])
                mask_pred['mask'] = torch.Size([2, 19248, 32])
                mask_pred['priors'] = torch.Size([19248, 4])
                mask_pred['proto'] = torch.Size([2, 138, 138, 32])
                mask_pred['score'] = ?
                mask_pred['inst'] = ?
                ?는 inference때 필요

            target_bbox_for_mask(list<tensor>)):
                [batch_size][num_objs,5]
        """
        
        predictions = {'loc':[], 'kernel': [], 'mask':[], 'conf' : []}
        yolact_target = yolo2yolact_target(targets)
        #! mask_pred['loc']
        # pred에서 bouding box만 가져오기
        box_loc = []
        for p in box_pred:
            B, A, H, W, C = p.shape
            box = p[...,:4].clone().view(B, -1, 4)
            box_loc.append(box)
        
        box_loc = torch.cat(box_loc, dim=1)
        #print("box_loc shape :", box_loc.shape) #[2, 25200, 4]
        predictions['loc'] = box_loc
        #print('kernel_pred shape : ', kernel_pred.shape) #[2, 25200, 64]
        predictions['kernel'] = kernel_pred

        #! mask_pred['priors']

        if self.priors == None:
            anchors = [[10, 13, 16,30, 33,23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
            anchors = torch.tensor(anchors).float()
            anchors = anchors.view(3, -1, 2)
            for i in range(len(box_pred)):
                
                B, A, H, W, C = box_pred[i].shape

                prior = make_priors(H, W, anchors[i])
                if i == 0:
                    self.priors = prior
                else:
                    self.priors = torch.cat([self.priors, prior], dim = 0)

            self.priors = self.priors.to(device)
        

        predictions['mask'] = mask_pred 

        return predictions, yolact_target

    def mask_loss(self, predictions, targets, gt_mask):
        """
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            mask preds, and prior boxes from SSD net.
                loc shape: torch.size(batch_size,num_priors,4)
                conf shape: torch.size(batch_size,num_priors,num_classes)
                masks shape: torch.size(batch_size,num_priors,mask_dim)
                priors shape: torch.size(num_priors,4)
                proto* shape: torch.size(batch_size,mask_h,mask_w,mask_dim)

            targets (list<tensor>): Ground truth boxes and labels for a batch,
                shape: [batch_size][num_objs,5] (last idx is the label).

            masks (list<tensor>): Ground truth masks for each object in each image,
                shape: [batch_size][num_objs,im_height,im_width]

            num_crowds (list<int>): Number of crowd annotations per batch. The crowd
                annotations should be the last num_crowds elements of targets and masks.
            
        """
        loc_data  = predictions['loc']
        kernel_data = predictions['kernel']
        mask_data = predictions['mask']
        priors    = self.priors

        # score_data = predictions['score'] if cfg.use_mask_scoring   else None   
        # inst_data  = predictions['inst']  if cfg.use_instance_coeff else None

        class_labels = [None] * len(targets) # Used in sem segm loss

        batch_size = loc_data.size(0)
        num_priors = priors.size(0)

        loc_t = loc_data.new(batch_size, num_priors, 4)
        gt_box_t = loc_data.new(batch_size, num_priors, 4)
        conf_t = loc_data.new(batch_size, num_priors).long()
        idx_t = loc_data.new(batch_size, num_priors).long()

        for idx in range(batch_size):
            box_truths      = targets[idx][:, :-1].data # object number x 5에서 object number x 4만 가져옴
            class_labels[idx] = targets[idx][:, -1].data.long() # 맨 마지막 열만 가져옴 long을 써서 앞에 숫자만 자름
            match(self.pos_threshold, self.neg_threshold,
                box_truths, priors.data, class_labels[idx],
                loc_t, conf_t, idx_t, idx)
                  
            gt_box_t[idx, :, :] = box_truths[idx_t[idx]] # 어떤 이미지에 대해서 19248 픽셀에서의 모든 바운딩 박스 타겟 만듬

        idx_t = Variable(idx_t, requires_grad=False)

        pos = conf_t > 0 # BOOL
        num_pos = pos.sum(dim=1, keepdim=True)

        loss = self.lincomb_mask_loss(pos, idx_t, loc_data, kernel_data, priors, mask_data, gt_mask, gt_box_t, class_labels, targets[0].device)

        # Divide all losses by the number of positives.
        # Don't do it for loss[P] because that doesn't depend on the anchors.
        total_num_pos = num_pos.data.sum().float()
        loss /= total_num_pos

        return loss
    

    def lincomb_mask_loss(self, pos, idx_t, loc_data, kernel_data, priors, proto_data, gt_masks, gt_box_t, labels, device, score_data = None, inst_data = None):

        masks_to_train = 100
        mask_alpha = 6.125
        
        mask_h = proto_data.size(2)
        mask_w = proto_data.size(3)

        process_gt_bboxes = True

        loss_m = 0
        for idx in range(kernel_data.size(0)):
            with torch.no_grad():
                gt_masks[idx] = gt_masks[idx].type(torch.float32)
                downsampled_masks = F.interpolate(gt_masks[idx].unsqueeze(0), (mask_h, mask_w),
                                                  mode='bilinear', align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()
                downsampled_masks = downsampled_masks.gt(0.5).float()

            cur_pos = pos[idx]
            pos_idx_t = idx_t[idx, cur_pos]
            
            if process_gt_bboxes:
                # Note: this is in point-form
                pos_gt_box_t = gt_box_t[idx, cur_pos]
                    # 현재 mask_data중에 현재 positive box인 애

            if pos_idx_t.size(0) == 0:
                continue

            proto_masks = proto_data[idx]
            #매칭되는 proto_masks
            proto_coef  = kernel_data[idx, cur_pos, :]

            # If we have over the allowed number of masks, select a random sample
            old_num_pos = proto_coef.size(0)
            # if old_num_pos > cfg.masks_to_train:
            if old_num_pos > masks_to_train:
                perm = torch.randperm(proto_coef.size(0))
                select = perm[:masks_to_train]

                proto_coef = proto_coef[select, :]
                pos_idx_t  = pos_idx_t[select]
                
                if process_gt_bboxes:
                    pos_gt_box_t = pos_gt_box_t[select, :]

            num_pos = proto_coef.size(0)
            mask_t = downsampled_masks[:, :, pos_idx_t]
            mask_t = mask_t.to(device)
            #   gt box에 매치되는 마스크
            label_t = labels[idx][pos_idx_t]  
            # gt box에 매치되는 레이블
            

            # Size: [mask_h, mask_w, num_pos]
            proto_masks = proto_masks.permute(1,2,0)
            pred_masks = proto_masks @ proto_coef.t()
    
            
            pred_masks = pred_masks.sigmoid()
            pred_masks = crop(pred_masks, pos_gt_box_t)

            pre_loss = F.binary_cross_entropy(torch.clamp(pred_masks, 0, 1), mask_t, reduction='none')# pre_loss = F.binary_cross_entropy(torch.clamp(pred_masks, 0, 1), mask_t, reduction='none')

            weight = mask_h * mask_w 
            pos_gt_csize = center_size(pos_gt_box_t)
            gt_box_width  = pos_gt_csize[:, 2] * mask_w
            gt_box_height = pos_gt_csize[:, 3] * mask_h
            pre_loss = pre_loss.sum(dim=(0, 1)) / gt_box_width / gt_box_height * weight

            # If the number of masks were limited scale the loss accordingly
            if old_num_pos > num_pos:
                pre_loss *= old_num_pos / num_pos

            loss_m += torch.sum(pre_loss)

        losses = loss_m * mask_alpha / mask_h / mask_w

        return losses
