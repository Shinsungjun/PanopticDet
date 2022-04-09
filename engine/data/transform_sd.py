'''
    This is data augmentation for segmentation and detection.
'''

import random
import math
import numpy as np
import numbers
import collections
import cv2

import torch
import torchvision
import torchvision.transforms.functional as F

class Compose(object):
    # Composes segtransforms: segtransform.Compose([segtransform.RandScale([0.5, 2.0]), segtransform.ToTensor()])
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, image, label, bbox):
        for t in self.segtransform:
            image, label, bbox = t(image, label, bbox)
        return image, label, bbox


class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, image, label, bbox):
        if not isinstance(image, np.ndarray) or not isinstance(label, np.ndarray):
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray"
                                "[eg: data readed by cv2.imread()].\n"))
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if not len(label.shape) == 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray labellabel with 2 dims.\n"))

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        # image = F.to_tensor(image)
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        return image, label, bbox


class Normalize(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)

        self.mean = mean
        self.std = std

    def __call__(self, image, label, bbox):
        if self.std is None:
            for t, m in zip(image, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
        #! if we need to normalize bbox plz add below
        return image, label, bbox


class RandScale(object):
    # Randomly resize image & label with scale factor in [scale_min, scale_max]
    def __init__(self, scale, aspect_ratio=None):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("segtransform.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("segtransform.RandScale() aspect_ratio param error.\n"))

    def __call__(self, image, label, bbox):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)

        #! added for bbox targets
        bbox = bbox.copy()
        if "boxes" in bbox:
            boxes = bbox["boxes"]
            scaled_boxes = boxes * torch.as_tensor([scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y])
            bbox["boxes"] = scaled_boxes
            # boxes[:, :4] *= torch.as_tensor([scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y])

        if "area" in bbox:
            area = bbox["area"]
            scaled_area = area * (scale_factor_x * scale_factor_y)
            bbox["area"] = scaled_area

        return image, label, bbox

class Resize(object):
    """
    Applies random scale augmentation.
    Reference: https://github.com/tensorflow/models/blob/master/research/deeplab/input_preprocess.py#L28
    Arguments:
        min_resize_value: Desired size of the smaller image side, no resize if set to None
        max_resize_value: Maximum allowed size of the larger image side, no limit if set to None
        resize_factor: Resized dimensions are multiple of factor plus one.
        keep_aspect_ratio: Boolean, keep aspect ratio or not. If True, the input
            will be resized while keeping the original aspect ratio. If False, the
            input will be resized to [max_resize_value, max_resize_value] without
            keeping the original aspect ratio.
        align_corners: If True, exactly align all 4 corners of input and output.
    """
    def __init__(self, min_resize_value=None, max_resize_value=None, resize_factor=None,
                 keep_aspect_ratio=True, align_corners=False):
        if min_resize_value is not None and min_resize_value < 0:
            min_resize_value = None
        if max_resize_value is not None and max_resize_value < 0:
            max_resize_value = None
        if resize_factor is not None and resize_factor < 0:
            resize_factor = None
        self.min_resize_value = min_resize_value
        self.max_resize_value = max_resize_value
        self.resize_factor = resize_factor
        self.keep_aspect_ratio = keep_aspect_ratio
        self.align_corners = align_corners

        if self.align_corners:
            warnings.warn('`align_corners = True` is not supported by opencv.')

        if self.max_resize_value is not None:
            # Modify the max_size to be a multiple of factor plus 1 and make sure the max dimension after resizing
            # is no larger than max_size.
            if self.resize_factor is not None:
                self.max_resize_value = (self.max_resize_value - (self.max_resize_value - 1) % self.resize_factor)

    def __call__(self, image, label, bbox):
        if self.min_resize_value is None:
            return image, label

        [orig_height, orig_width, _] = image.shape
        orig_min_size = np.minimum(orig_height, orig_width)

        # Calculate the larger of the possible sizes
        large_scale_factor = self.min_resize_value / orig_min_size
        large_height = int(math.floor(orig_height * large_scale_factor))
        large_width = int(math.floor(orig_width * large_scale_factor))
        large_size = np.array([large_height, large_width])

        new_size = large_size
        if self.max_resize_value is not None:
            # Calculate the smaller of the possible sizes, use that if the larger is too big.
            orig_max_size = np.maximum(orig_height, orig_width)
            small_scale_factor = self.max_resize_value / orig_max_size
            small_height = int(math.floor(orig_height * small_scale_factor))
            small_width = int(math.floor(orig_width * small_scale_factor))
            small_size = np.array([small_height, small_width])

            if np.max(large_size) > self.max_resize_value:
                new_size = small_size

        # Ensure that both output sides are multiples of factor plus one.
        if self.resize_factor is not None:
            new_size += (self.resize_factor - (new_size - 1) % self.resize_factor) % self.resize_factor
            # If new_size exceeds largest allowed size
            new_size[new_size > self.max_resize_value] -= self.resize_factor

        if not self.keep_aspect_ratio:
            # If not keep the aspect ratio, we resize everything to max_size, allowing
            # us to do pre-processing without extra padding.
            new_size = [np.max(new_size), np.max(new_size)]

        # TODO: cv2 uses align_corner=False
        # TODO: use fvcore (https://github.com/facebookresearch/fvcore/blob/master/fvcore/transforms/transform.py#L377)


        # cv2: (width, height)
        # image = cv2.resize(image.astype(np.float), (new_size[1], new_size[0]), interpolation=cv2.INTER_LINEAR)
        # label = cv2.resize(label.astype(np.float), (new_size[1], new_size[0]), interpolation=cv2.INTER_NEAREST)

        scale_factor_x = float(new_size[1])/float(orig_width)
        scale_factor_y = float(new_size[0])/float(orig_height)

        image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)

        #! added for bbox targets
        bbox = bbox.copy()
        if "boxes" in bbox:
            boxes = bbox["boxes"]
            scaled_boxes = boxes * torch.as_tensor([scale_factor_x, scale_factor_x, scale_factor_x, scale_factor_x])
            bbox["boxes"] = scaled_boxes

        if "area" in bbox:
            area = bbox["area"]
            scaled_area = area * (scale_factor_x * scale_factor_x)
            bbox["area"] = scaled_area
        # return image.astype(image_dtype), label.astype(label_dtype)
        return image, label, bbox





class Crop(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, image, label, bbox):
        h, w = label.shape

        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("segtransform.Crop() need padding while padding argument is None\n"))
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)
            label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.ignore_label)
        h, w = label.shape
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        image = image[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        label = label[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]


        fields = ["labels", "area"]
        # (j, i) = 시작점(x,y), w,h = output size
        bbox = bbox.copy()
        i,j = h_off, w_off
        if "boxes" in bbox:
            boxes = bbox["boxes"]
            max_size = torch.as_tensor([self.crop_w, self.crop_h], dtype=torch.float32)
            cropped_boxes = boxes - torch.as_tensor([j, i, j, i]).float()
            cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
            cropped_boxes = cropped_boxes.clamp(min=0)
            area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
            bbox["boxes"] = cropped_boxes.reshape(-1, 4)
            bbox["area"] = area
            fields.append("boxes")
        # remove elements for which the boxes or masks that have zero area
        if "boxes" in bbox or "masks" in bbox:
            # favor boxes selection when defining which elements to keep
            # this is compatible with previous implementation
            if "boxes" in bbox:
                cropped_boxes = bbox['boxes'].reshape(-1, 2, 2)
                keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
            else:
                keep = bbox['masks'].flatten(1).any(1)

            for field in fields:
                bbox[field] = bbox[field][keep]

        return image, label, bbox


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label, bbox):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)

            w = image.shape[1]
            bbox = bbox.copy()
            if "boxes" in bbox:
                boxes = bbox["boxes"]
                boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1., 1., -1., 1.]) \
                                + torch.as_tensor([w, 0, w, 0]).float()
                bbox["boxes"] = boxes

        return image, label, bbox


class RandomColorJitter(object):
    def __init__(self, b,c,s,h, p=0.5):
        self.p = p
        self.to_pil = torchvision.transforms.ToPILImage()
        self.color_aug = torchvision.transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h)

    def __call__(self, image, label, bbox):
        if random.random() < self.p:
            image = self.to_pil(image)
            image = self.color_aug(image)
            image = np.asarray(image)

        return image, label, bbox

#%####################################################################################
'''#TODO: make change below
'''

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        return image, label


class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image, label):
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label


class RandRotate(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(self, rotate, padding, ignore_label=255, p=0.5):
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransform.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
        return image, label



# class Resize(object):
#     # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
#     def __init__(self, size):
#         assert (isinstance(size, collections.Iterable) and len(size) == 2)
#         self.size = size

#     def __call__(self, image, label):
#         image = cv2.resize(image, self.size[::-1], interpolation=cv2.INTER_LINEAR)
#         label = cv2.resize(label, self.size[::-1], interpolation=cv2.INTER_NEAREST)
#         return image, label


class RGB2BGR(object):
    # Converts image from RGB order to BGR order, for model initialized from Caffe
    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, label


class BGR2RGB(object):
    # Converts image from BGR order to RGB order, for model initialized from Pytorch
    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, label
