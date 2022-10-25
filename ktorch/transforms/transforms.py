import math
import random
import numbers
from io import BytesIO
from collections.abc import Iterable

import PIL
from PIL import Image
import torch
import torchvision
import numpy as np

__all__ = ['RandomScale', 'RandomRotate90', 'RandomScaleBlur', 'RandomJPEGQuality', 
           'Cutout', 'CenterPadTo', 'CenterCropTo']


class RandomScale(object):
    def __init__(self, dst_size, min_scale_factor, max_scale_factor, interpolation=Image.BILINEAR):
        if not (isinstance(dst_size, int) or (isinstance(dst_size, Iterable) and len(dst_size) == 2)):
            raise TypeError('Got inappropriate size arg: {}'.format(dst_size))
        assert min_scale_factor <= max_scale_factor
        
        self.dst_size = dst_size
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.interpolation = interpolation
        
    def __call__(self, image):
        scale_factor = random.uniform(self.min_scale_factor, self.max_scale_factor)
        if isinstance(self.dst_size, int):
            dst_height, dst_width = self.dst_size, self.dst_size
        else:
            dst_height, dst_width = self.dst_size
        scaled_height = int(round(dst_height * scale_factor))
        scaled_width = int(round(dst_width * scale_factor))
        image = torchvision.transforms.functional.resize(
            image, size=(scaled_height, scaled_width), 
            interpolation=self.interpolation)
        return image
        
    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', min_scale_factor={0}'.format(self.min_scale_factor)
        format_string += ', max_scale_factor={0}'.format(self.max_scale_factor)
        format_string += ', interpolation={0})'.format(self.interpolation)
        return format_string


class RandomRotate90(object):
    def __init__(self, num_repeats=1, p=0.5):
        assert isinstance(num_repeats, int)
        self.num_repeats = num_repeats % 4
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:

            if self.num_repeats == 0:
                return image
            elif self.num_repeats == 1:
                return image.transpose(Image.ROTATE_90)
            elif self.num_repeats == 2:
                return image.transpose(Image.ROTATE_180)
            elif self.num_repeats == 3:
                return image.transpose(Image.ROTATE_270)
        return image
        
    def __repr__(self):
        class_name = self.__class__.__name__
        return class_name + '(num_repeats={}, p={})'.format(self.num_repeats, self.p)
        
        
class RandomScaleBlur(object):
    def __init__(self, min_scale_factor, max_scale_factor, p=0.5):
        assert min_scale_factor <= max_scale_factor
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.p = p
        
    def __call__(self, image):
        if random.random() < self.p:
            method_f = random.choice([PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC])
            method_b = random.choice([PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC])
            log_ratio = (math.log(self.min_scale_factor), math.log(self.max_scale_factor))
            factor = math.exp(random.uniform(*log_ratio))
            # print('method_f: {}, method_b: {}, factor: {}'.format(method_f, method_b, factor))
            src_height, src_width = image.size
            scaled_height = int(round(src_height * factor))
            scaled_width = int(round(src_width * factor))
            image = image.resize((scaled_height, scaled_width), resample=method_f)
            image = image.resize((src_height, src_width), resample=method_b)
        return image
        
    def __repr__(self):
        class_name = self.__class__.__name__
        return class_name + '(min_scale_factor={}, max_scale_factor={}, p={})'.format(
            self.min_scale_factor, self.max_scale_factor, self.p)


class RandomJPEGQuality(object):
    def __init__(self, min_quality=5, max_quality=95, p=0.5):
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.p = p
        
    def __call__(self, image):
        if random.random() < self.p:
            quality = random.randint(self.min_quality, self.max_quality)
            stream = BytesIO()
            image.save(stream, "JPEG", quality=quality)
            stream.seek(0)
            image = Image.open(stream)
        return image


def _get_image_size(img):
    if isinstance(img, Image.Image):
        return img.size
    elif isinstance(img, np.ndarray):
        return img.shape[:2][::-1]
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class CenterPadTo(object):
    def __init__(self, dst_size):
        if not (isinstance(dst_size, int) or (isinstance(dst_size, Iterable) and len(dst_size) == 2)):
            raise TypeError('Got inappropriate size arg: {}'.format(dst_size))
        self.dst_size = dst_size

    def __call__(self, image):
        width, height = _get_image_size(image)
        if isinstance(self.dst_size, int):
            dst_height, dst_width = self.dst_size, self.dst_size
        else:
            dst_height, dst_width = self.dst_size
            
        padding_x = max(dst_width - width, 0)
        padding_y = max(dst_height - height, 0)
        padding_top = padding_y // 2
        padding_left = padding_x // 2
        
        padding = (padding_left, padding_top, padding_x - padding_left, padding_y - padding_top)
        return torchvision.transforms.functional.pad(image, padding, 0, 'constant')
        
    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0})'.format(self.size)
        return format_string
        
        
class CenterCropTo(object):
    def __init__(self, dst_size):
        if not (isinstance(dst_size, int) or (isinstance(dst_size, Iterable) and len(dst_size) == 2)):
            raise TypeError('Got inappropriate size arg: {}'.format(dst_size))
        self.dst_size = dst_size

    def __call__(self, image):
        width, height = _get_image_size(image)
        if isinstance(self.dst_size, int):
            dst_height, dst_width = self.dst_size, self.dst_size
        else:
            dst_height, dst_width = self.dst_size

        dst_width = min(width, dst_width)
        dst_height = min(height, dst_height)
        return torchvision.transforms.functional.center_crop(image, (dst_height, dst_width))
        
    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0})'.format(self.size)
        return format_string


class Cutout(object):
    def __init__(self, mask_size, cutout_inside=False, mask_color=(0, 0, 0), p=0.5):
        self.mask_size = mask_size
        self.cutout_inside = cutout_inside
        self.mask_color = mask_color
        self.p = p

    @staticmethod
    def get_params(image, mask_size, cutout_inside):
        mask_size_half = mask_size // 2
        offset = 1 if mask_size % 2 == 0 else 0
        
        w, h = image.size
        if cutout_inside:
            cx_min, cx_max = mask_size_half, w + offset - mask_size_half
            cy_min, cy_max = mask_size_half, h + offset - mask_size_half
        else:
            # # Method I
            # cx_min, cx_max = 0, w + offset
            # cy_min, cy_max = 0, h + offset
            # Method II
            cx_min, cx_max = -mask_size_half + offset, w + mask_size_half
            cy_min, cy_max = -mask_size_half + offset, h + mask_size_half

        # `random.randint` return random integer in range [a, b], including both end points.
        # `np.random.randint` return random integers from `low` (inclusive) to `high` (exclusive).
        # 所以 cx_max 和 cy_max 不需要减 1
        cx = np.random.randint(cx_min, cx_max)
        cy = np.random.randint(cy_min, cy_max)
        x_min = cx - mask_size_half
        y_min = cy - mask_size_half
        x_max = x_min + mask_size
        y_max = y_min + mask_size
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, w)
        y_max = min(y_max, h)
        return x_min, y_min, x_max, y_max
        
    def __call__(self, image):
        if np.random.random() < self.p:
            x_min, y_min, x_max, y_max = self.get_params(image, self.mask_size, self.cutout_inside)
            image = np.asarray(image).copy()
            image[y_min:y_max, x_min:x_max] = self.mask_color
            image = Image.fromarray(image)
        return image


class RandomShift(object):
    def __init__(self, x_shift_min, x_shift_max, y_shift_min, y_shift_max, p=0.5):
        assert isinstance(x_shift_min, numbers.Integral)
        assert isinstance(x_shift_max, numbers.Integral)
        assert isinstance(y_shift_min, numbers.Integral)
        assert isinstance(y_shift_max, numbers.Integral)
        assert x_shift_min <= x_shift_max
        assert y_shift_min <= y_shift_max
        
        self.x_shift_min = x_shift_min
        self.x_shift_max = x_shift_max
        self.y_shift_min = y_shift_min
        self.y_shift_max = y_shift_max
        self.p = p
        
    def __call__(self, image):
        if random.random() < self.p:
            # random.randint return random integer in range [a, b], including both end points.
            x_shift = random.randint(self.x_shift_min, self.x_shift_max)
            y_shift = random.randint(self.y_shift_min, self.y_shift_max)
            input_w, input_h =  _get_image_size(image)
            # torchvision.transforms.functional.crop(img: torch.Tensor, top: int, left: int, height: int, width: int)
            # https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.crop.html
            return torchvision.transforms.functional.crop(image, -y_shift, -x_shift, input_h, input_w)
        return image
