
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class Identity():
    def __call__(self,x):
        return x

class Preprocess():
    def __init__(self, config):
        self.config = config
        self.transform_list = []
        self.transform_list.append(transforms.ToTensor())
        self.contruct_pipeline()

    def contruct_pipeline(self):
        config = self.config
        # clip
        if config["preprocess"]["clip"] == True:
            clip_min,clip_max = config["preprocess"]["clip_range"]
            self.transform_list.append(ClipTransform(clip_min,clip_max))
        # resize
        if config["preprocess"]["resize"] == True:
            osize = [config["preprocess"]["resize_size"], config["preprocess"]["resize_size"]]
            method = get_resize_method(config)
            self.transform_list.append(resize(osize,method))
        # crop 
        if config["preprocess"]["crop"] == True:
            pos = get_crop_pos(config)
            crop_size = config["preprocess"]["crop_size"]
            self.transform_list.append(FixedCrop(pos,crop_size))
        # flip
        if config["preprocess"]["flip"] == True:
            flip_direction = get_flip_direction(config)
            self.transform_list.append(FixedFlip(flip_direction))
        # normalization
        if config["preprocess"]["manual_norm"] == True:
            min_val,max_val = config["preprocess"]["manual_norm_range"]
            self.transform_list.append(NormalizeFromRange(min_val,max_val))

    def __call__(self):
        return transforms.Compose(self.transform_list)
    
class Postprocess():
    def __init__(self, config, resize_range):
        self.config = config
        self.resize_range = resize_range
        self.transform_list = []
        self.contruct_pipeline()
    
    def contruct_pipeline(self):
        config = self.config
        # normalization
        if config["preprocess"]["manual_norm"] == True:
            min_val,max_val = config["preprocess"]["manual_norm_range"]
            self.transform_list.append(NormalizeToRange(min_val,max_val))
        # flip
        if config["preprocess"]["flip"] == True:
            flip_direction = get_flip_direction(config)
            self.transform_list.append(FixedFlip(flip_direction)) 
        # resize
        if config["preprocess"]["resize"] == True:
            osize = self.resize_range
            method = get_resize_method(config)
            self.transform_list.append(resize(osize,method))
    
    def __call__(self):
        return transforms.Compose(self.transform_list)
    
class Preprocess_class_mask():
    def __init__(self,config):
        self.config = config
        self.transform_list = []
        self.transform_list.append(transforms.ToTensor())
        self.contruct_pipeline()

    def contruct_pipeline(self):
        # normalization
        config = self.config
        if config["preprocess"]["resize"] == True:
            osize = [config["preprocess"]["resize_size"], config["preprocess"]["resize_size"]]
            method = transforms.InterpolationMode.NEAREST
            self.transform_list.append(resize(osize,method))
        
    def __call__(self):
        return transforms.Compose(self.transform_list)
    

"""CLIP"""

class ClipTransform:
    def __init__(self, min_value=0.0, max_value=1.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, img):
        return torch.clamp(img, self.min_value, self.max_value)    
        
"""RESIZE""" 

def resize(osize, method):
    return transforms.Resize(osize, method,antialias=None)

def get_resize_method(config):
    if config["preprocess"]["resize_method"] == 'BILINEAR':
        method = transforms.InterpolationMode.BILINEAR
    elif config["preprocess"]["resize_method"] == 'BICUBIC':
        method = transforms.InterpolationMode.BICUBIC
    return method

"""CROP"""

class FixedCrop():
    def __init__(self, pos, size):
        self.left, self.top = pos
        self.height = self.width = size

    def __call__(self, img):
        # print("============",self.top, self.left, self.height, self.width)
        return F.crop(img, self.top, self.left, self.height, self.width)
    
def get_crop_pos(config):
    
    crop_size = config["preprocess"]["crop_size"]
    w = h = config["preprocess"]["resize_size"]

    # 随机选取一个点作为crop的左上角
    x = random.randint(0, np.maximum(0, w - crop_size))
    y = random.randint(0, np.maximum(0, h - crop_size))

    return (x,y)

"""FLIP"""

class FixedFlip():
    def __init__(self, direction):
        self.direction = direction 
        if self.direction == 'v':
            self.flip = transforms.RandomVerticalFlip(p=1.0)
        elif self.direction == 'h':
            self.flip = transforms.RandomHorizontalFlip(p=1.0)
        else:
            self.flip = Identity()
        
    def __call__(self,img):
        return self.flip(img)


def get_flip_direction(config):
    flip_direction = config["preprocess"]["flip_direction"]
    p = random.random()
    assert len(flip_direction) <= 2, "wrong configuration of flip direction"
    if len(flip_direction) == 2:
        flip_p_v = flip_p_h = 0.25
    elif len(flip_direction) == 1:
        if flip_direction == 'v':
            flip_p_v = 0.5
            flip_p_h = 0
        elif flip_direction == 'h':
            flip_p_v = 0
            flip_p_h = 0.5
    else:
        flip_p_v = 0
        flip_p_h = 0
    if p <= flip_p_v:
        return 'v'
    elif p <= flip_p_v + flip_p_h:
        return 'h'
    else:
        return None
    
"""NORMALIZATION"""

class NormalizeFromRange:
    def __init__(self, min_val, max_val):
        """
        将输入图像的像素值由intensity_range线性映射到(-1,1)之间
        """
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img_tensor):
        return 2 * (img_tensor - self.min_val) / (self.max_val - self.min_val) - 1
    
class NormalizeToRange:
    def __init__(self, min_val, max_val):
        """
        将输入图像的像素值由(-1,1)线性映射到intensity_range之间
        """
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img_tensor):
        
        return (img_tensor + 1) / 2 * (self.max_val - self.min_val) + self.min_val 
