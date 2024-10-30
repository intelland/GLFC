"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import random
from PIL import Image
import os

VALID_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF', '.nii', '.nii.gz', 
]

def is_valid_file(filepath):
    return any(filepath.endswith(extension) for extension in VALID_EXTENSIONS) 
    # 这里不能用 os.oath.isfile()检验 .nii.gz文件 是不会检测为文件的

def make_dataset(dir, config:dict):
    if "max_size" in config["dataset"].keys():
        max_dataset_size = config["dataset"]["max_size"]
    else:
        max_dataset_size = float('inf')
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    images = []
    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        if is_valid_file(filename):
            images.append(filepath)
    images.sort()
    data_len = len(images)
    if config["dataset"]["random_sample"] == True and max_dataset_size < data_len:
        random_seed = config["random_seed"]
        random.seed(random_seed)
        numbers = list(range(data_len))
        sampled_numbers = random.sample(numbers, max_dataset_size)
    else:
        sampled_numbers = range(data_len)
        
    images = [images[i] for i in sampled_numbers if i < data_len]
    images.sort()
    return images

def default_loader(path):
    return Image.open(path).convert('RGB')
