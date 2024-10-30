import os

import numpy as np
import torch

from code_dataset.base_dataset import BaseDataset
from code_dataset.image_folder import make_dataset
from code_util.data.prepost_process import Preprocess,Preprocess_class_mask

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, config):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, config)
           
        self.dir_A = os.path.join(config["dataset"]["dataroot"],  config["phase"]+"A")  # get the image directory
        self.dir_B = os.path.join(config["dataset"]["dataroot"],  config["phase"]+"B")
        self.A_paths = make_dataset(self.dir_A, config)  # get image paths
        self.B_paths = make_dataset(self.dir_B, config)  # get image paths
        if config["isTrain"] and config.get("MCL",{}).get("use_MCL",False) and config["MCL"].get("class_mask") == "prepared": 
            self.dir_class_mask = os.path.join(config["dataset"]["dataroot"], "mask" ,config["MCL"].get("class_mask_type"), "2D", config["phase"])
            self.class_mask_paths = make_dataset(self.dir_class_mask , config)
        else:
            self.class_mask_paths = []
        if config["preprocess"]["crop"] == True:
            assert(config["preprocess"]["resize_size"] >= config["preprocess"]["crop_size"])   # crop_size should be smaller than the size of loaded image

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths 
        """
        # read a image given a random integer index
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        assert os.path.basename(A_path) == os.path.basename(B_path), f"A_path ({A_path}) does not match B_path ({B_path})"
        # print(A_path)
        # print(B_path)
        A = self.read_image(A_path)
        B = self.read_image(B_path)
        
        
        # print(A.shape)
        # print(B.shape)
        # print("minmax of A: ",np.min(A),np.max(A))
        # print("minmax of B: ",np.min(B),np.max(B))
        transform = Preprocess(self.config)
        A_transform = B_transform = transform()

        A = A_transform(A)
        B = B_transform(B)
        # print(A.shape)
        # print(B.shape)
        # print(torch.max(A))
        # print(torch.min(A))
        # print(torch.min(B))
        # print(torch.max(B))

        class_mask = torch.tensor([])
        if self.class_mask_paths != []:
            class_mask_path = self.class_mask_paths[index]
            class_mask = self.read_image(class_mask_path)
            transform =  Preprocess_class_mask(self.config)
            class_mask_transform = transform()
            class_mask = class_mask_transform(class_mask) 

        return {'A': A, 'B': B, 'A_path': A_path, 'B_path': A_path, 'class_mask':class_mask}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
