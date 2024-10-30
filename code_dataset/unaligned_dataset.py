import os
import random

from code_dataset.base_dataset import BaseDataset
from code_dataset.image_folder import make_dataset
from code_util.data.prepost_process import Preprocess



class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, config):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, config)
        if config["isTrain"] == True:
            phase = "train"
        else:
            phase = "test"

        self.dir_A = os.path.join(config["dataset"]["dataroot"], phase+"A")  # get the image directory
        self.dir_B = os.path.join(config["dataset"]["dataroot"], phase+"B")

        self.A_paths = make_dataset(self.dir_A, config)  # get image paths
        self.B_paths = make_dataset(self.dir_B, config)  # get image paths

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        if config["preprocess"]["crop"] == True:
            assert(config["preprocess"]["resize_size"] >= config["preprocess"]["crop_size"])   # crop_size should be smaller than the size of loaded image


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.config["dataset"]["dataloader"]["paired"]:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A = self.read_image(A_path)
        B = self.read_image(B_path)

        transform = Preprocess(self.config)
        A_transform = B_transform = transform()
        
        # apply image transformation
        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_path': A_path, 'B_path': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
