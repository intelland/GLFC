"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
from torch.utils.data import DataLoader, random_split
from torch import default_generator,Generator
from code_dataset.base_dataset import BaseDataset
from code_util.util import seed_worker

def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """

    """
    根据dataset_name引入其对应的file
    """
    dataset_filename = "code_dataset." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    """
    从file中引入对应的dataset类 以遍历其中的所有类并匹配类名的方式
    因此一个被指定的dataset: filename和classname均需要保持对应的格式
    """
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        """
        满足两个条件就引入该类作为数据集类
        1. 和opt中指定的类名匹配
        2. 是BaseDataset的子类
        """
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    """
    否则就不引入 并声称配置或者数据集类编写出错
    """
    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(config):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    if config["dataset"]["random_sample"] == True:
        seed = config["random_seed"]
        generator = Generator().manual_seed(seed)
    else:
        generator = default_generator  
    dataset_class = find_dataset_using_name(config["dataset"]["dataset_mode"])
    
     
    if config["isTrain"] == True:
        config["phase"] = "validation"
        val_dataset = dataset_class(config)
        config["phase"] = "train"
        train_dataset = dataset_class(config)
        print("training set size: %d, validation set size: %d" % (len(train_dataset), len(val_dataset)))

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config["dataset"]["dataloader"]["batch_size"],
            shuffle = config["dataset"]["dataloader"]["shuffle"],
            num_workers = int(config["dataset"]["dataloader"]["num_workers"]),
            worker_init_fn=seed_worker,
            generator=generator)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size = 1,
            shuffle = False,
            num_workers = 1,
            worker_init_fn=seed_worker,
            generator=generator)
        return train_dataloader, val_dataloader, (len(train_dataset),len(val_dataset))
    else:
        test_dataset = dataset_class(config)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle = False,
            num_workers = 1,
            worker_init_fn=seed_worker,
            generator=generator)
        return test_dataloader, len(test_dataset)
