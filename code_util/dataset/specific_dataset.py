"""this file provide operations for some specific dataset 

1. MICCAI SynthRAD2023 Task2 brain 
"""

import os
import SimpleITK as sitk
import re
import random
import shutil
from code_util.dataset.prepare import split_3d_to_2d
from code_util.dataset.prepare import generate_mask_with_class_by_histogram,generate_mask_with_class_by_range
from code_util.dataset.analysis import count_slices_from_3d

def convert_SynthRAD2023_Task_2(data_root = "../datasets/Task2/brain/", output_dir = "./datasets/SynthRAD2023_cycleGAN/brain", mode = "train"):
    slice_num = 1000
    pattern = "^2B[A-C]\d{3}$"
    
    # 遍历data_root下所有以"2B"开头的文件夹
    data_root = os.path.join(data_root,mode)
    for folder_name in os.listdir(data_root):
        if re.match(pattern,folder_name):
            # 构造cbct和ct文件的路径
            cbct_path = os.path.join(data_root, folder_name, "cbct.nii.gz")
            ct_path = os.path.join(data_root, folder_name, "ct.nii.gz")

            # 选择cbct或ct，如果文件存在的话
            if os.path.exists(cbct_path) and os.path.exists(ct_path):
                pass 
            else:
                continue  # 如果cbct和ct不存在，跳过当前文件夹

            # 读取cbct或ct的z轴维度
            image = sitk.ReadImage(cbct_path)
            size = image.GetSize()[2]
        
            # 如果z轴维度大于slice_num，随机选取slice_num个索引
            if size > slice_num:
                split_list = random.sample(range(size), slice_num)
            else:
                split_list = list(range(size))

            
            # 将文件夹名作为前缀，分别对cbct和ct调用函数split_3d_to_2d
            print("process %s ......" % folder_name)
            prefix = folder_name + "_"
            split_3d_to_2d(cbct_path,os.path.join(output_dir,mode+"A"), split_list, prefix)
            split_3d_to_2d(ct_path, os.path.join(output_dir,mode+"B"), split_list, prefix)

def generate_class_mask_SynthRAD2023_Task_2(data_root = "../datasets/Task2/brain/", output_dir = "./datasets/SynthRAD2023/brain2",isTrain=True,method = "histogram",class_range = None):
    pattern = "^2B[A-C]\d{3}$"
    os.makedirs(output_dir,exist_ok=True)
    # 遍历data_root下所有以"2B"开头的文件夹
    for folder_name in os.listdir(data_root):
        if re.match(pattern,folder_name):
            # 构造ct文件的路径
            ct_path = os.path.join(data_root, folder_name, "ct.nii.gz")
            # 将文件夹名作为前缀，分别对cbct和ct调用函数split_3d_to_2d
            print("process %s ......" % folder_name)
            if method == "histogram":
                generate_mask_with_class_by_histogram(ct_path,output_dir,2,output_file_name = folder_name)
            elif method == "range":
                generate_mask_with_class_by_range(ct_path,output_dir,class_range=class_range,output_file_name = folder_name)

def prepare_mask_SynthRAD2023_Task_2(src_dir, dest_dir):
    # 正则表达式匹配模式
    pattern = re.compile(r'^2B[A-C]\d{3}$')
    
    # 确保目标文件夹存在
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # 遍历源文件夹
    for folder_name in os.listdir(src_dir):
        folder_path = os.path.join(src_dir, folder_name)
        
        # 检查是否为文件夹且名称符合模式
        if os.path.isdir(folder_path) and pattern.match(folder_name):
            # 找到文件夹中的 mask.nii.gz 文件
            mask_file_path = os.path.join(folder_path, 'mask.nii.gz')
            if os.path.exists(mask_file_path):
                # 生成新的文件名
                new_file_name = f"{folder_name}.nii.gz"
                new_file_path = os.path.join(dest_dir, new_file_name)
                
                # 复制并重命名文件
                shutil.copy(mask_file_path, new_file_path)
                print(f"Copied and renamed {mask_file_path} to {new_file_path}")
            
def count_slices_from_3d_dir_SynthRAD2023_Task_2(dir_path, pattern, axis='x'):
    total_slices = 0
    regex = re.compile(pattern)
    
    for foldername in os.listdir(dir_path):
        folder_path = os.path.join(dir_path, foldername)
        if os.path.isdir(folder_path) and regex.match(foldername):
            ct_file_path = os.path.join(folder_path, 'ct.nii.gz')
            if os.path.isfile(ct_file_path):
                total_slices += count_slices_from_3d(ct_file_path, axis)
    
    return total_slices

if __name__ == "__main__":
    convert_SynthRAD2023_Task_2(data_root= "/home/xdh/data/intelland/code/datasets/Task2/brain", output_dir= "/home/xdh/data/intelland/code/frameworks/InTransNet/file_dataset/SynthRAD2023/brain",isTrain=False)
    generate_class_mask_SynthRAD2023_Task_2(data_root= "/home/xdh/data/intelland/code/datasets/Task2/brain/division2/", output_dir= "/home/xdh/data/intelland/code/frameworks/InTransNet/file_dataset/SynthRAD2023/brain2/mask/3D",isTrain=True)