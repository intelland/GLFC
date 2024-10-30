"""
this file provides some tools for dataset analysis
"""

import os
import re
import SimpleITK as sitk
import matplotlib.pyplot as plt
from code_util.data.read_save import read_medical_image

def get_nii_value_range(folder_path):
    # 获取指定文件夹下所有文件夹的列表
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    min_of_all = float('inf')
    max_of_all = -float('inf')
    # 遍历每个文件夹
    for subfolder in subfolders:
        print(f"Processing folder: {subfolder}")
        
        # 获取当前文件夹下所有.nii文件的列表
        nii_files = [f.path for f in os.scandir(subfolder) if f.is_file() and (f.name.endswith('.nii') or f.name.endswith('.nii.gz'))]

        # 遍历每个.nii文件
        for nii_file in nii_files:
            # 读取.nii文件
            image = sitk.ReadImage(nii_file)

            # 获取体素值的范围
            min_value = sitk.GetArrayViewFromImage(image).min()
            max_value = sitk.GetArrayViewFromImage(image).max()

            if min_value < min_of_all:
                min_of_all = min_value
            if max_value > max_of_all:
                max_of_all = max_value

            # 输出结果
            print(f"  {os.path.basename(nii_file)} - Value Range: ({min_value}, {max_value})")
    
    # 输出结果
    print(f"Value Range of All Slices: ({min_of_all}, {max_of_all})")


def get_slice_value_range(nii_file_path):
    # 读取.nii文件
    image = sitk.ReadImage(nii_file_path)

    # 获取.nii文件的数组表示
    image_array = sitk.GetArrayFromImage(image)

    # 获取体素值范围
    min_values = []
    max_values = []

    #
    min_of_all = image_array.min()
    max_of_all = image_array.max()

    # 遍历每个切片
    for slice_index in range(image_array.shape[0]):
        # 获取当前切片的体素值范围
        min_value = image_array[slice_index, :, :].min()
        max_value = image_array[slice_index, :, :].max()

        # 添加到列表中
        min_values.append(min_value)
        max_values.append(max_value)

        # 输出结果
        print(f"Slice {slice_index + 1} - Value Range: ({min_value}, {max_value})")

    # 输出结果
    print(f"Value Range of All Slices: ({min_of_all}, {max_of_all})")
    return min_values, max_values

def plot_file_HU_histograms(nii_file_path):
    # 读取.nii文件
    image = sitk.ReadImage(nii_file_path)

    # 获取.nii文件的数组表示
    image_array = sitk.GetArrayFromImage(image)

    # 扁平化数组以获取所有体素值
    flattened_array = image_array.flatten()

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(flattened_array, bins=100, color='blue', alpha=0.7)
    plt.title('HU Value Distribution')
    plt.xlabel('HU Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def count_slices_from_3d(path_3d, axis):
    image_array = read_medical_image(path_3d)
    
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis in axis_map:
        return image_array.shape[axis_map[axis]]
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

def count_slices_from_3d_dir(dir_path, pattern, axis='x'):
    total_slices = 0
    regex = re.compile(pattern)
    for filename in os.listdir(dir_path):
        if regex.match(filename):
            path_3d = os.path.join(dir_path, filename)
            total_slices += count_slices_from_3d(path_3d, axis)
    return total_slices


if __name__ == '__main__':
    # 使用示例
    # file_path = '/home/xdh/data/intelland/datasets/SynthRAD2023/original/Task2/brain/division2/train/2BA002/ct.nii.gz'
    # get_slice_value_range(file_path)
    # plot_file_HU_histograms(file_path)

    # # 使用示例
    folder_path = '/home/xdh/data/intelland/datasets/SynthRAD2023/original/Task2/brain/division2/train'
    get_nii_value_range(folder_path)    


