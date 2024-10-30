import os
import re
from collections import defaultdict
import SimpleITK as sitk

def recontruct_from_twoDs(modality, threeD_id, files, output_dir):
    """
    处理每个分组的文件列表。

    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Processing group {modality, threeD_id}:")
    slices = []
    # print(files)
    for f in files:
        # print(f" - {f}")
        slice_image = sitk.ReadImage(f)
        slices.append(slice_image)

    stacked_image = sitk.JoinSeries(slices)
    output_filename = f'{threeD_id}_{modality}' + '.nii.gz'
    output_path = os.path.join(output_dir, output_filename)
    sitk.WriteImage(stacked_image, output_path)
    print(f'Saved {output_path}')

def find_and_process_files(input_dir, output_dir, pattern):
    """
    查找并处理匹配给定正则表达式模式的文件。
    
    :param directory: 需要查找文件的目录
    :param pattern: 用于匹配文件名的正则表达式模式
    """
    
    # 获取目录下的所有文件
    files = os.listdir(input_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建一个字典用于分组存储匹配的文件
    grouped_files = defaultdict(lambda: defaultdict(list))
    
    # 遍历所有文件并使用正则表达式进行匹配
    for f in files:
        match = re.match(pattern, f)
        if match:
            grouped_files[match.group(3)][match.group(1)].append(0)
    for f in files:
        match = re.match(pattern, f)
        if match:
            f = os.path.join(input_dir,f)
            grouped_files[match.group(3)][match.group(1)][int(match.group(2))] = f

    # 对每个分组进行处理
    for modality, threeDs in grouped_files.items():
        # if modality == "fake_B":
        if 1:
            for threeD_id, twoDs in threeDs.items():
                recontruct_from_twoDs(modality, threeD_id, twoDs, output_dir)

if __name__ == "__main__":
    input_dir = "/home/xdh/data/intelland/code/frameworks/InTransNet/file_result/CBCT2CT_pix2pix/test_latest/images"
    output_dir = "/home/xdh/data/intelland/code/frameworks/InTransNet/file_result/CBCT2CT_pix2pix/test_latest/3D"    
    pattern = r"^(.+)_(\d+)_(.+)\.nii\.gz$"
    
    find_and_process_files(input_dir, output_dir, pattern)
