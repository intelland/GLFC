"""
这个文件用于在实验结果中挑选效果好的样本
"""
import os
import re
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from code_util.data.read_save import read_medical_image
from code_util.data.convert import slice_nifti
from code_util.metrics.image_similarity_torch import SSIM_3D,PSNR_3D
from code_util.metrics.calculate import calculate


experiment_root = "./file_result/division2/stage4"
mask_root = "./file_dataset/SynthRAD2023/brain2/mask"
matplotlib.use('Agg')

def select_samples(experiment_names,mask_type,metric_method = "SSIM",test_epochs="latest",sample_type="images",sample_number = 10, window = [-1024,3000], device_id = 0):
    
    # mask
    mask_folder = os.path.join(mask_root,mask_type,"2D")

    # 选择计算的device
    if device_id != -1:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{device_id}')
        else:
            print(f'cuda:{device_id} is not avaliable, use cpu')
            device = torch.device('cpu')
    else:
        device = torch.device('cpu') 

    # 选择metric
    if metric_method == "SSIM":
        metric_fun = SSIM_3D
    elif metric_method == "PSNR":
        metric_fun = PSNR_3D
    else:
        metric_method = SSIM_3D

    # 根据experiment_list得到实验结果储存的路径
    if isinstance(test_epochs,str):
        test_epochs = [test_epochs for _ in experiment_names]
    result_paths = [os.path.join(experiment_root,experiment_name,"test_"+test_epoch,sample_type) for test_epoch,experiment_name in zip(test_epochs,experiment_names)]

    target_experiment_name,comparision_experiments_names = experiment_names[0],experiment_names[1:]
    target_experiment_path,comparision_experiments_path = result_paths[0],result_paths[1:]

    # 构造pattern
    pattern = re.compile(r'(2BA009_.*)_(real_A|fake_B|real_B|mask)\.nii\.gz')

    #匹配符合规格的sample
    target_file_names = os.listdir(target_experiment_path)
    target_identity_names = []
    for target_file_name in target_file_names:
        # 使用正则表达式提取信息
        match = pattern.match(target_file_name)
        if match:
            target_identity_name,target_file_type = match.groups()
            if int(target_identity_name.split("_")[-1]) > 100 and int(target_identity_name.split("_")[-1]) < 150:
                target_identity_names.append(target_identity_name)
    target_identity_names = list(set(target_identity_names))
    # 计算不同实验方法的结果
    ref_sample_type = "real_B"
    result_sample_type = "fake_B"
    
    metrics_all = []
    for target_identity_name in target_identity_names:
        metrics_row = []
        for result_path in result_paths:
            ref_sample_path = os.path.join(result_path,target_identity_name+"_"+ref_sample_type+".nii.gz")
            result_sample_path = os.path.join(result_path,target_identity_name+"_"+result_sample_type+".nii.gz")
            mask_path = os.path.join(mask_folder,target_identity_name+".nii.gz")
            metric = calculate(ref_sample_path, result_sample_path, metric_fun, mask_path = mask_path, class_mask_path = None, device_id = device_id)
            if isinstance(metric,list):
                metric = metric[0]
            metrics_row.append(metric)
        metrics_all.append(metrics_row)
    metrics_all = np.array(metrics_all)
    # 选择目标实验相比于对比实验最优的sample_number组样本

    # Calculate differences between the target experiment and comparison experiments
    target_metrics = metrics_all[:, 0]
    comparison_metrics = metrics_all[:, 1:]
    diff = target_metrics[:, None] - comparison_metrics
    best_diff = np.min(diff,axis=1)
    # Select the top 'sample_number' samples with the largest difference
    top_indices = np.argsort(best_diff)[-sample_number:]
    selected_samples = [target_identity_names[idx] for idx in top_indices]

    num_samples = len(top_indices)
    num_experiments = len(experiment_names)+1

    # 创建一个大图以并排显示所有图像
    fig, axes = plt.subplots(num_samples,num_experiments, figsize=(num_experiments*3,num_samples*3))

    for i in range(num_samples):
        for j in range(num_experiments):
            identity_name = target_identity_names[top_indices[i]]
            if j == num_experiments - 1:
                sample_path = os.path.join(result_paths[j-1],identity_name+"_real_B.nii.gz")
            else:
                sample_path = os.path.join(result_paths[j],identity_name+"_fake_B.nii.gz")
            sample_img = read_medical_image(sample_path)
            sample_img = np.clip(sample_img,*window)
            axes[i,j].imshow(sample_img,cmap='gray')
            axes[i,j].axis('off')
        
        axes[i,j].imshow(sample_img,cmap='gray')
        axes[i,j].axis('off')
            
    # 保存组合图像
    output_path = os.path.join("./",f"combined.png")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭图像以释放内存

    return selected_samples
    # 将这sample_number组样本绘制出来


import pandas as pd


def compare_metrics_from_csvfile(experiment_names, metric, metrics_file_name, test_epochs="latest"):
    if isinstance(test_epochs, str):
        test_epochs = [test_epochs for _ in experiment_names]
    
    result_paths = [os.path.join(experiment_root, experiment_name, "test_" + test_epoch) for test_epoch, experiment_name in zip(test_epochs, experiment_names)]

    metrics_data = {}
    
    # Find the metrics file in result_paths
    cbct = False
    for exp_name, path in zip(experiment_names, result_paths):
        metrics_file_path = os.path.join(path, metrics_file_name)
        
        if os.path.exists(metrics_file_path):
            # Read the CSV file
            df = pd.read_csv(metrics_file_path).iloc[:-2]
            selected_data = df.iloc[1::2]

            # 提取字符串索引中的数字部分，并将其转换为整数
            selected_data['numeric_index'] = selected_data['Sequence and type'].apply(lambda x: int(x.split('_')[-1]))

            # 根据数字索引排序
            selected_data = selected_data.sort_values(by='numeric_index', ascending=True).reset_index(drop=True)

            # 删除临时列并重置索引
            # df = df.drop(columns='numeric_index').reset_index(drop=True)
            
            metrics_data[exp_name] = selected_data[metric]
            if cbct == False:
                selected_data = df.iloc[::2]
                # 提取字符串索引中的数字部分，并将其转换为整数
                selected_data['numeric_index'] = selected_data['Sequence and type'].apply(lambda x: int(x.split('_')[-1]))

                # 根据数字索引排序
                selected_data = selected_data.sort_values(by='numeric_index', ascending=True).reset_index(drop=True)
                metrics_data['cbct'] = selected_data[metric]
                print(selected_data['numeric_index'])
                cbct = True
        else:
            print(f"Metrics file not found for experiment: {exp_name} at path: {metrics_file_path}")
    # Plotting the data
    plt.figure(figsize=(10, 6))
    
    for exp_name, data in metrics_data.items():
        plt.plot(data, label=exp_name)
    
    plt.xlabel('Epoch (selected even rows)')
    plt.ylabel(metric)
    plt.title(f'Comparison of {metric} across experiments')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to file
    output_path = os.path.join("./", f"compare_{metric}.png")
    plt.savefig(output_path)
    print(f"Comparison plot saved to: {output_path}")

def sample_slicing(experiment_names, identity_name, axis, position, save_path = "./", window_range=None,test_epochs="latest"):
    if isinstance(test_epochs, str):
        test_epochs = [test_epochs for _ in experiment_names]
    
    result_paths = [os.path.join(experiment_root, experiment_name, "test_" + test_epoch,"3D") for test_epoch, experiment_name in zip(test_epochs, experiment_names)]
    os.makedirs(save_path,exist_ok=True)
    sample_types = ["fake_B.nii.gz"]*len(result_paths) + ["real_B.nii.gz","real_A.nii.gz"]
    experiment_names = experiment_names + [experiment_names[-1]]*2
    result_paths = result_paths + [result_paths[-1]]*2
    for i,result_path in enumerate(result_paths):
        # 提取文件名（不带扩展名）作为保存的基础名称
        sample_type = sample_types[i]
        file_path = os.path.join(result_path,identity_name+"_"+sample_type)
        print(file_path)
        base_name = os.path.basename(file_path).split(".")[0]
        print(base_name)
        # 定义保存路径
        # save_png_path = os.path.join(save_path,experiment_names[i]+"_"+sample_type.split(".")[0]+".png")
        save_png_path = os.path.join(save_path,experiment_names[i]+"_"+sample_type.split(".")[0]+".nii.gz")
        # 也可以选择保存为 .nii.gz 格式，如果需要
        slice_nifti(file_path, axis, position, save_path=save_png_path, window_range=window_range)

if __name__ == "__main__":
    experiment_names = ["CBCT2CT_Unet_64_4", "CBCT2CT_swinUnet_8", "CBCT2CT_pix2pix_lsgan_unet64"]
    selected_samples = select_samples(experiment_names,mask_type="small", metric_method="SSIM", sample_number=5)
    