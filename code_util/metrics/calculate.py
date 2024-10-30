from code_util.data.read_save import read_medical_image
from code_util.util import is_finite_value
import pandas as pd
import os
import re
import torch
import numpy as np

def calculate(ct_path, sct_path, fun, mask_path=None, class_mask_path=None, device_id = None):
    """
    Calculate the metrics of two images.
    """
    if device_id == None:
        ct = read_medical_image(ct_path)
        sct = read_medical_image(sct_path)
        if mask_path:
            mask = read_medical_image(mask_path)
        else:
            mask = None
        if class_mask_path:
            class_mask = read_medical_image(class_mask_path)
        else:
            class_mask = None
    else:
        if device_id != -1:
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{device_id}')
            else:
                print(f'cuda:{device_id} is not avaliable, use cpu')
                device = torch.device('cpu')
        else:
            device = torch.device('cpu') 
        ct = torch.from_numpy(read_medical_image(ct_path)).to(device)
        sct = torch.from_numpy(read_medical_image(sct_path)).to(device)
        if mask_path:
            mask = torch.from_numpy(read_medical_image(mask_path)).to(device)
        else:
            mask = None
        if class_mask_path:
            class_mask = torch.from_numpy(read_medical_image(class_mask_path)).to(device)
        else:
            class_mask = None

    masks = [None]
    if not isinstance(class_mask,type(None)):
        class_masks = []
        if isinstance(class_mask,np.ndarray):
            unique_values = sorted(np.unique(class_mask))
            for val in unique_values:
                class_masks.append((class_mask == val).astype(np.uint8))
        elif isinstance(class_mask,torch.Tensor):
            unique_values = sorted(torch.unique(class_mask))
            for val in unique_values:
                class_masks.append((class_mask == val).int())
        if not isinstance(mask,type(None)):
            class_masks = [class_mask*mask for class_mask in class_masks]
        masks = class_masks
    elif not isinstance(mask,type(None)):
        masks = [mask]
    metrics = []
    if len(masks) == 1:
        Ls = [4024]
    else:
        Ls = [-250-(-1024),250-(-250),3000-250]
    for i,mask in enumerate(masks):
        parameters = {
            "ct": ct,
            "sct": sct,
            "L": Ls[i],
            "window_size": 7,
            "mask": mask
        }
        metric = fun(**parameters)
        if is_finite_value(metric):
            metrics.append(metric)
        else:
            metrics.append(0)
    # print(metrics)
    return metrics

def calculate_folder(data_folder, result_folder=None, mask_folder = None, class_mask_folder = None, metric_names = ["SSIM","PSNR"], device_id = None):
    if not result_folder:
        result_folder = os.path.dirname(data_folder)
    if device_id == None:
        from code_util.metrics.image_similarity_numpy import MSSIM_3D,MSE_3D,MAE_3D,PSNR_3D,RMSE_3D,SSIM_3D,Med_MSSIM_3D 
        metric_funs = {
        "MSSIM": MSSIM_3D,
        "SSIM": SSIM_3D,
        "PSNR": PSNR_3D,
        "MSE": MSE_3D,
        "MAE": MAE_3D,
        "RMSE": RMSE_3D,
        "Med_MSSIM": Med_MSSIM_3D
    }
    else:
        from code_util.metrics.image_similarity_torch import MSSIM_3D,MSE_3D,MAE_3D,PSNR_3D,RMSE_3D,SSIM_3D
        metric_funs = {
        "MSSIM": MSSIM_3D,
        "SSIM": SSIM_3D,
        "PSNR": PSNR_3D,
        "MSE": MSE_3D,
        "MAE": MAE_3D,
        "RMSE": RMSE_3D,
    }
    # 获取文件夹下所有文件名
    file_names = os.listdir(data_folder)
        
    # 定义字典用于存储每个序号下的文件路径
    file_paths = {}

    # 定义正则表达式来提取序号和类型
    # pattern = re.compile(r'2BA(\d+)_(real_A|fake_B|real_B|mask)\.nii\.gz')
    pattern = re.compile(r'2B(A|B|C)(\d+)_(real_A|fake_B|real_B)\.nii\.gz')
    # 遍历文件名
    for file_name in file_names:
        # 使用正则表达式提取信息
        match = pattern.match(file_name)
        if match:
            group, seq, file_type = match.groups()
            # seq, file_type = match.groups()

            # 创建一个键，形如 (序号, 类型)
            key = int(seq)

            # 根据类型存储文件路径
            if key not in file_paths:
                file_paths[key] = {}
            if file_type == 'real_A':
                file_paths[key]['cbct'] = os.path.join(data_folder, file_name)
            elif file_type == 'fake_B':
                file_paths[key]['sct'] = os.path.join(data_folder, file_name)
            elif file_type == 'real_B':
                file_paths[key]['ct'] = os.path.join(data_folder, file_name)
    
    pattern_mask = re.compile(r'2B(A|B|C)(\d+)\.nii\.gz')
    if mask_folder != None:
        mask_names = os.listdir(mask_folder)
        for mask_name in mask_names:
            match = pattern_mask.match(mask_name)
            if match:
                group, seq = match.groups()
                key = int(seq)
                file_paths[key]['mask'] = os.path.join(mask_folder, mask_name)

    pattern_class_mask = re.compile(r'2B(A|B|C)(\d+)\.nii\.gz')
    if class_mask_folder != None:
        class_mask_names = os.listdir(class_mask_folder)
        for class_mask_name in class_mask_names:
            match = pattern_class_mask.match(class_mask_name)
            if match:
                group, seq = match.groups()
                key = int(seq)
                file_paths[key]['class_mask'] = os.path.join(class_mask_folder, class_mask_name)

    metrics = {'Sequence and type': []}
    # 遍历每个序号并计算指标
    for seq, file_paths_dict in file_paths.items():
        cbct_path = file_paths_dict.get('cbct')
        ct_path = file_paths_dict.get('ct')
        sct_path = file_paths_dict.get('sct')
        mask_path = file_paths_dict.get('mask')
        class_mask_path = file_paths_dict.get('class_mask')

        metrics['Sequence and type'].append("cbct_ct" + "_" + str(seq))
        metrics['Sequence and type'].append("sct_ct" + "_" + str(seq))
        # 确保ct和sct文件都存在
        if ct_path and sct_path and cbct_path:
            print("Processing sequence", seq)
            print(cbct_path)
            print(ct_path)
            print(sct_path)
            print(mask_path)
            print(class_mask_path)

            # 计算指标
            for metric_name in metric_names:
               
                if mask_path != None:   
                    full_mask_metric_cbct = calculate(ct_path, cbct_path, metric_funs[metric_name], mask_path = mask_path, class_mask_path = None, device_id = device_id)
                    full_mask_metric_sct = calculate(ct_path, sct_path, metric_funs[metric_name], mask_path = mask_path, class_mask_path = None, device_id = device_id)
                    column_name = metric_name + "_" + "all" 
                    if column_name not in metrics.keys():
                        metrics[column_name] = [full_mask_metric_cbct[0]]
                        metrics[column_name].append(full_mask_metric_sct[0])
                    else:
                        metrics[column_name].append(full_mask_metric_cbct[0])
                        metrics[column_name].append(full_mask_metric_sct[0])
                if class_mask_path != None:
                    class_mask_metric_cbct = calculate(ct_path, cbct_path, metric_funs[metric_name], mask_path = mask_path, class_mask_path = class_mask_path, device_id = device_id)
                    class_mask_metric_sct = calculate(ct_path, sct_path, metric_funs[metric_name], mask_path = mask_path, class_mask_path = class_mask_path, device_id = device_id)
                    for (i, class_metric) in enumerate(class_mask_metric_cbct):
                        column_name = metric_name + "_" + str(i)
                        if column_name not in metrics.keys():
                            metrics[column_name] = [class_metric]
                        else:
                            metrics[column_name].append(class_metric)
                    for (i, class_metric) in enumerate(class_mask_metric_sct):
                        column_name = metric_name + "_" + str(i)
                        if column_name not in metrics.keys():
                            metrics[column_name] = [class_metric]
                        else:
                            metrics[column_name].append(class_metric)
    for key in metrics.keys():
        if key == 'Sequence and type':
            metrics[key].append("cbct_ct_mean")
            metrics[key].append("sct_ct_mean")
        else:
            cbct_metrics = metrics[key][::2]
            cbct_mean_metric = sum(cbct_metrics)/len(cbct_metrics)
            sct_metrics = metrics[key][1::2]
            sct_mean_metric = sum(sct_metrics)/len(sct_metrics)
            metrics[key].append(cbct_mean_metric)
            metrics[key].append(sct_mean_metric)

    # print(metrics)
    # Create DataFrame
    results_df = pd.DataFrame(metrics)

    # Print the average of the last two rows
    print(results_df.iloc[-2:])
    
    # 保存结果到CSV文件
    if device_id == None:
        cal_tool = "numpy"
    else:
        cal_tool = "torch"
    if mask_folder == None:
        mask_postfix = "wo_mask"
    else:
        mask_postfix = os.path.normpath(mask_folder).split(os.path.sep)[-3]
    if class_mask_folder == None:
        class_mask_postfix = "wo_class"
    else:
        class_mask_postfix = os.path.normpath(class_mask_folder).split(os.path.sep)[-3]
    metrics_file = 'metrics_results_%s_%s_%s_%s_L.csv' % (metric_names[0],cal_tool,mask_postfix,class_mask_postfix)
    results_df.to_csv(os.path.join(result_folder, metrics_file), index=False)

if __name__ == '__main__':
    data_folder = "/home/xdh/data/intelland/code/frameworks/InTransNet/file_result/CBCT2CT_pix2pix/test_latest/3D"
    result_folder = "/home/xdh/data/intelland/code/frameworks/InTransNet/file_result/CBCT2CT_pix2pix/test_latest"
    calculate_folder(data_folder, result_folder)
   
