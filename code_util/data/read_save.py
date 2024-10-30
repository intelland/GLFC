import SimpleITK as sitk
from PIL import Image
import numpy as np
from code_util.data.prepost_process import Postprocess
from code_util import util
import os
"""
READ

read_medical_image: 读取.nii.gz或.nii格式的医学图像 返回np.ndarray
read_natural_image: 读取.jpg或.png格式的自然图像 返回np.ndarray
get_image_params: 读取.nii.gz或.nii格式的医学图像的参数 返回dict

"""
def read_medical_image(path):
    """
    Read nii file and return the numpy array of the image.
    """
    image = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(image)
    # image_array = np.array(image_array)
    # print("image_array.shape:", image_array.shape)
    # print(type(image_array))
    return image_array

def read_natural_image(path):

    # PIL的Image按照(width,height)的方式读取图像 和cv2正好相反
    return np.array(Image.open(path).convert('RGB'))

def get_image_params(image_path:str):
    """
    Read image file by path and return the size of the image.
    """
    # determine medical or natural image
    if image_path.endswith(".nii.gz") or image_path.endswith(".nii"):
        
        image = sitk.ReadImage(image_path)
        
        image_params = dict()
        image_params = {
        'size' : np.array(image.GetSize()),
        'spacing': np.array(image.GetSpacing()),
        'origin': np.array(image.GetOrigin()),
        'direction': np.array(image.GetDirection())
    }
    else: 
        image = Image.open(image_path)
        image_params = {
            'size' : np.array(image.size)
        }
    return image_params

"""
SAVE

save_image_4_final 将输入图像做最后的储存 需要根据参考图像对其进行resize 然后决定储存为医学图像或是自然图像
save_image_4_show 将输入图像做临时储存 无需resize 总是保存为自然图像格式
write_nii 将输入图像保存为医学图像格式

"""

def save_image_4_final(image, ref_path,target_path, config):
    """
    Save a numpy image to the disk for final results
    """

    img_params = get_image_params(ref_path)
    size = tuple(np.flip(img_params['size']))
    transform = Postprocess(config,size)()
    image = transform(image)   
    image = util.tensor2np(image)
    if ref_path.endswith(".nii.gz") or ref_path.endswith(".nii"):
        write_nii(image, img_params, target_path)
    else: 
        write_jpg(image, img_params, target_path)

def save_image_4_show(image_numpy, image_path):
    """Save a numpy image to the disk for showing on the html page

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_test_image(visuals, img_paths, config, save_list = ["real_A", "real_B", "fake_B"]):
    target_path_base = os.path.join(config["work_dir"],"images")
    os.makedirs(target_path_base ,exist_ok=True)

    # 将测试结果全部保存到本地
    for label, image in visuals.items():
        if label in save_list:
            # 找到label对应的reference image
            if "A" in label:
                ref_path = img_paths['A_path'][0]
            else:
                ref_path = img_paths['B_path'][0]
            file_name = os.path.basename(ref_path)
            # 规定output image和reference image之间的名称关系
            if "cbct" in file_name:
                target_file_name = file_name.replace("slice_","").replace(".","_"+label+".",1) 
            else: 
                target_file_name = file_name.replace("slice_","").replace(".","_"+label+".",1) 
            target_path = os.path.join(target_path_base,target_file_name)
            # 根据指定的reference image和名称关系保存 output image
            save_image_4_final(image,ref_path,target_path,config)


def write_nii(image_array, image_params, nii_path):
    """
    Write nii file from numpy array and parameters.
    """
    if isinstance(image_array, np.ndarray) == False:
        image_array = np.array(image_array)

    size = np.flip(image_params['size'])
    if (image_array.shape != size).any():
        print("image_array.shape:", image_array.shape)
        print("size form the image:", size)
        raise ValueError('The size of the image is not the same as the size in the parameters.')
    space = np.squeeze(np.array(image_params['spacing']))
    origin = np.squeeze(np.array(image_params['origin']))
    direction = np.squeeze(np.array(image_params['direction']))
    # print("space:", space)
    # print("origin:", origin)
    # print("direction:", direction)
    image = sitk.GetImageFromArray(image_array)
    image.SetSpacing(space)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    sitk.WriteImage(image, nii_path)

def write_jpg(image_array, image_params, jpg_path):
    if isinstance(image_array, np.ndarray) == False:
        image_array = np.array(image_array)

    size = np.flip(image_params['size'])
    if (image_array.shape != size).any():
        print("image_array.shape:", image_array.shape)
        print("size form the image:", size)
        raise ValueError('The size of the image is not the same as the size in the parameters.')
    image_pil = Image.fromarray(image_array)
    image_pil.save(jpg_path)
