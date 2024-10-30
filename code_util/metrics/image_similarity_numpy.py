import numpy as np

def SSIM_3D(ct: np.array, sct: np.array, mask = None, L: float = 4024.0, **kwargs):
    """
    Calculate the SSIM of two 3D images.
    """
    K1 = 0.01
    K2 = 0.03
    ct = ct.astype(np.float32)
    sct = sct.astype(np.float32)
    if ct.shape != sct.shape:
        raise ValueError('The shapes of the images are not the same.')
    if ct.ndim != 3:
        raise ValueError('The dimension of the images is not 3.')
    if isinstance(mask,np.ndarray):
        ct = ct[mask > 0.5] + 1024
        sct = sct[mask > 0.5] + 1024
    ct_mean = np.mean(ct)
    sct_mean = np.mean(sct)
    ct_std = np.std(ct)
    sct_std = np.std(sct)
    ct_sct_cov = np.mean((ct - ct_mean) * (sct - sct_mean))
    c1 = (K1 * L) ** 2
    c2 = (K2 * L) ** 2

    ssim_numerator = (2 * ct_mean * sct_mean + c1) * (2 * ct_sct_cov + c2)
    ssim_denominator = (ct_mean ** 2 + sct_mean ** 2 + c1) * (ct_std ** 2 + sct_std ** 2 + c2)
    ssim = ssim_numerator / ssim_denominator

    return ssim

def MSSIM_3D(ct:np.array,sct:np.array,mask:np.array=None, kernel_size = 7, L = 4024, **kwargs):
    """
    Calculate the mean SSIM of two 3D images.
    """
    # Check if the shapes of the images are the same
    if not isinstance(mask,np.ndarray):
        mask = np.ones_like(ct)
    if ct.shape != sct.shape or ct.shape != mask.shape or sct.shape != mask.shape:
        raise ValueError('The shapes of the images are not the same.')
    # Check if the dimension of the images is 3
    if ct.ndim != 3:
        raise ValueError('The dimension of the images is not 3.')
    # iterate over the point in the mask
    ssim = np.zeros_like(ct)
    ct = ct.astype(np.float32)
    sct = sct.astype(np.float32)
    
    for i in range(mask.shape[0]):
        print("i:",i)
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i,j,k] > 0.5:
                    # 以i,j,k为中心的kernel_size*kernel_size*kernel_size的cube
                    # 判断是否越界
                    ct_cube = get_cube(ct,i,j,k,kernel_size)
                    sct_cube = get_cube(sct,i,j,k,kernel_size)
                    mask_cude = get_cube(mask,i,j,k,kernel_size)
                    ssim[i,j,k] = SSIM_3D(ct_cube,sct_cube,mask_cude,L)
    if np.sum(mask) != 0:
        output = np.sum(ssim) / np.sum(mask)
    else:
        output = 1
    return output

def Med_MSSIM_3D(ct:np.array,sct:np.array,mask:np.array=None, kernel_size = 7, L = 4024, **kwargs):
    """
    Calculate the mean SSIM of two 3D images.
    """
    # Check if the shapes of the images are the same
    if not isinstance(mask,np.ndarray):
        mask = np.ones_like(ct)
    if ct.shape != sct.shape or ct.shape != mask.shape or sct.shape != mask.shape:
        raise ValueError('The shapes of the images are not the same.')
    # Check if the dimension of the images is 3
    if ct.ndim != 3:
        raise ValueError('The dimension of the images is not 3.')
    # iterate over the point in the mask
    ssim = np.zeros_like(ct)
    ct = ct.astype(np.float32)
    sct = sct.astype(np.float32)
    
    for i in range(mask.shape[0]):
        print("i:",i)
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i,j,k] > 0.5:
                    # 以i,j,k为中心的kernel_size*kernel_size*kernel_size的cube
                    # 判断是否越界
                    ct_cube = get_cube(ct,i,j,k,kernel_size)
                    sct_cube = get_cube(sct,i,j,k,kernel_size)
                    mask_cude = get_cube(mask,i,j,k,kernel_size)
                    L = np.max(ct_cube) - np.min(ct_cube)
                    ssim[i,j,k] = SSIM_3D(ct_cube,sct_cube,mask_cude,L)
    if np.sum(mask) != 0:
        output = np.sum(ssim) / np.sum(mask)
    else:
        output = 1
    return output
    
def get_cube(img:np.array,i:int,j:int,k:int,kernel_size:int):
    """
    Get the cube with the center at (i,j,k) in the image.
    """
    # 做越界的截断
    i_start = max(0,i - kernel_size // 2)
    i_end = min(img.shape[0],i + kernel_size // 2 + 1)
    j_start = max(0,j - kernel_size // 2)
    j_end = min(img.shape[1],j + kernel_size // 2 + 1)
    k_start = max(0,k - kernel_size // 2)
    k_end = min(img.shape[2],k + kernel_size // 2 + 1)
    
    return img[i_start:i_end,j_start:j_end,k_start:k_end]


def PSNR_3D(ct:np.array,sct:np.array,mask:np.array=None, L = 4024, **kwargs):
    """
    Calculate the PNSR of two 3D images.
    """
    if ct.shape != sct.shape:
        raise ValueError('The shapes of the images are not the same.')
    if ct.ndim != 3:
        raise ValueError('The dimension of the images is not 3.')
    ct = ct.astype(np.float32)
    sct = sct.astype(np.float32)
    if mask is not None:
        ct = ct[mask > 0.5]
        sct = sct[mask > 0.5]
    mse = np.mean((ct - sct) ** 2)
    psnr = 10 * np.log10(L ** 2 / mse)
    return psnr


def MSE_3D(ct:np.array,sct:np.array,mask:np.array=None, **kwargs):
    """
    Calculate the MSE of two 3D images.
    """
    if ct.shape != sct.shape:
        raise ValueError('The shapes of the images are not the same.')
    if ct.ndim != 3:
        raise ValueError('The dimension of the images is not 3.')
    
    ct = ct.astype(np.float32)
    sct = sct.astype(np.float32)
    if mask is not None:
        ct = ct[mask > 0.5]
        sct = sct[mask > 0.5]
    mse = np.mean((ct - sct) ** 2)
    return mse

def MAE_3D(ct:np.array,sct:np.array,mask:np.array=None, **kwargs):
    """
    Calculate the MAE of two 3D images.
    """
    if ct.shape != sct.shape:
        raise ValueError('The shapes of the images are not the same.')
    if ct.ndim != 3:
        raise ValueError('The dimension of the images is not 3.')
    ct = ct.astype(np.float32)
    sct = sct.astype(np.float32)
    if mask is not None:
        ct = ct[mask > 0.5]
        sct = sct[mask > 0.5]
    mae = np.mean(np.abs(ct - sct))
    return mae

def RMSE_3D(ct:np.array,sct:np.array,mask:np.array=None, **kwargs):
    """
    Calculate the RMSE of two 3D images.
    """
    if ct.shape != sct.shape:
        raise ValueError('The shapes of the images are not the same.')
    if ct.ndim != 3:
        raise ValueError('The dimension of the images is not 3.')
    ct = ct.astype(np.float32)
    sct = sct.astype(np.float32) 
    if mask is not None:
        ct = ct[mask > 0.5]
        sct = sct[mask > 0.5]
    # print(mask)
    # print(ct.shape)
    rmse = np.sqrt(np.mean((ct - sct) ** 2))
    return rmse
