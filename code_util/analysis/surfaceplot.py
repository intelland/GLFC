import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def surface(images: list, legends: list = []):
    """
    绘制多个2D图像的强度表面图到同一张对比图上。
    
    :param images: 由np.ndarray组成的list，每个元素为2D图像。
    :param legends: 图例列表，若提供则添加到图中。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置x和y坐标
    x = np.linspace(0, 1, images[0].shape[1])
    y = np.linspace(0, 1, images[0].shape[0])
    X, Y = np.meshgrid(x, y)

    for idx, img in enumerate(images):
        # 使用 z 值为图像强度
        ax.plot_surface(X, Y, img, alpha=1, label=legends[idx] if legends else None, cmap = "viridis")

    if legends:
        # 添加图例
        ax.legend()
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Intensity')

    ax.set_zlim((-1024,3000))
    plt.show()