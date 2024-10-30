
import torch.nn as nn

from code_network.modules.unet import Up,Down,DoubleConv,OutConv
from code_network.modules.transformer import TransformerLayer
from code_network.modules.mamba import MambaLayer
from code_network.modules.general import Identity

from code_network.resnet import ResnetBlock


class XEUNet(nn.Module):
    # X enhanced U-Net
    def __init__(self, input_nc, output_nc, ngf = 64, down_step = 4, x_blocks = [32,16,8,4], x_residual = True, bilinear=False, **kwargs):
        super().__init__()
        self.n_channels = input_nc
        self.n_classes = output_nc
        self.bilinear = bilinear
        self.down_step = down_step
        
        self.inc = (DoubleConv(input_nc, ngf))
        self.skips = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        factor = 2 if bilinear else 1
        for i in range(down_step):
            self.downs.append(Down(ngf*2**i, ngf*2**(i+1)))
            self.ups.append(Up(ngf*2**(down_step-i), ngf*2**(down_step-i-1) // factor, bilinear))
        self.outc = nn.Sequential(OutConv(ngf, output_nc), nn.Tanh())

    def forward(self, x):
        x = self.inc(x)
        x_skips = []
        for i in range(self.down_step):
            x_skips.append(self.skips[i](x))
            x = self.downs[i](x)
        for i in range(self.down_step):
            x = self.ups[i](x,x_skips[self.down_step-i-1])
        result = self.outc(x)
        return result
    
class REUNet(XEUNet):
    # ResNet enhanced U-Net
    def __init__(self, input_nc, output_nc, ngf = 64, down_step = 4, res_blocks = [32,16,8,4], res_residual = True, bilinear=False, **kwargs):
        super().__init__(input_nc, output_nc, ngf, down_step, res_blocks, res_residual, bilinear)
        self.skips = nn.ModuleList()
        for i in range(down_step):
            if res_blocks[i] != 0:
                self.skips.append(nn.Sequential(*[ResnetBlock(ngf*2**i) for _ in range(res_blocks[i])]))
            else:
                self.skips.append(Identity())

class MEUNet(XEUNet):
    # Mamba enhanced U-Net
    def __init__(self, input_nc, output_nc, ngf = 64, down_step = 4, mamba_blocks = [32,16,8,4], patch_sizes = [8,4,2,1], mamba_residual = True, bilinear=False, **kwargs):
        super().__init__(input_nc, output_nc, ngf, down_step, mamba_blocks, mamba_residual, bilinear)
        self.skips = nn.ModuleList()
        for i in range(down_step):
            if mamba_blocks[i] != 0:
                self.skips.append(MambaLayer(ngf*2**i, mamba_blocks[i], patch_sizes[i], mamba_residual))
            else:
                self.skips.append(Identity())

class TEUNet(XEUNet):
    # Transformer enhanced U-Net
    def __init__(self, input_nc, output_nc, ngf = 64, down_step = 4, transformer_depths = [32,16,8,4], patch_sizes = [8,4,2,1], transformer_residual = True, bilinear=False, **kwargs):
        super().__init__(input_nc, output_nc, ngf, down_step, transformer_depths, transformer_residual, bilinear)
        self.skips = nn.ModuleList()
        for i in range(down_step):
            if transformer_depths[i] != 0:
                self.skips.append(TransformerLayer(ngf*2**i, transformer_depths[i], image_size=256 ,patch_size = patch_sizes[i], residual= transformer_residual))
            else:
                self.skips.append(Identity())