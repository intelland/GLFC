import torch
import torch.nn as nn

from code_network.modules.general import ContinusParalleConv
from code_network.modules.unet import Up,Down,DoubleConv,OutConv

""" Parts of the U-Net model """

class Unet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf = 64, down_step = 4, bilinear=False, **kwargs):
        super(Unet, self).__init__()
        self.n_channels = input_nc
        self.n_classes = output_nc
        self.bilinear = bilinear
        self.down_step = down_step
        
        self.inc = (DoubleConv(input_nc, ngf))
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
            x_skips.append(x)
            x = self.downs[i](x)
        for i in range(self.down_step):
            x = self.ups[i](x,x_skips[self.down_step-i-1])
        result = self.outc(x)
        return result

class UnetPlusPlus(nn.Module):
    def __init__(self, input_nc, output_nc, deep_supervision=False, norm = "batch", **kwargs):
        super(UnetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision
        self.filters = [64, 128, 256, 512, 1024]
        
        self.CONV3_1 = ContinusParalleConv(512*2, 512, pre_Norm = True, norm = norm)
 
        self.CONV2_2 = ContinusParalleConv(256*3, 256, pre_Norm = True, norm = norm)
        self.CONV2_1 = ContinusParalleConv(256*2, 256, pre_Norm = True, norm = norm)
 
        self.CONV1_1 = ContinusParalleConv(128*2, 128, pre_Norm = True, norm = norm)
        self.CONV1_2 = ContinusParalleConv(128*3, 128, pre_Norm = True, norm = norm)
        self.CONV1_3 = ContinusParalleConv(128*4, 128, pre_Norm = True, norm = norm)
 
        self.CONV0_1 = ContinusParalleConv(64*2, 64, pre_Norm = True, norm = norm)
        self.CONV0_2 = ContinusParalleConv(64*3, 64, pre_Norm = True, norm = norm)
        self.CONV0_3 = ContinusParalleConv(64*4, 64, pre_Norm = True, norm = norm)
        self.CONV0_4 = ContinusParalleConv(64*5, 64, pre_Norm = True, norm = norm)
 
 
        self.stage_0 = ContinusParalleConv(input_nc, 64, pre_Norm = False, norm = norm)
        self.stage_1 = ContinusParalleConv(64, 128, pre_Norm = False, norm = norm)
        self.stage_2 = ContinusParalleConv(128, 256, pre_Norm = False, norm = norm)
        self.stage_3 = ContinusParalleConv(256, 512, pre_Norm = False, norm = norm)
        self.stage_4 = ContinusParalleConv(512, 1024, pre_Norm = False, norm = norm)
 
        self.pool = nn.MaxPool2d(2)
    
        self.upsample_3_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1) 
 
        self.upsample_2_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1) 
        self.upsample_2_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1) 
 
        self.upsample_1_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) 
        self.upsample_1_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) 
        self.upsample_1_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) 
 
        self.upsample_0_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
        self.upsample_0_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
        self.upsample_0_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
        self.upsample_0_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
 
        
        # 分割头
        self.final_super_0_1 = nn.Sequential(
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, output_nc, 3, padding=1),
          nn.Tanh()
        )        
        self.final_super_0_2 = nn.Sequential(
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, output_nc, 3, padding=1),
          nn.Tanh()
        )        
        self.final_super_0_3 = nn.Sequential(
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, output_nc, 3, padding=1),
          nn.Tanh()
        )        
        self.final_super_0_4 = nn.Sequential(
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, output_nc, 3, padding=1),
          nn.Tanh()
        )        
 
        
    def forward(self, x):
        x_0_0 = self.stage_0(x)
        x_1_0 = self.stage_1(self.pool(x_0_0))
        x_2_0 = self.stage_2(self.pool(x_1_0))
        x_3_0 = self.stage_3(self.pool(x_2_0))
        x_4_0 = self.stage_4(self.pool(x_3_0))
        
        x_0_1 = torch.cat([self.upsample_0_1(x_1_0) , x_0_0], 1)
        x_0_1 =  self.CONV0_1(x_0_1)
        
        x_1_1 = torch.cat([self.upsample_1_1(x_2_0), x_1_0], 1)
        x_1_1 = self.CONV1_1(x_1_1)
        
        x_2_1 = torch.cat([self.upsample_2_1(x_3_0), x_2_0], 1)
        x_2_1 = self.CONV2_1(x_2_1)
        
        x_3_1 = torch.cat([self.upsample_3_1(x_4_0), x_3_0], 1)
        x_3_1 = self.CONV3_1(x_3_1)
 
        x_2_2 = torch.cat([self.upsample_2_2(x_3_1), x_2_0, x_2_1], 1)
        x_2_2 = self.CONV2_2(x_2_2)
        
        x_1_2 = torch.cat([self.upsample_1_2(x_2_1), x_1_0, x_1_1], 1)
        x_1_2 = self.CONV1_2(x_1_2)
        
        x_1_3 = torch.cat([self.upsample_1_3(x_2_2), x_1_0, x_1_1, x_1_2], 1)
        x_1_3 = self.CONV1_3(x_1_3)
 
        x_0_2 = torch.cat([self.upsample_0_2(x_1_1), x_0_0, x_0_1], 1)
        x_0_2 = self.CONV0_2(x_0_2)
        
        x_0_3 = torch.cat([self.upsample_0_3(x_1_2), x_0_0, x_0_1, x_0_2], 1)
        x_0_3 = self.CONV0_3(x_0_3)
        
        x_0_4 = torch.cat([self.upsample_0_4(x_1_3), x_0_0, x_0_1, x_0_2, x_0_3], 1)
        x_0_4 = self.CONV0_4(x_0_4)
    
    
        if self.deep_supervision:
            out_put1 = self.final_super_0_1(x_0_1)
            out_put2 = self.final_super_0_2(x_0_2)
            out_put3 = self.final_super_0_3(x_0_3)
            out_put4 = self.final_super_0_4(x_0_4)
            return [out_put1, out_put2, out_put3, out_put4]
        else:
            out_put4 = self.final_super_0_4(x_0_4)
            return out_put4
    

