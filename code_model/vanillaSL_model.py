import torch
import numpy as np

from .base_model import BaseModel
from code_network import define_network
from code_network.tools.loss import MCLLoss, RPLoss, give_loss_by_name
from code_util.model.network import clip_grad

class vanillaSLModel(BaseModel):
    
    def __init__(self, config):

        BaseModel.__init__(self, config)
        # torch.autograd.set_detect_anomaly = True
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1']
        # self.loss_names = ['G_L1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']
        # define networks
        self.netG = define_network(config, net_type = "g")
        self.config = config
        
        if self.isTrain:
            # define loss functions
            lr = config["network"]["lr"]
            beta1 = config["network"]["beta1"] 
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL1_sum = torch.nn.L1Loss(reduction='sum')
            self.use_MCL = config.get("MCL",{}).get("use_MCL",False)
            self.use_PRL = config.get("PRL",{}).get("use_PRL",False)
            self.use_grad_clip = config.get("grad_clip",{}).get("use_grad_clip",False)

            if self.use_MCL == True:
                self.class_mask = config["MCL"].get("class_mask")    
                self.criterionMCL = MCLLoss(class_mask_range = config["MCL"]["class_mask_range"], class_weight = config["MCL"]["class_weight"], class_norm = config["MCL"]["class_norm"])
                self.loss_names = self.loss_names + ['G_L1_0','G_L1_1','G_L1_2']
                self.visual_names = self.visual_names + ['class_mask_matrix']
            if self.use_PRL == True:
                patch_loss_type = config["PRL"].get("loss","L1")
                patch_loss = give_loss_by_name(patch_loss_type)
                self.criterionPRL = RPLoss(patch_loss=patch_loss, patch_size=config["PRL"].get("patch_size",7), patch_num=config["PRL"].get("patch_num",10), norm=config["PRL"].get("norm",True))

            # use_MCL/use_PRL 不能同时为True
            assert not ([self.use_MCL, self.use_PRL].count(True) > 1)
            
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
            self.optimizers.append(self.optimizer_G)            
            

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.class_mask_matrix = input['class_mask'].to(self.device)
        self.image_paths = {'A_path':input['A_path'],'B_path':input["B_path"]}

    def forward(self):
        
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def cal_loss_G(self):
        """Calculate GAN and L1 loss for the generator"""
        if self.use_MCL == True:
            if self.class_mask == "prepared":
                pass
            elif self.class_mask == "realtime_man":
                self.loss_G_L1, [self.loss_G_L1_0, self.loss_G_L1_1, self.loss_G_L1_2], self.class_mask_matrix = self.criterionMCL(self.fake_B, self.real_B)
            
        elif self.use_PRL == True:
            self.loss_G_L1 = self.criterionPRL(self.fake_B, self.real_B)
        else:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
            
        self.loss_G = self.loss_G_L1
      
    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()  
        self.cal_loss_G()
        self.loss_G.backward()
        
        if self.use_grad_clip == True:
            mode = self.config["grad_clip"].get("grad_clip_method","mean")
            layer = self.config["grad_clip"].get("grad_clip_layer","each")
            clip_grad(self.netG, mode, layer)

        self.optimizer_G.step()             # update G's weights

    def calculate_loss(self):
        with torch.no_grad():
            self.forward()
            self.cal_loss_G()
    
    
    
