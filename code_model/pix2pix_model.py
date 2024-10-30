import torch
from code_model.base_model import BaseModel
from code_network import define_network
from code_network.utils.loss import GANLoss

class Pix2PixModel(BaseModel):
    
    def __init__(self, config):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, config)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        

        self.netG = define_network(config, net_type="g")

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = define_network(config, net_type="d")

        if self.isTrain:
            # define loss functions
            gan_mode = config["network"]["gan_mode"]
            lr = config["network"]["lr"]
            beta1 = config["network"]["beta1"]
            self.criterionGAN = GANLoss(gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        # print("minmax of input A to model: ",torch.min(self.real_A), torch.max(self.real_A))
        # print("minmax of input B to model: ",torch.min(self.real_B), torch.max(self.real_B))
        self.image_paths = {'A_path':input['A_path'],'B_path':input["B_path"]}

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def cal_loss_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

    # def backward_D(self):

    def cal_loss_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.config["network"]["lambda_L1"]
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

    # def backward_G(self):
        
    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.cal_loss_D()               # calculate gradients for D
        self.loss_D.backward()
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.cal_loss_G()                  # calculate graidents for G
        self.loss_G.backward()
        self.optimizer_G.step()             # update G's weights

    def calculate_loss(self):
        with torch.no_grad():
            self.forward()
            self.cal_loss_D()
            self.cal_loss_G()
           