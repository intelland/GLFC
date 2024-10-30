import torch
import torch.nn as nn

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class MCLLoss(nn.Module):
    """Multiple Contrast Loss
    """

    def __init__(self, class_mask_range = None, class_weight = None, class_norm = False):
        super(MCLLoss, self).__init__()
        self.class_mask_range = class_mask_range
        # 将class_mask_range从min,max归一化到-1到1
        min_value, max_value = class_mask_range[0] # 默认第一个class的range就是min,max
        for i, class_range in enumerate(class_mask_range):
            lower_bound,upper_bound = class_range
            # 从min_value~max_value归一化到-1到1
            class_mask_range[i] = [2 * (lower_bound - min_value) / (max_value - min_value) - 1, 2 * (upper_bound - min_value) / (max_value - min_value) - 1]
        self.class_weight = class_weight
        # 将class_weight归一化到0到1
        class_weight_sum = sum(class_weight)
        self.class_weight = [weight / class_weight_sum for weight in class_weight]

        self.class_norm = class_norm
        self.loss = MaskedL1Loss()
        self.L1_sum = nn.L1Loss(reduction='sum')
    
    def forward(self, pre_img, real_img, class_mask = None):
        if class_mask != None:
            # 使用临时传递的class_mask计算loss
            single_class_masks = []
            unique_values = sorted(torch.unique(class_mask))
            for val in unique_values:
                single_class_masks.append((class_mask == val).int())
        else:
            # class mask 用于可视化
            class_mask = torch.zeros_like(real_img, dtype=torch.int)
            for i, class_range in enumerate(self.class_mask_range):
                    lower_bound,upper_bound = class_range
                    class_mask[(real_img >= lower_bound) & (real_img <= upper_bound)] = i
            single_class_masks = [(real_img >= lower_bound) & (real_img <= upper_bound).int() for (lower_bound,upper_bound) in self.class_mask_range]
        single_class_losses = []
        for i, single_class_mask in enumerate(single_class_masks):
            if self.class_norm == True:
                lower_bound,upper_bound = self.class_mask_range[i]
                pre_img_temp = 2 * (pre_img - lower_bound) / (upper_bound - lower_bound) - 1
                real_img_temp = 2 * (real_img - lower_bound) / (upper_bound - lower_bound) - 1
            loss_temp = self.loss(pre_img, real_img, single_class_mask)
            # loss_temp = self.loss(pre_img_temp, real_img_temp, single_class_mask)
            single_class_losses.append(loss_temp*self.class_weight[i])
        loss = sum(single_class_losses)
        # 将class_mask归一化到-1到1
        class_mask = 2 * (class_mask.float() - 0) / (len(self.class_mask_range) - 1) - 1
        return loss, single_class_losses, class_mask
    
class RPLoss(nn.Module):
    """Random Patch Loss
    """

    def __init__(self, patch_loss = torch.nn.L1Loss(), patch_size = 64, patch_num = 100, norm = False):
        super(RPLoss, self).__init__()
        self.patch_size = patch_size
        self.patch_num = patch_num
        self.criterionLoss = patch_loss
        self.norm = norm

    def forward(self, pre_img, real_img):
        
        for i in range(self.patch_num):
            pre_patch = self.get_patch(pre_img)
            real_patch = self.get_patch(real_img)
            if self.norm == True:
                lower_bound,upper_bound = real_img.min(), real_img.max()
                pre_patch = 2 * (pre_patch - lower_bound) / (upper_bound - lower_bound) - 1
                real_patch = 2 * (real_patch - lower_bound) / (upper_bound - lower_bound) - 1
            if i == 0:
                loss = self.criterionLoss(pre_patch,real_patch)
            else:
                loss += self.criterionLoss(pre_patch,real_patch)
        return loss / self.patch_num

    def get_patch(self, img):
        img_size = img.size()
        x = torch.randint(0, img_size[2] - self.patch_size, (1,))[0]
        y = torch.randint(0, img_size[3] - self.patch_size, (1,))[0]
        return img[:,:,x:x+self.patch_size,y:y+self.patch_size]
    
    

def give_loss_by_name(loss_name:str) -> nn.Module:
    if loss_name == "L1":
        return nn.L1Loss()
    elif loss_name == "L1_sum":
        return nn.L1Loss(reduction='sum')

            
class MaskedL1Loss(nn.Module):
    """Masked L1 Loss
    """

    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.L1 = nn.L1Loss()
        self.L1_sum = nn.L1Loss(reduction='sum')

    def forward(self, pre_img, real_img, mask):
        # # 对一个batch中的样本分别计算并取平均
        # pre_img_mask = torch.mul(pre_img, mask)
        # real_img_mask = torch.mul(real_img, mask)
        # print(pre_img_mask.min(),pre_img_mask.max())
        # print(real_img_mask.min(),real_img_mask.max())
        # loss = 0
        # for i in range(pre_img.size(0)):
        #     loss += self.L1_sum(pre_img_mask[i], real_img_mask[i]) / mask[i].sum()

        # return loss / pre_img.size(0)

        counts = mask.sum()
        if counts == 0:
            loss = 0
        else:
            loss = self.L1_sum(torch.mul(pre_img, mask), torch.mul(real_img, mask)) / counts
        return loss
    

            
    