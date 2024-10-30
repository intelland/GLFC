from torch.optim import lr_scheduler

def get_scheduler(optimizer, config):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    
    if config["model"]["lr_policy"] == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + config["model"]["start_epoch"] - config["model"]["l_decay_flat"]) / float(config["model"]["l_decay_down"] + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif config["model"]["lr_policy"] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config["model"]["lr_decay_iters"], gamma=0.1)
    elif config["model"]["lr_policy"] == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif config["model"]["lr_policy"] == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["model"]["cos_decay_cycle"], eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', config["model"]["lr_policy"])
    return scheduler