"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
import json
import pprint
import os

from tqdm import tqdm
from code_dataset import create_dataset
from code_model import create_model
from code_config.parser import parse
from code_record.visualizer import Visualizer
from code_util import util
from code_util.cam.grad_cam import GradCAM


if __name__ == '__main__':
    
    # opt >>>> config
    config = parse("train")
    current_time = time.localtime()
    formatted_time = time.strftime("%Y%m%d_%H%M%S", current_time)
   
    config["work_dir"] = os.path.join(config["record"]["checkpoints_dir"],config["name"],formatted_time)
    os.makedirs(config["work_dir"],exist_ok=True)

    # save configuration 
    config_path = os.path.join(config["work_dir"],"train_config.json")
    with open(config_path, 'w') as json_file:
        json.dump(config, json_file, indent=4)
    pprint.pprint(config)

    # dataset
    train_loader, val_loader, (train_len,val_len) = create_dataset(config)  
    
    print('The number of training images = %d' % train_len)
    print('val is %s enabled' % "" if val_len > 0 else "not")

    # random seed
    seed = config["random_seed"]
    util.set_random_seed(seed)

    # model
    model = create_model(config)      # init model
    model.setup(config)               # load network for test; set scheduler for train

    # visualizer 
    visualizer = Visualizer(config)   
 
    # total_iters = 0                # the total number of training iterations
    # Initialize tqdm for the outer loop
    for epoch in tqdm(range(1, config["model"]["l_decay_flat"] + config["model"]["l_decay_down"] + 1), desc="Epochs"):    
        model.train()

        epoch_start_time = time.time()  # timer for entire epoch
        epoch_losses = {}  # Initialize dictionary to store losses for the epoch
        epoch_loss_count = 0

        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        # Initialize tqdm for the inner loop
        for i, data in enumerate(tqdm(train_loader, desc="Training Iterations", leave=False)):
        # for i, data in enumerate(train_loader):
            # total_iters += config["dataset"]["dataloader"]["batch_size"]
            epoch_iter += config["dataset"]["dataloader"]["batch_size"]

            if epoch_iter % config["record"]["record_loss_per_iter"] == 0:
                iter_start_time = time.time()  # timer for computation per iteration
                t_data = iter_start_time - iter_data_time

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if config["record"]["use_visdom"]:
                if epoch_iter % config["record"]["display_visdom_per_iter"] == 0:  # visdom
                    model.compute_visuals()
                    visualizer.display_on_visdom(model.get_current_visuals(), epoch, epoch_iter, phase="train")

            if config["record"]["use_html"]:
                if epoch_iter % config["record"]["display_html_per_iter"] == 0:  # html
                    model.compute_visuals()
                    visualizer.display_on_html(model.get_current_visuals(), epoch, epoch_iter)

            losses = model.get_current_losses()
            for k, v in losses.items():
                    if k in epoch_losses:
                        epoch_losses[k] += v
                    else:
                        epoch_losses[k] = v
            epoch_loss_count += 1

            if epoch_iter % config["record"]["record_loss_per_iter"] == 0:    # loss to txt and visdom
                t_comp = (time.time() - iter_start_time) / config["dataset"]["dataloader"]["batch_size"]
                visualizer.record_current_losses(losses, t_comp, t_data, epoch, epoch_iter, phase="train_iter")
            
            iter_data_time = time.time()

        # Calculate average loss for the epoch
        t_comp = time.time() - epoch_start_time
        avg_losses = {k: v / epoch_loss_count for k, v in epoch_losses.items()}
        visualizer.record_current_losses(avg_losses, t_comp, t_data = None, epoch = 1, phase = "train_epoch")
        if config["record"]["use_visdom"]:
            visualizer.plot_current_losses(epoch, 0, avg_losses, phase="train")
        model.update_learning_rate()  # update learning rates

        if val_len > 0:
            if epoch % config["record"]["val_per_epoch"] == 0:
                model.clear_loss()
                total_losses = model.get_current_losses()
                model.eval()
                epoch_iter = 0
                # Initialize tqdm for the val loop
                for i, data in enumerate(tqdm(val_loader, desc="val Iterations", leave=False)):
                    epoch_iter += 1
                    model.set_input(data) 
                    model.calculate_loss()          
                    losses = model.get_current_losses()
                    total_losses = util.merge_dicts_add_values(total_losses, losses)
                total_losses = util.dict_divided_by_number(total_losses, epoch_iter)
                model.compute_visuals()
                visuals = model.get_current_visuals()
                if config["record"]["use_visdom"]:  # visdom
                    visualizer.display_on_visdom(visuals, epoch=epoch, phase="val") 
                    visualizer.plot_current_losses(epoch, 0, total_losses, phase="val")
                if config["record"]["use_html"]:  # html
                    visualizer.display_on_html(visuals, epoch=epoch)  # html
                visualizer.record_current_losses(total_losses, epoch=epoch, phase = 'val')  # txt

                if config["record"].get("CAM",{}).get("use_cam",False):
                    grad_cam = GradCAM(model.netG, target_layers=["layer4"], use_cuda=False)
                    grayscale_cam = grad_cam(input_tensor=model.real_A, target = model.real_B)
                    
                    
        if epoch % config["record"]["save_model_per_epoch"] == 0:  # cache our model every <save_epoch_freq> epochs
            # print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        # print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, config["model"]["l_decay_flat"] + config["model"]["l_decay_down"], time.time() - epoch_start_time))
