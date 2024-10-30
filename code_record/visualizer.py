import numpy as np
import os
import sys
import time
import torch

from code_util import util
from code_util.data import read_save
from . import html
from subprocess import Popen, PIPE



if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, config):
        """Initialize the Visualizer class
        Parameters:    
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        对train和test 有不同的处理
        """

        self.config = config  # cache the option
        self.isTrain = self.config["isTrain"]
        self.name = config["name"]
        self.work_dir = config["work_dir"]
       
        
        # visdom
        if config["record"]["use_visdom"] == True:
            self.use_visdom = True
            self.init_visdom()
        # html
        if self.config["record"]["use_html"] == True:  
            self.use_html = True
            self.init_html()
        # loss log file 
        if self.config["isTrain"]: 
            self.log_train_name = os.path.join(self.work_dir, 'loss_train.txt')
            with open(self.log_train_name, "w") as log_file:
                print("loss_train: ",self.log_train_name)
                now = time.strftime("%c")
                log_file.write('================ Training at (%s) ================\n' % now)
           
            self.log_val_name = os.path.join(self.work_dir, 'loss_val.txt')
            print("loss_val: ",self.log_val_name)
            with open(self.log_val_name, "w") as log_file:
                log_file.write('================ val at (%s) ================\n' % now)
                

    def init_visdom(self):
        import visdom
        self.win_loss_train = 1
        self.win_loss_val = 2
        self.win_loss_train_val = "win_loss_train_val"
        self.win_img_train = 4
        self.win_img_val = 5
        self.win_text = 6

        self.server = "http://" + self.config["record"]["visdom_server"]
        self.port = self.config["record"]["visdom_port"]
        self.env = self.name
        self.ncols = self.config["record"]["visdom_ncols"]
        self.vis = visdom.Visdom(server=self.server, port=self.port, env= self.env)
        if not self.vis.check_connection():
            self.create_visdom_connections()

    def init_html(self):
        self.web_dir = os.path.join(self.work_dir,'web')
        print('create web directory %s...' % self.web_dir)
        os.makedirs(self.web_dir,exist_ok=True)
        self.win_size = self.config["record"]["display_size_html"]
        if self.config["isTrain"] == True:
            img_train_dir = os.path.join(self.web_dir, 'train')
            os.makedirs(img_train_dir,exist_ok=True)
            title = 'Experiment name = %s | train' % self.name
            self.webpage_train = html.HTML(self.web_dir, title, filename = "train", refresh=30)
            self.img_dir = [img_train_dir]
    
            img_val_dir = os.path.join(self.web_dir, 'val')
            os.makedirs(img_val_dir,exist_ok=True)
            title = 'Experiment name = %s | val' % self.name
            self.webpage_val = html.HTML(self.web_dir, title, filename = "val", refresh=30)
            self.img_dir.append(img_val_dir)
        else: 
            img_test_dir = os.path.join(self.web_dir, 'test')
            os.makedirs(img_test_dir,exist_ok=True)
            title = 'Experiment = %s, Epoch = %s' % (self.name, self.config["results"]["test_epoch"])
            self.webpage_test = html.HTML(self.web_dir, title, filename = "test", refresh=30)
            self.img_dir = [img_test_dir]

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_on_visdom(self, visuals, epoch=0, epoch_iter=0, phase='train'):
        """Display current results on Visdom.

        Parameters:
            visuals (dict) -- dictionary of images to display or save
            epoch (int)    -- current epoch
            epoch_iter (int) -- iteration within the epoch
            phase (str)    -- 'train' or 'val'
        """
        win = getattr(self, f'win_img_{phase}')
        title = f'{self.name} - Epoch: {epoch}, Iter: {epoch_iter} {phase}'
        ncols = min(self.ncols, len(visuals)) if self.ncols > 0 else 1
        h, w = next(iter(visuals.values())).shape[:2]
        table_css = f"""
        <style>
            table {{border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}}
            table td {{width: {w}px; height: {h}px; padding: 4px; outline: 4px solid black}}
        </style>"""

        images = []
        label_html = ''
        label_html_row = ''

        for idx, (label, image) in enumerate(visuals.items(), start=1):
            image_numpy = util.tensor2im(image)
            label_html_row += f'<td>{label}</td>'
            images.append(image_numpy.transpose([2, 0, 1]))

            if idx % ncols == 0 or idx == len(visuals):
                label_html += f'<tr>{label_html_row}</tr>'
                label_html_row = ''

        while len(images) % ncols != 0:
            images.append(np.ones_like(image_numpy.transpose([2, 0, 1])) * 255)
            label_html_row += '<td></td>'
        if label_html_row:
            label_html += f'<tr>{label_html_row}</tr>'

        try:
            self.vis.images(images, nrow=ncols, win=win, padding=2, opts=dict(title=title + ' images'))
            # self.vis.text(table_css + f'<table>{label_html}</table>', win=getattr(self, f'win_text'),
            #             opts=dict(title=title + ' labels'))
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def display_on_html(self, visuals, epoch = 0, epoch_iter = 0):
        """
        save current results to an HTML file.
        """
        if self.config["isTrain"] == True:
            epoch_str = str(epoch)
            if epoch_iter != 0: # train
                epoch_iter_str = str(epoch_iter)
                img_dir = self.img_dir[0]
                webpage = self.webpage_train
            else: # val
                epoch_iter_str = "None"
                img_dir = self.img_dir[1]
                webpage = self.webpage_val
        else: # test
            epoch_str = self.config["results"]["test_epoch"]
            epoch_iter_str = str(epoch_iter)
            img_dir = self.img_dir[0]
            webpage = self.webpage_test
        
        # save images to the disk
        for label, image in visuals.items():
            # print("label: ", label)
            # print("shape:", image.shape)
            # print("minmax of tensor: ", torch.min(image),torch.max(image))
            image_numpy = util.tensor2im(image)
            # print("minmax of numpy: ", np.min(image_numpy),np.max(image_numpy))
            img_path = os.path.join(img_dir, '%s_%s_%s.png' % (epoch_str,epoch_iter_str, label))
            read_save.save_image_4_show(image_numpy, img_path)

        # update website
        webpage.add_header('epoch %s iter %s' % (epoch_str,epoch_iter_str))
        ims, txts, links = [], [], []

        for label, image_numpy in visuals.items():
            image_numpy = util.tensor2im(image)
            img_path = '%s_%s_%s.png' % (epoch_str,epoch_iter_str, label)
            ims.append(img_path)
            txts.append(label)
            links.append(img_path)
        webpage.add_images(ims, txts, links, width=self.win_size)
        webpage.save()


    def plot_current_losses(self, epoch, counter_ratio, losses, phase):
        """Display the current losses on Visdom.

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (dict)         -- training losses stored in the format of (name, float) pairs
            phase (str)           -- 'train' or 'val'
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {
                'train': {'X': [], 'Y': [], 'legend': list(losses.keys())},
                'val': {'X': [], 'Y': [], 'legend': list(losses.keys())}
            }

        plot_phase = self.plot_data[phase]
        # current_epoch = epoch + (counter_ratio if phase == 'train' else 0)
        current_epoch = epoch
        plot_phase['X'].append(current_epoch)
        plot_phase['Y'].append([losses[k] for k in plot_phase['legend']])

        try:
            for i, loss_name in enumerate(plot_phase['legend']):
                self.vis.line(
                    X=np.array([current_epoch]),
                    Y=np.array([plot_phase['Y'][-1][i]]),
                    win=self.win_loss_train_val,
                    name=f'{phase}_{loss_name}',
                    update='append' if self.vis.win_exists(self.win_loss_train_val) else None,
                    opts=dict(
                        title="Train and val loss over time",
                        xlabel='epoch',
                        ylabel='loss',
                        showlegend=True
                    ) if not self.vis.win_exists(self.win_loss_train_val) else None
                )
        except VisdomExceptionBase:
            self.create_visdom_connections()


    # losses: same format as |losses| of plot_current_losses
    def record_current_losses(self, losses, t_comp=0, t_data=0, epoch = 0, epoch_iter = 0, phase = "train"):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        if phase == "train_iter": # train
            log_name = self.log_train_name
            message = '(epoch: %s, iters: %s, time: %.3f, data: %.3f) ' % (epoch, epoch_iter, t_comp, t_data)
        elif phase == "train_epoch":
            log_name = self.log_train_name
            message = '(epoch: %s, time: %.3f) ' % (epoch,t_comp)
        else: # val
            log_name = self.log_val_name
            message = '(epoch: %s) ' % (epoch)
        
        for k, v in losses.items():
            message += '%s: %.4f ' % (k, v)
        # print(message)  # print the message
        with open(log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

