import matplotlib.pyplot as plt
import re
import os

from code_util import util

def plot_losses_from_file(train_loss, val_loss, loss_names, output_image = 'loss.png', iters=False, split=False, epoch_interval=1):
    
    def construct_output_position(loss_pos,output_image):
        if os.path.isdir(os.path.dirname(output_image)):
            output_path = output_image
        else:
            default_path = os.path.join(os.path.dirname(loss_pos),output_image)
            output_path = default_path
        output_path = os.path.splitext(output_path)[0]
        return output_path
    
    def read_losses(file_path, is_train):
        data = {'epoch': [], 'iters': [], 'losses': {loss: [] for loss in loss_names}}
        pattern = re.compile(r'\(epoch: (\d+), iters: ([\dNone]+), time: [\d.]+, data: [\d.]+\)')
        
        if file_path:
            with open(file_path, 'r') as file:
                for line in file:
                    match = pattern.search(line)
                    if match:
                        epoch = int(match.group(1))
                        iter_value = match.group(2)
                        iters_value = int(iter_value) if iter_value != 'None' else None
                        
                        data['epoch'].append(epoch)
                        data['iters'].append(iters_value)
                        for loss in loss_names:
                            loss_pattern = re.compile(rf'{loss}: ([\d.]+)')
                            loss_match = loss_pattern.search(line)
                            if loss_match:
                                data['losses'][loss].append(float(loss_match.group(1)))
                            else:
                                data['losses'][loss].append(None)
        return data

    def filter_by_interval(data, interval):
        filtered_data = {'epoch': [], 'losses': {loss: [] for loss in loss_names}}
        for i in range(0, len(data['epoch']), interval):
            filtered_data['epoch'].append(data['epoch'][i])
            for loss in loss_names:
                filtered_data['losses'][loss].append(data['losses'][loss][i])
        return filtered_data
    
    def select_first_loss_per_epoch(data):
        selected_data = {'epoch': [], 'losses': {loss: [] for loss in loss_names}}
        seen_epochs = set()
        for i in range(len(data['epoch'])):
            epoch = data['epoch'][i]
            if epoch not in seen_epochs:
                seen_epochs.add(epoch)
                selected_data['epoch'].append(epoch)
                for loss in loss_names:
                    selected_data['losses'][loss].append(data['losses'][loss][i])
        return selected_data

    train_data = read_losses(train_loss, is_train=True)
    val_data = read_losses(val_loss, is_train=False)
    output_image = construct_output_position(train_loss,output_image)
    
    if iters is False:
        train_data = select_first_loss_per_epoch(train_data)
        if epoch_interval > 1:
            train_data = filter_by_interval(train_data, epoch_interval)
            val_data = filter_by_interval(val_data, epoch_interval)
    
    if split:
        if train_loss:
            plt.figure(figsize=(10, 6))
            for loss in loss_names:
                plt.plot(train_data['epoch'], train_data['losses'][loss], 'x-', label=f'Train {loss}')
            plt.title(f'Training Losses Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{output_image}_train.png')
            plt.close()
        
        if val_loss:
            plt.figure(figsize=(10, 6))
            for loss in loss_names:
                plt.plot(val_data['epoch'], val_data['losses'][loss], 'o-', label=f'Validation {loss}')
            plt.title(f'Validation Losses Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{output_image}_val.png')
            plt.close()
    else:
        plt.figure(figsize=(10, 6))
        for loss in loss_names:
            if train_loss:
                plt.plot(train_data['epoch'], train_data['losses'][loss], 'x-', label=f'Train {loss}')
            if val_loss:
                plt.plot(val_data['epoch'], val_data['losses'][loss], 'o-', label=f'Validation {loss}')
        plt.title(f'Losses Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_image}.png')
        plt.close()



if __name__ == "__main__":
    # 使用示例
    experiment_name = ""
    experiment_folder = util.find_latest_experiment(experiment_name)
    train_loss = os.path.join(experiment_folder,'loss_train.txt')
    val_loss = os.path.join(experiment_folder,'loss_val.txt')
    loss_names = ['G_L1']
    save_name = 'losses_plot.png'
    plot_losses_from_file(train_loss,val_loss, loss_names, save_name, epoch_interval=5)
