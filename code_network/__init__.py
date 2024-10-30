import importlib
from torch import nn

from code_network.tools.initialization import init_gpus,init_weights
from code_network.modules.general import get_norm_layer

def find_network_using_name(network_file, network_name):
    """Import the module "code_network/[network_file].py".

    """
    # 和code_data中找dataset类一样的逻辑
    network_filename = "code_network." + network_file
    networklib = importlib.import_module(network_filename)
    network = None
    target_network_name = network_name.replace('_', '')
    for name, cls in networklib.__dict__.items():
        if name.lower() == target_network_name.lower() \
           and issubclass(cls,nn.Module):
            network = cls

    if network is None:
        print("In %s.py, there should be a subclass of nn.Module with class name that matches %s in lowercase." % (network_filename, target_network_name))
        exit(0)

    return network

def define_network(config, net_type = "g"):

    gpu_ids =  config["model"]["gpu_ids"]
    norm = config["network"]["norm"]

    parameters = {
        # general
        "input_nc": config["dataset"]["image_channel"],
        "output_nc": config["dataset"]["image_channel"],
        "norm_layer": get_norm_layer(norm),

        # unet/unet++
        "ngf": config["network"].get("ngf"),
        "num_downs" : config["network"].get("num_downs"),
        "use_dropout": config["network"].get("dropout"),
        "deep_supervision": config["model"].get("deep_supervision"),

        # swin/mamba unet
        "patch_sizes": config["network"].get("patch_sizes"),
        
        # MEUNet
        "down_step": config["network"].get("down_step"),
        "patch_sizes": config["network"].get("patch_sizes"),
        "mamba_blocks":  config["network"].get("mamba_blocks"),
        "mamba_residual": config["network"].get("residual"),
        "f_a": config["network"].get("f_a"),
        
        #TEUNet:
        "transformer_depths":  config["network"].get("transformer_depths"),
        "transformer_residual": config["network"].get("residual"),
         
        # REUNet:
        "res_blocks": config["network"].get("res_blocks"),
        "res_residual": config["network"].get("residual"),
         
        # discriminator
        "ndf": config["network"].get("ndf"),
        "n_layers_D": config["network"].get("n_layers_D")
    }

    if net_type == "g":
        network_name = config["network"]["netG"]
        network_file = config["network"]["filename"]
    else:
        netD = config["network"]["netD"]
        if netD == "basic":
            network_name = "NLayerDiscriminator"
        elif netD == "pixel":
            network_name = "PatchGAN"
        network_file = config["network"]["filename_d"]
        parameters["input_nc"] = int(2 * parameters["input_nc"]) 
    
    ClassNetG = find_network_using_name(network_file, network_name)
    net = ClassNetG(**parameters)
    net = init_gpus(net,gpu_ids)
    if config["isTrain"] == True:
        init_weights(net, config["network"]["init_type"], config["network"]["init_gain"])

    return net        

    
