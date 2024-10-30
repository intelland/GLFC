import os
from tqdm import tqdm
from code_dataset import create_dataset
from code_model import create_model
from code_config.parser import parse
from code_record.visualizer import Visualizer
from code_util.data.read_save import save_test_image
import pprint
from code_util.cam.grad_cam import GradCAM
import cv2

if __name__ == '__main__':

    config = parse("test")
    results_dir = os.path.join(config["results"]["results_dir"], config["name"], '{}_{}'.format("test", config["results"]["test_epoch"]))
    config["work_dir"] = results_dir
    config["record"]["validation"] = False
    pprint.pprint(config)
    
    # dataset
    dataset, test_len = create_dataset(config)  # create a dataset given dataset_mode and other configurations

    print('The number of training images = %d' % test_len)

    # model
    model = create_model(config)      # create a model given opt.model and other options
    model.setup(config)               # regular setup: load and print networks; create schedulers

    # create a website
    visualizer = Visualizer(config)

    # test with eval mode. This only affects layers like batchnorm and dropout.
    model.eval()
    epoch_iter = 0

    for i, data in enumerate(tqdm(dataset, desc="Testing")):
        epoch_iter += 1
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_paths = model.get_image_paths()     # get image paths
        
        # Display results to HTML if needed
        if config["record"]["use_html"]:
            if epoch_iter % config["record"]["display_html_per_iter"] == 0:  # save images to an HTML file
                # print('processing (%04d)-th image... %s' % (i, img_paths))
                visualizer.display_on_html(visuals, epoch_iter=epoch_iter)
        
        if config["record"].get("CAM",{}).get("use_CAM",False):
           
            if epoch_iter % config["record"]["CAM"]["display_CAM_per_iter"] == 0:
                target_layer_name = config["record"]["CAM"]["CAM_layer"]
                # 打印所有name
                # for name, module in model.netG.named_modules():
                #     print(name)
                # 找到对应的layer
                target_layer = None
                for name, module in model.netG.named_modules():
                    if name == target_layer_name:
                        target_layer = module
                        break
                grad_cam = GradCAM(model.netG, target_layers=[target_layer], use_cuda=True)
                grayscale_cam = grad_cam(input_tensor=model.real_A, target = model.real_B)
                # print(grayscale_cam.shape)
                # 将其保存到本地
                save_path = os.path.join(config["work_dir"],"CAM")
                os.makedirs(save_path,exist_ok=True)
                save_name = os.path.basename(img_paths["A_path"][0])
                save_name = os.path.splitext(save_name)[0]
                save_name = save_name + "_CAM.jpg"
                save_path = os.path.join(save_path,save_name)
                # 保存为灰度图
                grayscale_cam = grayscale_cam[0]*255
                cv2.imwrite(save_path,grayscale_cam)
                
        # Save all test results locally
        save_list = ["real_A", "real_B", "fake_B"]
        save_test_image(visuals, img_paths, config, save_list)


