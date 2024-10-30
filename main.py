import os
def main():
    
    # SynthRad2023 
    if SynthRad2023 == True:

        from code_util.dataset.specific_dataset import convert_SynthRAD2023_Task_2

        input_dir = "../../../datasets/SynthRAD2023/original/Task2/brain/division3"
        output_dir = "./file_dataset/SynthRAD2023/brain3"

        convert_SynthRAD2023_Task_2(input_dir,output_dir,isTrain=False)

    # split 3D to 2D
    if split_3D_to_2D == True:
        
        from code_util.dataset.prepare import process_all_3D_volumes

        input_dir = "../../../datasets/SynthRAD2023/original/Task2/brain/division2"
        output_dir = "./file_dataset/SynthRAD2023/brain"
        pattern = "^2B[A-C]\d{3}$"

        process_all_3D_volumes(input_dir,output_dir,pattern)

    if mask == True:
        from code_util.dataset.specific_dataset import prepare_mask_SynthRAD2023_Task_2
        mask_root = "../../datasets/SynthRAD2023/original/Task2/brain/division3"
        input_mask_dir = os.path.join(mask_root,mode)
        output_mask_root = "./file_dataset/SynthRAD2023/brain3/mask"
        output_mask_dir = os.path.join(output_mask_root,"class_2_big","3D",mode)
        prepare_mask_SynthRAD2023_Task_2(input_mask_dir,output_mask_dir)
    
    if mask_2D == True:
        from code_util.dataset.prepare import process_all_3D_volumes
        pattern = "^2B[A-C]\d{3}$"
        mask_root = "./file_dataset/SynthRAD2023/brain3/mask"
        input_mask_dir = os.path.join(mask_root,"class_2_small","3D",mode)
        output_mask_dir = os.path.join(mask_root,"class_2_small","2D",mode)
        process_all_3D_volumes(input_mask_dir,output_mask_dir,pattern)

    # class_mask 
    if class_mask == True:
        from code_util.dataset.specific_dataset import generate_class_mask_SynthRAD2023_Task_2
        data_root= "/home/xdh/data/intelland/datasets/SynthRAD2023/original/Task2/brain/division3"
        output_root = "./file_dataset/SynthRAD2023/brain3/mask/class_2_small/3D"
        input_dir = os.path.join(data_root,mode)
        output_dir = os.path.join(output_root,mode)
        if class_mask_type == "auto":
            generate_class_mask_SynthRAD2023_Task_2(input_dir,output_dir)
        elif class_mask_type == "man":
            class_range = [[-1024,3000],[-250,3000]]
            generate_class_mask_SynthRAD2023_Task_2(input_dir,output_dir,method="range",class_range=class_range)

    # reconstruct 
    if reconstruct == True:

        from code_util.dataset.reconstruct import find_and_process_files

        input_dir = os.path.join("./file_result",experiment_name,os.path.join(test_epoch,"images"))
        output_dir = os.path.join("./file_result",experiment_name,os.path.join(test_epoch,"3D"))   
        pattern = r"^(.+)_(\d+)_(.+)\.nii\.gz$"
        
        find_and_process_files(input_dir, output_dir, pattern)

    # calculate metrics
    if calculate_metrics == True:

        from code_util.metrics.calculate import calculate_folder
        data_folder = os.path.join("./file_result",experiment_name,os.path.join(test_epoch,"3D"))
        mask_folfer = os.path.join("./file_dataset/SynthRAD2023/brain3/mask/","class_2_"+mask_type,"3D/test")
        class_mask_folder = os.path.join("./file_dataset/SynthRAD2023/brain3/mask/",class_mask_class,"3D/test")
        result_folder = os.path.join("./file_result",experiment_name,test_epoch)
        # metric_names = [SSIM_type,"PSNR","MSE","MAE","RMSE"]
        metric_names = [SSIM_type,"PSNR"]

        calculate_folder(data_folder, result_folder, mask_folder = mask_folfer, class_mask_folder=class_mask_folder, metric_names=metric_names, device_id=device_id)
        # calculate_folder(data_folder, result_folder, SSIM_type="SSIM")

    # evaluate 

    # loss
    if plot_loss == True:
        from code_util.analysis.loss import plot_losses_from_file
        from code_util import util
        experiment_folder = util.find_latest_experiment(experiment_name)
        train_loss = os.path.join(experiment_folder,'loss_train.txt')
        val_loss = os.path.join(experiment_folder,'loss_val.txt')
        loss_names = ['G_L1']
        save_name = 'losses_plot.png'
        plot_losses_from_file(train_loss,val_loss, loss_names, save_name, split=False, epoch_interval=5)


if __name__ == "__main__":
    mode = "train" # train test validation 
    SynthRad2023 = 0
    split_3D_to_2D = False # 从原始数据集开始 只需要运行SynthRad2023即可 其包含了将3D切分为2D的过程
    mask = 0
    class_mask = 0
    class_mask_type = "man"
    mask_2D = 0
    test_epoch = "test_latest"
    reconstruct = 1
    calculate_metrics = 1
    mask_type = "big"
    class_mask_class = "class_3"
    plot_loss = False
    SSIM_type = "SSIM"
    device_id = 0
    experiment_name = "CBCT2CT_Unet_64_4"
    main()

