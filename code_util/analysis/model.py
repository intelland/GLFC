import torch

def load_models_and_print_size(model_paths, model_class=None):
    if not isinstance(model_paths, list):
        model_paths = [model_paths]

    for path in model_paths:
        try:
            state_dict = torch.load(path)

            # 如果模型是OrderedDict，需要先初始化模型实例
            if isinstance(state_dict, dict) or isinstance(state_dict, torch.nn.modules.container.OrderedDict):
                if model_class is None:
                    print(f"Error: Please provide the 'model_class' parameter to load the state_dict for model at {path}.")
                    continue

                # 创建模型实例
                model = model_class()
                model.load_state_dict(state_dict)
            else:
                model = state_dict  # 如果模型不是OrderedDict则直接使用

            # 打印模型参数的大小
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model at {path} has {total_params} parameters.")
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
