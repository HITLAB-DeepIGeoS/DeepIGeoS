import torch


def device_config(config):
    if torch.cuda.is_available():
        print("GPU is available. Run with GPU")
        try:
            if len(config.exp.gpu_ids) == 1:
                # Single GPU
                config.exp.multi_gpu = False
                config.exp.device = torch.device(f"cuda:{config.exp.gpu_ids[0]}")
            else:
                # Multi GPU
                config.exp.multi_gpu = True
                pass
        except RuntimeError as e:
            print(e)
    else:
        config.exp.device = torch.device("cpu")
        print("GPU is not available. Run with CPU")

