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
                if len(config.exp.gpu_ids) != torch.cuda.device_count():
                    raise ValueError(f"All GPU ids {config.exp.gpu_ids} are not available!")
                config.exp.multi_gpu = True
                config.exp.nodes = 1
                config.exp.ngpus_per_node = len(config.exp.gpu_ids)
                config.exp.world_size = config.exp.nodes * config.exp.ngpus_per_node

        except RuntimeError as e:
            print(e)
    else:
        config.exp.device = torch.device("cpu")
        print("GPU is not available. Run with CPU")
