import torch
import random
import numpy as np
import torch.distributed as dist

import torch.multiprocessing as mp
from multiprocessing import Manager
from multiprocessing.managers import BaseManager

from utils.utils import get_args
from utils.dirs import create_dirs
from utils.device import device_config
from utils.logger import MetricsLogger
from utils.config import process_config

from models.networks import build_model
from data_loaders.brats2021_3d import get_dataloaders
from trainers.brats_3d_rnet_trainer import Brats3dRnetTrainer


def main():
    # Get config path & process config file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # Create experiments dirs
    create_dirs((
        config.exp.tensorboard_dir, 
        config.exp.last_ckpt_dir, 
        config.exp.best_ckpt_dir,
        config.exp.val_pred_dir
    ))

    # Device config (GPU / CPU)
    device_config(config)

    # Create logger 
    if config.exp.multi_gpu:
        # shared between processes
        BaseManager.register('MetricsLogger', MetricsLogger)
        manager = BaseManager()
        manager.start()
        logger = manager.MetricsLogger(config)

    # Run main workers
    if config.exp.multi_gpu:
        mp.spawn(
            main_worker, 
            nprocs=config.exp.world_size, 
            args=(config, logger,)
        )
    else:
        # Load datasets
        dataloaders = get_dataloaders(config)
        
        # Build model
        model = build_model(config)

        # Create logger
        logger = MetricsLogger(config)

        # Create trainer
        trainer = Brats3dRnetTrainer(model, dataloaders, config, logger)

        # Train
        trainer.train()


def main_worker(rank, config, logger):
    # Initialize worker group
    setup(rank, config.exp.world_size)

    # Set cuda visible device
    torch.cuda.set_device(rank)
    
    # Set configures for each gpu
    config.data.num_workers = int(config.data.num_workers / config.exp.ngpus_per_node)
    config.exp.device = torch.device(f"cuda:{rank}")
    config.exp.rank = rank

    # Set random seed
    random.seed(1111)
    np.random.seed(1111)
    torch.manual_seed(1111)
    
    # Load datasets
    dataloaders = get_dataloaders(config)
    
    # Build model
    model = build_model(config)

    # Create trainer
    trainer = Brats3dRnetTrainer(model, dataloaders, config, logger)

    # Train
    trainer.train()

    # Cleanup process
    cleanup()


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:12355',
        world_size=world_size,
        rank=rank
    )


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    main()