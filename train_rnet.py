import torch
import random
import argparse
import numpy as np

from utils.dirs import create_dirs
from utils.config import process_config
from utils.device import device_config
from utils.logger import MetricsLogger

from models.networks import build_model
from data_loaders.brats2021_3d import get_dataloaders
from trainers.brats_3d_rnet_trainer import Brats3dRnetTrainer


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def set_randomness(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    # Get config path & process config file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # Set random seed
    set_randomness(seed=1111)

    # Device config (GPU / CPU)
    device_config(config)

    # Create experiments dirs
    create_dirs((
        config.exp.tensorboard_dir, 
        config.exp.last_ckpt_dir, 
        config.exp.best_ckpt_dir,
        config.exp.val_pred_dir
    ))

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


if __name__ == '__main__':
    main()