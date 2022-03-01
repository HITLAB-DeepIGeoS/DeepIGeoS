import os
import json
from dotmap import DotMap


def get_config_from_json(json_file):
    # parse the configurations from config.json
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace
    config = DotMap(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    
    config.exp.tensorboard_dir = os.path.join(config.exp.exp_dir, 
                                              "logs/",
                                              config.exp.name)
    config.exp.last_ckpt_dir = os.path.join(config.exp.exp_dir, 
                                            "last_ckpts/",
                                            config.exp.name)
    config.exp.best_ckpt_dir = os.path.join(config.exp.exp_dir, 
                                            "best_ckpts/",
                                            config.exp.name)
    if config.exp.save_val_pred:
        config.exp.val_pred_dir = os.path.join(config.exp.exp_dir, 
                                               "val_preds/",
                                               config.exp.name)
    
    return config