import numpy as np
from torch.utils.tensorboard import SummaryWriter

class MetricsLogger:

    def __init__(self, config):
        self.config = config
        self.init_logger()
        self.init_tensorboard()

    def init_logger(self):
        self.logger = {}
        for split in self.config.logger.splits:
            for metric in self.config.logger.metrics:
                self.logger[f"{split}_{metric}"] = []

    def init_tensorboard(self):
        self.summary_writer = SummaryWriter(self.config.exp.tensorboard_dir)
    
    def reset(self):
        for k in self.logger.keys():
            self.logger[k] = []

    def update(self, split, result_dict):
        for metric, value in result_dict.items():
            self.logger[f"{split}_{metric}"].append(value)

    def get_value(self, split, metric):
        return np.mean(self.logger[f"{split}_{metric}"])

    def summarize(self, split):
        log_msgs = []
        for metric in self.config.logger.metrics:
            log_msgs.append(f"{metric}: {np.mean(self.logger[f'{split}_{metric}']):.3f}")
        print(f"[{split}] " + ", ".join(log_msgs))
    
    def write_tensorboard(self, step):
        for split in self.config.logger.splits:
            summary_dict = {}
            for metric in self.config.logger.metrics:
                summary_dict.update({f"{metric}": np.mean(self.logger[f"{split}_{metric}"])})
            self.summary_writer.add_scalars(split, summary_dict, step)