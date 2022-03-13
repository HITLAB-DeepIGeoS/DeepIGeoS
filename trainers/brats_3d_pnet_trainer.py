import os
import numpy as np
from tqdm import tqdm

import torch
import torchio as tio
import torch.distributed as dist

from utils.dirs import create_dirs
from models.metrics import iou, dsc, assd


class Brats3dPnetTrainer:

    def __init__(self, model, dataloaders, config, logger):
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        self.logger = logger
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        self.init_checkpoint()
        self.best_loss = np.inf
        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

    def set_loss_fn(self):
        if self.config.trainer.loss == "cross_entropy":
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Loss {config.trainer.loss} is not supported")

    def set_optimizer(self):
        if self.config.trainer.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                             lr=self.config.trainer.learning_rate,
                                             momentum=self.config.trainer.momentum,
                                             weight_decay=self.config.trainer.weight_decay)
        elif self.config.trainer.optimizer == "adam":
            self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                              lr=self.config.trainer.learning_rate)
        else:
            raise ValueError(f"Optimizer {self.config.trainer.optmizer} is not supported")

    def set_lr_scheduler(self):
        if self.config.trainer.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.config.trainer.step_size,
                                                                self.config.trainer.gamma)
        else:
            raise ValueError(f"LR Scheduler {self.config.trainer.lr_scheduler} is not supported")

    def init_checkpoint(self):
        last_ckpt = None
        if os.path.exists(self.config.exp.last_ckpt_dir):
            last_ckpts = sorted(os.listdir(self.config.exp.last_ckpt_dir))
            if last_ckpts:
                if self.config.exp.multi_gpu:
                    dist.barrier()
                    last_ckpt = torch.load(os.path.join(self.config.exp.last_ckpt_dir, last_ckpts[-1]), 
                                           map_location=self.config.exp.device)
                else:
                    last_ckpt = torch.load(os.path.join(self.config.exp.last_ckpt_dir, last_ckpts[-1]))

        if last_ckpt:
            if self.config.exp.multi_gpu:
                self.model.module.load_state_dict(last_ckpt['model_state_dict'])
            else:
                self.model.load_state_dict(last_ckpt['model_state_dict'])
            self.start_epoch = last_ckpt['epoch']
            self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(last_ckpt["lr_scheduler_state_dict"])
            print(f"Restored latest checkpoint from {os.path.join(self.config.exp.last_ckpt_dir, last_ckpts[-1])}")
        else:
            self.start_epoch = 0
            print("No trained checkpoints. Start training from scratch.")
                
    def save_checkpoint(self, epoch):
        # Save last checkpoint
        if self.config.exp.multi_gpu:
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        torch.save({"epoch": epoch + 1,
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "lr_scheduler_state_dict": self.lr_scheduler.state_dict()
                    }, f"{self.config.exp.last_ckpt_dir}/lask_ckpt_epoch_{epoch:04}.pt")
        last_ckpts = sorted(os.listdir(self.config.exp.last_ckpt_dir))
        if len(last_ckpts) > self.config.exp.max_to_keep_ckpt:
            os.remove(f"{self.config.exp.last_ckpt_dir}/{last_ckpts[0]}")
        
        # Save best checkpoint
        if self.best_loss > self.logger.get_value("valid", "loss"):
            self.best_loss = self.logger.get_value("valid", "loss")
            torch.save(state_dict, f"{self.config.exp.best_ckpt_dir}/best_ckpt_epoch_{epoch:04}.pt")
            print(f"Saved best model {f'{self.config.exp.best_ckpt_dir}/best_ckpt_epoch_{epoch:04}.pt'}")
            best_ckpts = sorted(os.listdir(self.config.exp.best_ckpt_dir))
            if len(best_ckpts) > 1:
                os.remove(f"{self.config.exp.best_ckpt_dir}/{best_ckpts[0]}")

    def train_epoch(self):
        cumu_loss = 0.
        cumu_iou = 0.
        cumu_dsc = 0.
        cumu_assd = 0.

        iter_cnt = 0
        inf_cnt = 0
        self.model.train()
        for _, inputs, true_labels in tqdm(self.dataloaders["train"], 
                                           desc="train phase",
                                           total=len(self.dataloaders["train"])):
            iter_cnt += 1

            inputs_norm = self.norm_transform(inputs.squeeze(dim=1)) # (N, C, W, H, D)
            inputs_norm = inputs_norm.unsqueeze(dim=1)
            inputs_norm = inputs_norm.to(self.config.exp.device)
            true_labels = true_labels.to(self.config.exp.device).type(torch.long) # (N, W, H, D)

            pred_logits = self.model(inputs_norm)
            loss = self.loss_fn(pred_logits, true_labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred_labels = torch.argmax(pred_logits, dim=1)
            pred_onehot = torch.nn.functional.one_hot(pred_labels, self.config.model.n_classes).permute(0, 4, 1, 2, 3)
            true_onehot = torch.nn.functional.one_hot(true_labels, self.config.model.n_classes).permute(0, 4, 1, 2, 3)

            cumu_loss += loss.to("cpu").item()
            cumu_iou += iou(pred_onehot, true_onehot, include_background=False).mean()
            cumu_dsc += dsc(pred_onehot, true_onehot, include_background=False).mean()
            assd_score = assd(pred_onehot, true_onehot, include_background=False).mean()
            if np.isinf(assd_score):
                inf_cnt += 1
            else:
                cumu_assd += assd_score

        result_dict = {
            "loss": cumu_loss / iter_cnt,
            "iou":  cumu_iou  / iter_cnt,
            "dsc":  cumu_dsc  / iter_cnt,
        }
        if (iter_cnt - inf_cnt) == 0:
            result_dict.update({"assd": np.inf})
        else:
            result_dict.update({"assd": cumu_assd / (iter_cnt - inf_cnt)})
        return result_dict

    def valid_epoch(self, val_save_dir_epoch):
        cumu_loss = 0.
        cumu_iou = 0.
        cumu_dsc = 0.
        cumu_assd = 0.

        iter_cnt = 0
        inf_cnt = 0
        self.model.eval()
        with torch.no_grad():
            for image_paths, inputs, true_labels in tqdm(self.dataloaders["valid"], 
                                                         desc="valid phase",
                                                         total=len(self.dataloaders["valid"])):
                iter_cnt += 1
                
                inputs_norm = self.norm_transform(inputs.squeeze(dim=1)) # (N, C, W, H, D)
                inputs_norm = inputs_norm.unsqueeze(dim=1)
                inputs_norm = inputs_norm.to(self.config.exp.device)
                true_labels = true_labels.to(self.config.exp.device).type(torch.long) # (N, W, H, D)

                pred_logits = self.model(inputs_norm)
                loss = self.loss_fn(pred_logits, true_labels)

                pred_labels = torch.argmax(pred_logits, dim=1)
                pred_onehot = torch.nn.functional.one_hot(pred_labels, self.config.model.n_classes).permute(0, 4, 1, 2, 3)
                true_onehot = torch.nn.functional.one_hot(true_labels, self.config.model.n_classes).permute(0, 4, 1, 2, 3)

                cumu_loss += loss.to("cpu").item()
                cumu_iou += iou(pred_onehot, true_onehot, include_background=False).mean()
                cumu_dsc += dsc(pred_onehot, true_onehot, include_background=False).mean()
                assd_score = assd(pred_onehot, true_onehot, include_background=False).mean()
                if np.isinf(assd_score):
                    inf_cnt += 1
                else:
                    cumu_assd += assd_score

                if val_save_dir_epoch:
                    pred_onehot_target = pred_onehot[:, 1, ...].cpu()
                    for i, image_path in enumerate(image_paths):
                        pred_labelmap = tio.LabelMap(tensor=pred_onehot_target[i].unsqueeze(dim=0))
                        pred_labelmap.save(os.path.join(val_save_dir_epoch,
                                                        os.path.basename(image_path.replace("_flair", "_pred"))))

        result_dict = {
            "loss": cumu_loss / iter_cnt,
            "iou":  cumu_iou  / iter_cnt,
            "dsc":  cumu_dsc  / iter_cnt,
        }
        if (iter_cnt - inf_cnt) == 0:
            result_dict.update({"assd": np.inf})
        else:
            result_dict.update({"assd": cumu_assd / (iter_cnt - inf_cnt)})
        return result_dict

    def train(self):
        for epoch in range(self.start_epoch, self.config.trainer.num_epochs):
            print(f"Epoch {epoch:4.0f}/{self.config.trainer.num_epochs - 1}")
            # Sync all processes before start training
            if self.config.exp.multi_gpu:
                dist.barrier()

            # Shuffle each sampler
            if self.config.exp.multi_gpu:
                self.dataloaders["train"].sampler.set_epoch(epoch)

            # Train
            train_result_dict = self.train_epoch()
            self.logger.update("train", train_result_dict)

            # Valid
            if self.config.exp.save_val_pred:
                # save valid prediction
                val_save_dir_epoch = os.path.join(self.config.exp.val_pred_dir, f"epoch_{epoch:03}")
                if not self.config.exp.multi_gpu or (self.config.exp.multi_gpu and self.config.exp.rank == 0):
                    create_dirs([val_save_dir_epoch])
                if self.config.exp.multi_gpu:
                    dist.barrier()
                valid_result_dict = self.valid_epoch(val_save_dir_epoch)
            else:
                # without saving
                valid_result_dict = self.valid_epoch()
            self.logger.update("valid", valid_result_dict)

            # Learning rate scheduling
            self.lr_scheduler.step()

            # Wait other process before save and logging
            if self.config.exp.multi_gpu:
                dist.barrier()

            if not self.config.exp.multi_gpu or (self.config.exp.multi_gpu and self.config.exp.rank == 0):
                # Save checkpoint
                self.save_checkpoint(epoch)

                # Save logs to tensorboard
                self.logger.write_tensorboard(step=epoch)

                # Print epoch history and reset logger
                self.logger.summarize("train")
                self.logger.summarize("valid")
                self.logger.reset()