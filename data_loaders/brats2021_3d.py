import os
import random
import torchio as tio
from torch.utils.data import Dataset, DataLoader

from data_loaders.transforms import get_transform


class BratsDataset(Dataset):

    def __init__(self, paths, transform=None):
        self.paths = paths
        self._set_file_paths()
        self.transform = transform
        
    def __len__(self):
        return len(self.label_paths)
            
    def __getitem__(self, index):
        subject = tio.Subject(
            image = tio.ScalarImage(self.image_paths[index]),
            label = tio.LabelMap(self.label_paths[index]),
        )
        if self.transform:
            subject = self.transform(subject)

        # Shape : (C, W, H, D), (W, H, D)
        return subject.image.data, subject.label.data[0, ...]

    def _set_file_paths(self):
        self.image_paths = []
        self.label_paths = []
        
        for path in self.paths:
            base = os.path.basename(path)
            self.image_paths.append(f"{path}/{base}_flair.nii.gz")
            self.label_paths.append(f"{path}/{base}_seg.nii.gz")


def get_dataloaders(config):
    train_dir = os.path.join(config.data.data_dir, "train")
    train_valid_files = sorted(os.listdir(train_dir))
    
    random.shuffle(train_valid_files)
    split_idx = int(len(train_valid_files) * 0.8)
    train_paths = [os.path.join(train_dir, f) for f in train_valid_files[:split_idx]]
    valid_paths = [os.path.join(train_dir, f) for f in train_valid_files[split_idx:]]

    dataloaders = {}
    for split in ["train", "valid"]:
        if split == "train":
            train_dataset = BratsDataset(paths=train_paths, transform=get_transform(split))
            dataloaders[split] = DataLoader(
                train_dataset,
                batch_size=config.data.batch_size, 
                shuffle=True, 
                num_workers=config.data.num_workers,
            )
        elif split == "valid":
            valid_dataset = BratsDataset(paths=valid_paths, transform=get_transform(split))
            dataloaders[split] = DataLoader(
                valid_dataset,
                batch_size=config.data.batch_size, 
                shuffle=False, 
                num_workers=config.data.num_workers,
            )
        
    return dataloaders