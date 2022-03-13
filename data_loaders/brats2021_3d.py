import os
import random
import torchio as tio
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

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
        subject.add_image(tio.LabelMap(tensor=(subject.image.data.clone() > 0), 
                                       affine=subject.image.affine),
                          image_name="crop_mask")
        if self.transform:
            subject = self.transform(subject)

        # Shape : list, (C, W, H, D), (W, H, D)
        return self.image_paths[index], subject.image.data.float(), subject.label.data[0, ...]

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
    train_paths = sorted([os.path.join(train_dir, f) for f in train_valid_files[:split_idx]])
    valid_paths = sorted([os.path.join(train_dir, f) for f in train_valid_files[split_idx:]])

    dataloaders = {}
    for split in ["train", "valid"]:
        if split == "train":
            train_dataset = BratsDataset(paths=train_paths, transform=get_transform(split))
            if config.exp.multi_gpu:
                train_sampler = DistributedSampler(train_dataset)
                shuffle = False
            else:
                train_sampler = None
                shuffle = True
            dataloaders[split] = DataLoader(
                dataset=train_dataset,
                sampler=train_sampler,
                batch_size=config.data.batch_size, 
                shuffle=shuffle, 
                num_workers=config.data.num_workers,
                pin_memory=True
            )
        elif split == "valid":
            valid_dataset = BratsDataset(paths=valid_paths, transform=get_transform(split))
            if config.exp.multi_gpu:
                valid_sampler = DistributedSampler(valid_dataset)
            else:
                valid_sampler = None
            dataloaders[split] = DataLoader(
                dataset=valid_dataset,
                sampler=valid_sampler,
                batch_size=config.data.batch_size, 
                shuffle=False, 
                num_workers=config.data.num_workers,
                pin_memory=True
            )
        
    return dataloaders