# DeepIGeoS
DeepIGeoS Paper Implementation [[paper](https://arxiv.org/abs/1707.00652)]

## Environments
- Ubuntu 16.04
- Python 3.7.11

## Requirements
```shell
dotmap
GeodisTK==0.1.7
opencv-python==4.5.5.62
scipy==1.7.3
SimpleITK==2.1.1
tensorboard==2.8.0
torch==1.10.2
torchio==0.18.73
torchvision==0.11.3
tqdm==4.62.3
pyqt

```

## Download Datasets
```shell
$ bash load_datasets.sh
```

## Train

### 1. Train P-Net
```shell
$ python train_pnet.py -c configs/config_pnet.json
```

### 2. Train R-Net
```shell
$ python train_rnet.py -c configs/config_rnet.json
```

## Tensorboard
```shell
$ tensorboard --logdir experiments/logs/
```

## Run
```shell
$ python3 main_deepigeos.py
