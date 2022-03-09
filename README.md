# :brain: DeepIGeoS

: DeepIGeoS Paper Implementation
> :page_facing_up: [DeepIGeoS: A Deep Interactive Geodesic Framework for Medical Image Segmentation (2018)](https://ieeexplore.ieee.org/abstract/document/8370732)

Steps are below : 
- [x] check prerequisites
- [x] download Dataset 
- [x] Train (P-Net, R-Net)
- [x] 
.
.
.
- [x] background of this work 

## :pushpin: Prerequisites
Please check versions of python and tensor flow before you start. If you need them, you should either upgrade versions or install them.
### ☺︎ Environments
```
- Ubuntu 16.04
- Python 3.7.11
```
### ☺︎ Libraries 

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
## :pushpin: Datasets

```shell
$ bash load_datasets.sh
```

## :pushpin: Train

### ☺︎ Train P-Net

```shell
$ python train_pnet.py -c configs/config_pnet.json
```

### ☺︎ Train R-Net
```shell
$ python train_rnet.py -c configs/config_rnet.json
```

### ☺︎ Tensorboard
```shell
$ tensorboard --logdir experiments/logs/
```

### ☺︎ Run
```shell
$ python main_deepigeos.py
```



## :pushpin: Backgound 

### ☺︎ DeepIGeoS
#### purpose : 
- a deep learning-based interactive segmentation method to imporve the results obtained by an automatic CNN and to reduce user interactions during refinement for high accuracy 
- integrate user interactions as hard constraints into a back-propagatable Conditional Random Field.
- image segmentation에서 중요한 것은 시간보다 정확도 임. 

<div>
<img width="900" alt="Screen Shot 2022-01-24 at 01 08 28" src="https://user-images.githubusercontent.com/40614421/150687508-05f27a1c-12da-4cfe-8f1c-3b787c651a2e.png">
</div>

- Two stage pipeline 
  : 1st CNN (obtains automatically initial segmentation) + 2nd CNN (refines initial segmentation w/ small # of user interactions that we encode as geodesic distance maps)
  - P-net : use CNN to obtain an **initial automatic segmentation** + CRF
  - R-net: refine the segmentation by taking as input **the original image, the initial segmentation and the user interactions**. 즉, provides clicks/scribbles to correct Foreground와 Background. + CRF + 4 inputs (original image, initial segmentation, gedesic distance map based on foregroun/background

<div>
<img width="400" alt="Screen Shot 2022-01-24 at 01 08 00" src="https://user-images.githubusercontent.com/40614421/150687476-90accb09-9249-403a-8298-6f7f5834ab66.png">
</div>

> both P-net and R-net are connected with a CRF, which is modeled as an RNN (CRF-Net) so that it can be trained jointly with P-Net/R-Net by back-propagation. 
  - CRFs (원본과 segmentationt사이의 E를 최소화,[Conditional Random Fields as Recurrent Neural Networks](https://arxiv.org/abs/1502.03240)): enhance segmentation accuracy by introducing spatial consistency. Such methods encourage segmentation consistency between adjacent pixel pairs with high similarity. In order to better model long-range connections within the image,a fully connected CRF was used to establith pairwise potentials on all pairs of pixels in the image. 즉, 쉽게 말하면 resolution을 높이는 것.
 </div>
<img width="400" alt="Screen Shot 2022-01-16 at 19 57 42" src="https://user-images.githubusercontent.com/40614421/150639045-971777fa-574a-469d-b1e1-e8f208300bf8.png">
 </div>
[FIGURE](https://vladlen.info/publications/efficient-inference-in-fully-connected-crfs-with-gaussian-edge-potentials/)



  - [Geodesic Distance Maps](https://towardsdatascience.com/preserving-geodesic-distance-for-non-linear-datasets-isomap-d24a1a1908b2) : Interactions with the same label are converted into a distance map. The euclidean dis- tance was used due to its simplicity. However, the euclid- ean distance treats each direction equally and does not take the image context into account. In contrast, the geodesic dis- tance helps to better differentiate neighboring pixels with different appearances, and improves label consistency in homogeneous regions. 

</div>
<img width="400" alt="Screen Shot 2022-01-16 at 19 57 42" src="https://user-images.githubusercontent.com/40614421/152308002-46a416a6-0464-452c-8d71-9107b03dce69.png">
</div>

<img width="534" alt="Screen Shot 2022-02-03 at 17 34 25" src="https://user-images.githubusercontent.com/40614421/152308215-a90cf79c-2a89-4266-b26c-59d167599e10.png"> 
</div>

</div>
<img width="740" alt="Screen Shot 2022-02-03 at 17 34 20" src="https://user-images.githubusercontent.com/40614421/152308287-bbdafcc7-95cf-4cb9-8ed7-d8f85311e45b.png">
</div>
 
 

### ☺︎ [Dataset](https://drive.google.com/file/d/1Bv5Nc8lZOWdAeaX9uA5_uwUgGQ8sP96p/view?usp=drivesdk)
- [Ref. paper](https://arxiv.org/pdf/2107.02314.pdf)
- The BraTS dataset describes a retrospective collection of brain tumor mpMRI scans acquired from multiple different institutions under standard clinical conditions, but with different equipment and imaging protocols, resulting in a vastly heterogeneous image quality reflecting diverse clinical practice across different institutions. Inclusion criteria comprised pathologically confirmed diagnosis and available MGMT promoter methylation status. These data have been updated, since BraTS 2020, increasing the total number of cases from 660 to 2,000.

<div>
<img width="498" alt="Screen Shot 2022-02-03 at 13 32 37" src="https://user-images.githubusercontent.com/40614421/152281580-f2e62d08-4e3d-444f-8324-4eb97406187f.png">
</div>


#### ☻ [Seg3D](https://github.com/SCIInstitute/Seg3D/releases)
eg3D is a free volume segmentation and processing tool developed by the NIH Center for Integrative Biomedical Computing at the University of Utah Scientific Computing and Imaging (SCI) Institute. Seg3D combines a flexible manual segmentation interface with powerful higher-dimensional image processing and segmentation algorithms from the Insight Toolkit. Users can explore and label image volumes using volume rendering and orthogonal slice view windows.

- [Tutorial](http://sciinstitute.github.io/seg3d.pages/Manuals/Seg3DTutorial.html#software-requirements)
- [Dataset](https://www.sci.utah.edu/cibc-software/cibc-datasets.html)

##### sample : Segmenting a Brain Dataset

<div>
<img width="700" alt="Screen Shot 2022-02-03 at 14 09 38" src="https://user-images.githubusercontent.com/40614421/152287376-0d698a87-482f-4a3e-8fbc-fe491485eacb.png">
</div>


### ☺︎ Image augmentation 
#### [Bias field correction](https://www.slicer.org/wiki/Modules:MRIBiasFieldCorrection-Documentation-3.6) 
어느 부위가 시간차로 점점 어두워지는데, 랜덤하게 발생. 이것을 비슷하게 만들어 주는 전처리 

</div>
<img width="562" alt="Screen Shot 2022-02-03 at 20 05 40" src="https://user-images.githubusercontent.com/40614421/152331150-9c6218bd-f1ca-4687-94cc-7a8dfed9cb8e.png">
</div>

#### augmentation affine transformation
- https://torchio.readthedocs.io/transforms/augmentation.html

</div>
<img width="698" alt="Screen Shot 2022-02-03 at 20 02 46" src="https://user-images.githubusercontent.com/40614421/152330714-56f07a87-4697-44f2-bd60-bf20e257cb1b.png">
</div>
elastic transformation 


#### visualization 
ROI에 따라서 
#B C H W D
image[0,0,:,:,D//2]
image[0,0,:,W//2,:]


### ☺︎ Interaction, [Pyqt](https://build-system.fman.io/pyqt5-tutorial)
zeros-like?
3D에서 하나의 slice만 고려함. 

https://dev-overload.tistory.com/44
pyside2
https://wikidocs.net/35742
https://www.pythonguis.com/tutorials/first-steps-qt-creator/
http://www.itksnap.org/pmwiki/pmwiki.php


-------------------------------------------------------------------------

### ☺︎ [P-net code](https://drive.google.com/drive/folders/1saFAiJ-U_t5_hI8Y9fwOXCcyAJgmG_7P) 

### ☺︎ R-net code 


#### ☻ [Languages and tool](https://github.com/Ileriayo/markdown-badges)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

#### ☻ [Wandb](https://wandb.ai/soyounson/test)

Wandb: https://pebpung.github.io/wandb/2021/10/06/WandB-1.html

https://halfundecided.medium.com/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-cnn-convolutional-neural-networks-%EC%89%BD%EA%B2%8C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0-836869f88375

#### ☻ [Pytorch 한국사용자](https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html)

### ☺︎ [torchIO](https://torchio.readthedocs.io/)
TorchIO is an open-source Python library for efficient loading, preprocessing, augmentation and patch-based sampling of 3D medical images in deep learning, following the design of PyTorch.

It includes multiple intensity and spatial transforms for data augmentation and preprocessing. These transforms include typical computer vision operations such as random affine transformations and also domain-specific ones such as simulation of intensity artifacts due to MRI magnetic field inhomogeneity (bias) or k-space motion artifacts.

### ☺︎ [Pytorch](https://data-panic.tistory.com/13) + [image](https://kubig-2021-2.tistory.com/66)
- 1. 이미지 데이터 
- 2. 필요한 라이브러리 갖고오기
  - torchvision에서 이미지를 전처리 할 수 있는 transforms을 사용 
  - torch.utils.data에서 Dataset과 DataLoader를 사용하여 데이터를 가져옴
- 3. 데이터 전처리 
  - transforms.Compose를 통해 Resize, ToTensor, Normalize를 시켜줌 (이미지 높이, 너비등이 다를수 있어서 정규화 진행)
  - ToTensor를 통해 이미지를 텐서형태로 바꿔줌
- 4. DataLoader를 통해 데이터 올리기
  - 이번에는 ImageFolder를 통해 불러온 trainset을 DataLoader를 사용하여 Batch형식으로 네트워크에 올릴 수 있도록 준비
- 5. 이미지 데이터 시각화



https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html

### Ref 
- 이영석님 코드 : 
- HITLAB notion: https://www.notion.so/modulabs/HIT-LAB-783449555af449f89c0b2f2dc4a5a72b?p=05e5484efffe4d70a49ec00cf46b7319
#### etc
upload figure : https://github.com/soyounson/Modulabs/issues/new


-------------------------------------------------------------------------
## **이영석님이 공유해주신 내용** 

### ☻ 2D P-Net 구현 및 학습
- 2D P-Net을 구현 및 학습한 코드입니다.
    - Google Colab Pro 환경에서 학습
- 데이터셋은 PASCAL VOC 2012 `trainaug`를 사용하였습니다.
    - class index 1번인 `aeroplane`을 binary segmentation 하도록 수정
- 학습 결과는 `wandb`를 사용해서 기록하도록 구현하였습니다.
    - Loss, IoU, Dice Score, Prediction 결과 이미지들(validation set에서 10장) 기록

### ☻ PASCAL VOC 2012 Dataset
https://github.com/yassouali/pytorch-segmentation#datasets

[DeepLabV3](https://arxiv.org/abs/1802.02611)과 같은 segmentation 관련 논문을 보면, [Semantic contours from inverse detectors](http://home.bharathh.info/pubs/pdfs/BharathICCV2011.pdf)의 annotation을 추가로 사용하여, `trainaug`라고 부르는 10582장의 이미지로 구성된 데이터로 학습을 수행합니다.

`trainaug` 데이터셋을 바로 내려받는 링크는 없어서, 위의 링크를 참고해 데이터셋을 구성하는 방법을 정리하였습니다.

1. Pascal VOC 2012 datset
    - [여기](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)에서 원본 PASCAL VOC 2012 데이터셋을 내려받아 압축을 풀면 `VOCtrainval_11-May-2012/VOCdevkit/VOC2012`의 경로에 이미지와 annotation들이 구성되어 있는 것을 확인할 수 있습니다.
2. Augment dataset
    - [여기](https://www.dropbox.com/sh/jicjri7hptkcu6i/AACHszvCyYQfINpRI1m5cNyta?dl=0&lst=)에서 내려받은 `Segmentation` 디렉토리의 4개 파일(`train_aug`, `trainval_aug`, `val_aug`, `test_aug`)을 1에서 내려받은 경로인 `VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation`에 옮겨줍니다. (이름이 같은 파일은 덮어써도 됩니다.)
    - 마지막으로, [여기](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0)에서 `trainaug`를 위한 annotation이 있는 `SegmentationClassAug`를 내려받아 `VOCtrainval_11-May-2012/VOCdevkit/VOC2012`에 옮겨줍니다.
3. 확인
    - 위의 과정을 통해 다음의 구조로 데이터셋이 구성되어있는 것을 확인할 수 있습니다.

```
VOCtrainval_11-May-2012
├── VOCdevkit
│   ├── VOC2012
│   │   ├── Annotations
│   │   ├── ImageSets
│   │   │   ├── Segmentation
│   │   │       ├── test_aug.txt
│   │   │       ├── train_aug.txt
│   │   │       ├── trainval_aug.txt
│   │   │       ├── val_aug.txt
│   │   ├── JPEGImages
│   │   ├── SegmentationClass
│   │   ├── SegmentationClassAug
│   │   ├── SegmentationObject
```
