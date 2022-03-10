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
- [ ] 
```
*** working here ***
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
```

## :pushpin: Prerequisites
Please check versions of python and tensor flow before you start. If you need them, you should either upgrade versions or install them.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)


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
### ☺︎ Result 

From top to bottom, the figures are the original image, label, P-Net prediction, R-Net refined prediction
<div>
<img width="1010" alt="Screen Shot 2022-03-05 at 1 13 44 PM" src="https://user-images.githubusercontent.com/40614421/157642805-2c0d3e73-af88-48e6-be61-e75d98570680.png">
</div>

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


### ☺︎ Image augmentation 
[Bias field correction](https://www.slicer.org/wiki/Modules:MRIBiasFieldCorrection-Documentation-3.6) 
어느 부위가 시간차로 점점 어두워지는데, 랜덤하게 발생. 이것을 비슷하게 만들어 주는 전처리 

</div>
<img width="562" alt="Screen Shot 2022-02-03 at 20 05 40" src="https://user-images.githubusercontent.com/40614421/152331150-9c6218bd-f1ca-4687-94cc-7a8dfed9cb8e.png">
</div>

### ☺︎ augmentation affine transformation
https://torchio.readthedocs.io/transforms/augmentation.html

</div>
<img width="698" alt="Screen Shot 2022-02-03 at 20 02 46" src="https://user-images.githubusercontent.com/40614421/152330714-56f07a87-4697-44f2-bd60-bf20e257cb1b.png">
</div>
elastic transformation 

### ☺︎ Data Visualization 
#### ☻ [Seg3D](https://github.com/SCIInstitute/Seg3D/releases)
Seg3D is a free volume segmentation and processing tool developed by the NIH Center for Integrative Biomedical Computing at the University of Utah Scientific Computing and Imaging (SCI) Institute. Seg3D combines a flexible manual segmentation interface with powerful higher-dimensional image processing and segmentation algorithms from the Insight Toolkit. Users can explore and label image volumes using volume rendering and orthogonal slice view windows.

- [Tutorial](http://sciinstitute.github.io/seg3d.pages/Manuals/Seg3DTutorial.html#software-requirements)
- [Dataset](https://www.sci.utah.edu/cibc-software/cibc-datasets.html)

#### sample : Segmenting a Brain Dataset
<div>
<img width="700" alt="Screen Shot 2022-02-03 at 14 09 38" src="https://user-images.githubusercontent.com/40614421/152287376-0d698a87-482f-4a3e-8fbc-fe491485eacb.png">
</div>



