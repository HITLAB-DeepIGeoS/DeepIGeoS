# :brain: DeepIGeoS

: DeepIGeoS Paper Implementation
> :page_facing_up: [DeepIGeoS: A Deep Interactive Geodesic Framework for Medical Image Segmentation (2018)](https://ieeexplore.ieee.org/abstract/document/8370732)

## :pushpin: Prerequisites
Please check versions of python and tensor flow before you start. If you need them, you should either upgrade versions or install them.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

### ☺︎ Environments
`Ubuntu 16.04`

`Python 3.7.11`


### ☺︎ Libraries 
```
shell
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
Download dataset using `load_datasets.sh`
```
$ bash load_datasets.sh
```

## :pushpin: Train

### ☺︎ Train P-Net
First, train P-net
```
$ python train_pnet.py -c configs/config_pnet.json
```

### ☺︎ Train R-Net
Train R-net
```
$ python train_rnet.py -c configs/config_rnet.json
```

### ☺︎ Tensorboard
```
$ tensorboard --logdir experiments/logs/
```

### ☺︎ Run
run `main_deepigeos.py` as below:
```
$ python main_deepigeos.py
```
### ☺︎ Result 
Now, check results.

From top to bottom, the figures are the original image, label, P-Net prediction, R-Net refined prediction
<div>
<img width="1010" alt="Screen Shot 2022-03-05 at 1 13 44 PM" src="https://user-images.githubusercontent.com/40614421/157642805-2c0d3e73-af88-48e6-be61-e75d98570680.png">
</div>

## :pushpin: Backgound 

### ☺︎ DeepIGeoS
To perform DeepIGeoS, we have studied and understood the original paper. 

- A deep learning-based interactive segmentation method to imporve the results obtained by an automatic CNN and to reduce user interactions during refinement for high accuracy 
- Integrate user interactions as hard constraints into a back-propagatable Conditional Random Field.
- In the image segmentation, the accuracy is a curial factor to evaluate models
- Architecture of DeepIGeoS is 

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
  - CRFs (minimize the E between original and segmentationt,[Conditional Random Fields as Recurrent Neural Networks](https://arxiv.org/abs/1502.03240)): enhance segmentation accuracy by introducing spatial consistency. Such methods encourage segmentation consistency between adjacent pixel pairs with high similarity. In order to better model long-range connections within the image,a fully connected CRF was used to establith pairwise potentials on all pairs of pixels in the image. The purpose of this process is a resolution increasement. 
  - **But here CRFs is not implemented!**
  - 
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
**working here**

ref : https://torchio.readthedocs.io/transforms/augmentation.html



### ☺︎ Data Visualization 
**working here**


