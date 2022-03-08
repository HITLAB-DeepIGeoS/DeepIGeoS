from PyQt5.QtWidgets import*
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5 import *
from PyQt5 import uic
from PyQt5.QtGui import QImage, qRgb
import numpy as np
import nibabel as nib
import cv2, os
import GeodisTK

import glob
import ipywidgets as ipyw
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torchio as tio

from utils.geodis_toolkits import geodismap
from models.networks import P_RNet3D
from data_loaders.transforms import get_transform

def clk_seg(usrId, count, path, int_pos, int_neg, axis, img, pn, clk):
    pos = (int_pos==0)
    neg = (int_neg==0)
    
    clk = (clk.x(), clk.y()-7)
    
    if pn == 1:

        if not f'pos_{count}.npy' in os.listdir(path):
            np.save(f'../res/{usrId}/seg/{axis}/pos_{count}.npy', pos)

        img = cv2.circle(img, (clk[0], clk[1]), 8, (0, 0, 0), 3)

        cv2.imwrite(f'../res/{usrId}/seg/{axis}/{count}.png', img)

        pos = np.load(f'../res/{usrId}/seg/{axis}/pos_{count}.npy')
        pos[clk[1], clk[0]] = 1
        np.save(f'../res/{usrId}/seg/{axis}/pos_{count}.npy', pos)

        return img

    else:

        if not f'neg_{count}.npy' in os.listdir(path):
            np.save(f'../res/{usrId}/seg/{axis}/neg_{count}.npy', neg)

        img = cv2.rectangle(img, (clk[0]-8, clk[1]-8), (clk[0]+8, clk[1]+8), (0, 0, 0), 3)

        cv2.imwrite(f'../res/{usrId}/seg/{axis}/{count}.png', img)

        neg = np.load(f'../res/{usrId}/seg/{axis}/neg_{count}.npy')
        neg[clk[1], clk[0]] = 1
        np.save(f'../res/{usrId}/seg/{axis}/neg_{count}.npy', neg)

        return img


def nextImage( usrId, imgs, segs, ax, count, pn, clk=(0,0)):

    if ax==0 : axis= 'X'
    elif ax==1 : axis= 'Y'
    elif ax==2 : axis= 'Z'

    path = f'../res/{usrId}/seg/{axis}/'

    if count >= imgs.shape[0]:
        count = imgs.shape[0]


    if ax == 0:
        seg = segs[count,:,:]
        iH, iW = imgs[count,:,:].shape
    elif ax == 1: 
        seg = segs[:,count,:]
        iH, iW = imgs[:,count,:].shape
    elif ax == 2: 
        seg = segs[:,:,count]
        iH, iW = imgs[:,:,count].shape


    if not f'{count}.png' in os.listdir(path):
        int_pos = np.uint8(255*np.ones([iH*2, iW*2]))
        int_neg = np.uint8(255*np.ones([iH*2, iW*2]))
        if ax == 0:
            img = imgs[count,:,:]          
        elif ax == 1: 
            img = imgs[:,count,:]          
        elif ax == 2: 
            img = imgs[:,:,count]           

        img = cv2.divide(img, img.max())
        img = cv2.resize(img, (iW*2, iH*2))
        img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        pos = (int_pos==0)
        neg = (int_neg==0)

    else:
        int_pos = np.uint8(255*np.ones([iH*2, iW*2]))
        int_neg = np.uint8(255*np.ones([iH*2, iW*2]))

        img = cv2.imread(f'../res/{usrId}/seg/{axis}/{count}.png')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


    seg = cv2.divide(seg, seg.max())
    seg[np.where(seg!=0)]=1
    seg = cv2.normalize(src=seg, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    seg = cv2.resize(seg, (iW*2, iH*2))

    if clk == (0,0):
        return img, seg

    img = clk_seg(usrId, count, path, int_pos, int_neg, axis, img, pn, clk)

    return img, seg

        
def geodesic_distance_2d(I, S, lamb, iter):
    return GeodisTK.geodesic2d_raster_scan(I, S, lamb, iter)


def seg_reduction(int_seg):
    h, w = int_seg.shape
    idx = np.where(int_seg==1)
    
    for i in range(idx[0].shape[0]):
        int_seg[idx[0][i], idx[1][i] +1] =1

        int_seg[idx[0][i]+1, idx[1][i] +0] =1
        int_seg[idx[0][i]+1, idx[1][i] +1] =1
        
    int_seg = cv2.resize(int_seg, (int(w/2), int(h/2)), interpolation = cv2.INTER_NEAREST)
    
    return int_seg

def save_func(imgs, path, usrId):
    file_path = []
    for (root, directories, files) in os.walk(path):
        for file in files:
            if '.npy' in file:
                file_path.append(os.path.join(root, file))
                
    int_pos_result = np.uint8(255*np.ones(imgs.shape))
    int_neg_result = np.uint8(255*np.ones(imgs.shape))

    for i in file_path:
        axis = i.split('/')[-2]
        int_side = i.split('/')[-1].split('.')[0].split('_')[0]
        count = int(i.split('/')[-1].split('.')[0].split('_')[1])
        
        int_seg = np.load(i)
        int_seg = int_seg.astype('uint8')

        if int_side == 'pos':
            if axis == 'X':
                int_pos_result[count,:,:] = seg_reduction(int_seg)
            elif axis == 'Y':
                int_pos_result[:,count,:] = seg_reduction(int_seg)
            elif axis == 'Z':
                int_pos_result[:,:,count] = seg_reduction(int_seg)
                
        elif int_side == 'neg':
            if axis == 'X':
                int_neg_result[count,:,:] = seg_reduction(int_seg)
            elif axis == 'Y':
                int_neg_result[:,count,:] = seg_reduction(int_seg)
            elif axis == 'Z':
                int_neg_result[:,:,count] = seg_reduction(int_seg)
                
    int_pos_result = (int_pos_result==1) 
    int_neg_result = (int_neg_result==1)

    np.save(f'../res/{usrId}/result/int_pos_result.npy', int_pos_result)
    np.save(f'../res/{usrId}/result/int_neg_result.npy', int_neg_result)

    return int_pos_result, int_neg_result



def pnet_inference(
    image_path,
    save_path,
    pnet, 
    transform, 
    device
):
    """
    P-Net inference function
    
    Args:
        image_path: file path of input image (ex. image_path.nii.gz)
        save_path:  file path to save result (ex. pnet_pred.nii.gz)
        pnet:       trained pnet model (torch.nn.Module)
        transform:  preprocessing transforms (torchio.Compose)
        device:     torch device (torch.device)
    """

    # read image and make subject to apply transform
    subject = tio.Subject(
        image = tio.ScalarImage(image_path),
    )
    subject = transform(subject)

    # make numpy array to torch tensor
    input_image = subject.image.data
    input_tensor = input_image.unsqueeze(dim=0).to(device)

    # inference
    with torch.no_grad():
        pred_logits = pnet(input_tensor)
    
    # logits to labels
    pred_labels = torch.argmax(pred_logits, dim=1)

    # labels to one hot labels (ex. [1, 2, 3] -> [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    pred_onehot = torch.nn.functional.one_hot(pred_labels, 2).permute(0, 4, 1, 2, 3)
    pred_onehot_target = pred_onehot[:, 1, ...]

    # save result
    pred_labelmap = tio.LabelMap(tensor=pred_onehot_target.cpu())
    pred_labelmap.save(save_path)    
    
    
def rnet_inference(
    image_path, 
    pnet_pred_path,
    fg_point_path, 
    bg_point_path, 
    save_path,
    rnet, 
    transform, 
    device
):
    """
    R-Net inference function
    
    Args:
        image_path:     file path of input image (ex. image_path.nii.gz)
        pnet_pred_path: file path of pnet prediction label (ex. pnet_pred.nii.gz)
        fg_point_path:  foreground user interaction points file path (ex. fg_points.npy)
        bg_point_path:  background user interaction points file path (ex. bg_points.npy)
        save_path:      file path to save result (ex. pnet_pred.nii.gz)
        rnet:           trained rnet model (torch.nn.Module)
        transform:      preprocessing transforms (torchio.Compose)
        device:         torch device (torch.device)
    """

    # read image and pnet prediction and make subject to apply transform
    subject = tio.Subject(
        image = tio.ScalarImage(image_path),
        pnet_pred = tio.LabelMap(pnet_pred_path)
    )
    subject = transform(subject)

    # cast numpy array to torch tensor
    input_image = subject.image.data
    input_tensor = input_image.unsqueeze(dim=0).to(device)

    pnet_pred_label = subject.pnet_pred.data
    pnet_pred_tensor = pnet_pred_label.unsqueeze(dim=0).to(device)

    # read random point numpy array
    sf, sb = np.load(fg_point_path), np.load(bg_point_path)

    # get geodismap from random points and apply transform
    sf, sb = sf.astype(np.float32), sb.astype(np.float32)
    fore_dist_map, back_dist_map = geodismap(sf, sb, image_path)
    fore_dist_map = torch.Tensor(transform(np.expand_dims(fore_dist_map.transpose(1, 2, 0), axis=0)))
    back_dist_map = torch.Tensor(transform(np.expand_dims(back_dist_map.transpose(1, 2, 0), axis=0)))

    # make rnet input tensor
    rnet_inputs = torch.cat([
        input_tensor,
        pnet_pred_tensor, 
        fore_dist_map.unsqueeze(dim=1).to(device), 
        back_dist_map.unsqueeze(dim=1).to(device)
    ], dim=1)

    # inference
    with torch.no_grad():
        pred_logits = rnet(rnet_inputs)
    
    # logits to labels
    pred_labels = torch.argmax(pred_logits, dim=1)

    # labels to one hot labels (ex. [1, 2, 3] -> [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    pred_onehot = torch.nn.functional.one_hot(pred_labels, 2).permute(0, 4, 1, 2, 3)
    pred_onehot_target = pred_onehot[:, 1, ...]

    # save result
    pred_labelmap = tio.LabelMap(tensor=pred_onehot_target.cpu())
    pred_labelmap.save(save_path)