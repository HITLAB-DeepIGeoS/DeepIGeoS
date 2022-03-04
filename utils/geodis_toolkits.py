import cv2
import random
import numpy as np
from multiprocessing import Pool

import torch
import GeodisTK
import SimpleITK as sitk


def focusregion_index(pred_array):

    """find index for each axis which has the biggest summation value"""

    # pred_array (H,W,D)

    h, w, d = None, None, None

    thres = 0
    for i in range(pred_array.shape[0]):
        if np.sum(pred_array[i]) > thres:
            h = i
            thres = np.sum(pred_array[i])

    thres = 0
    for i in range(pred_array.shape[1]):
        if np.sum(pred_array[:, i]) > thres:
            w = i
            thres = np.sum(pred_array[:, i])

    thres = 0
    for i in range(pred_array.shape[2]):
        if np.sum(pred_array[:, :, i]) > thres:
            d = i
            thres = np.sum(pred_array[:, :, i])

    return h, w, d


def randompoint(seg):

    # random point selection via component analysis

    """Then the user interactions on each mis-segmented
    region are simulated by randomly sampling n pixels in that
    region. Suppose the size of one connected under-segmented
    or over-segmented region is Nm, we set n for that region to
    0 if Nm < 30 and dNm/100 e otherwise based on experience."""

    seg_shape = seg.shape
    seg_array = np.array(seg, dtype=np.uint8)
    focus_h, focus_w, focus_d = focusregion_index(seg_array)
    output = np.zeros(shape=seg_shape)

    if None not in [focus_h, focus_w, focus_d]:
        # h
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
            seg_array[focus_h, :, :]
        )
        for i in range(1, len(stats)):
            region_size = stats[i][4]
            if region_size >= 30:
                number_n = int(np.ceil(region_size / 100))
                index_list = np.random.choice(region_size, number_n, replace=False)
                output[
                    focus_h,
                    np.where(labels == i)[0][index_list],
                    np.where(labels == i)[1][index_list],
                ] = 1

        # w
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
            seg_array[:, focus_w, :]
        )
        for i in range(1, len(stats)):
            region_size = stats[i][4]
            if region_size >= 30:
                number_n = int(np.ceil(region_size / 100))
                index_list = np.random.choice(region_size, number_n, replace=False)
                output[
                    np.where(labels == i)[0][index_list],
                    focus_w,
                    np.where(labels == i)[1][index_list],
                ] = 1

        # d
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
            seg_array[:, :, focus_d]
        )
        for i in range(1, len(stats)):
            region_size = stats[i][4]
            if region_size >= 30:
                number_n = int(np.ceil(region_size / 100))
                index_list = np.random.choice(region_size, number_n, replace=False)
                output[
                    np.where(labels == i)[0][index_list],
                    np.where(labels == i)[1][index_list],
                    focus_d,
                ] = 1

    return output


def randominteraction(pred_array, label_array):
    # oversegmented regions
    overseg = np.where(pred_array - label_array == 1, 1, 0)
    sb = randompoint(overseg)  # background

    # undersegmented regions
    underseg = np.where(pred_array - label_array == -1, 1, 0)
    sf = randompoint(underseg)  # foreground
    return sb, sf


def geodismap(sf, sb, image_path):

    # shape needs to be aligned.
    # original image shape : h, w, d

    # sf: foreward interaction
    # sb: backward interaction

    """
    Get 3D geodesic disntance by raser scanning.
    I: input image array, can have multiple channels, with shape [D, H, W] or [D, H, W, C]
       Type should be np.float32.
    S: binary image where non-zero pixels are used as seeds, with shape [D, H, W]
       Type should be np.uint8.
    spacing: a tuple of float numbers for pixel spacing along D, H and W dimensions respectively.
    lamb: weighting betwween 0.0 and 1.0
          if lamb==0.0, return spatial euclidean distance without considering gradient
          if lamb==1.0, the distance is based on gradient only without using spatial distance
    iter: number of iteration for raster scanning.
    """

    image = sitk.ReadImage(image_path)
    I = sitk.GetArrayFromImage(image)
    I = np.asarray(I, np.float32)

    spacing_raw = image.GetSpacing()
    spacing = [spacing_raw[2], spacing_raw[1],spacing_raw[0]]

    sf = np.array(sf, dtype=np.uint8).transpose(2, 0, 1)
    sb = np.array(sb, dtype=np.uint8).transpose(2, 0, 1)

    with Pool(2) as p:
        fore_dist_map, back_dist_map = p.starmap(GeodisTK.geodesic3d_raster_scan, 
                                                 [(I, sf, spacing, 1, 2), (I, sb, spacing, 1, 2)])

    return fore_dist_map, back_dist_map


def get_geodismaps(image_paths, true_labels, pred_labels, transform):
    pred_labels_np = np.array(pred_labels)
    true_labels_np = np.array(true_labels)

    fore_dist_map_batch = np.empty((pred_labels_np.shape[0], 1, *pred_labels_np.shape[1:]), dtype=np.float32)
    back_dist_map_batch = np.empty((pred_labels_np.shape[0], 1, *pred_labels_np.shape[1:]), dtype=np.float32)

    for i, (image_path, pred_label_np, true_label_np) in enumerate(zip(image_paths, 
                                                                       pred_labels_np, 
                                                                       true_labels_np)):
        sb, sf = randominteraction(pred_label_np, true_label_np)
        fore_dist_map, back_dist_map = geodismap(sf, sb, image_path)

        fore_dist_map_batch[i] = transform(np.expand_dims(fore_dist_map.transpose(1, 2, 0), axis=0))
        back_dist_map_batch[i] = transform(np.expand_dims(back_dist_map.transpose(1, 2, 0), axis=0))

    return fore_dist_map_batch, back_dist_map_batch