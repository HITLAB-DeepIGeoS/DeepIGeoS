import torch


def labels_to_one_hot(labels, n_classes, smooth=1e-6):
    '''
    label (b, ...) -> one hot label (b, c, ...)
    '''
    shape = labels.shape
    one_hot = torch.zeros((shape[0], n_classes) + shape[1:], device=labels.device)
    result = one_hot.scatter_(1, labels.type(torch.long).unsqueeze(1), 1.0) + smooth

    return result


def acc(true_labels, pred_labels, smooth=1e-6):
    """
    y_true, y_pred : (B, W, H, D)
    """
    true_labels_f = torch.flatten(true_labels)
    pred_labels_f = torch.flatten(pred_labels)

    corrects = torch.sum(true_labels_f == pred_labels_f)
    score = (corrects + smooth) / (len(pred_labels_f) + smooth)
    
    return score.to("cpu").numpy()


def iou(true_labels, pred_labels, smooth=1e-6):
    """
    y_true, y_pred : (B, W, H, D)
    """
    true_labels_f = torch.flatten(true_labels)
    pred_labels_f = torch.flatten(pred_labels)

    intersection = torch.sum(true_labels_f * pred_labels_f)
    union = torch.sum(true_labels_f) + torch.sum(true_labels_f) - intersection
    score = (intersection + smooth) / (union + smooth)

    return score.to("cpu").numpy()


def dsc(true_labels, pred_labels, smooth=1e-6):
    """
    y_true, y_pred : (B, W, H, D)
    """
    true_labels_f = torch.flatten(true_labels)
    pred_labels_f = torch.flatten(pred_labels)

    intersection = torch.sum(true_labels_f * pred_labels_f)
    score = (2. * intersection + smooth) / (torch.sum(true_labels_f) + torch.sum(pred_labels_f) + smooth)

    return score.to("cpu").numpy()


def batch_assd(outputs, labels, smooth=1e-6):
    """
    y_true, y_pred : (B, W, H, D)
    """
    pass