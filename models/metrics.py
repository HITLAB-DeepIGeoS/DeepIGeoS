import monai


def iou(pred_onehot, true_onehot, include_background=False):
    """
    Params
        true_onehot: (B, C, W, H, D)
        pred_onehot: (B, C, W, H, D)

    Returns
        score: (B, C)
    """
    conf_matrix = monai.metrics.get_confusion_matrix(
        pred_onehot, 
        true_onehot, 
        include_background
    )
    score = monai.metrics.compute_confusion_matrix_metric(
        "threat score", 
        conf_matrix
    )
    return score.to("cpu").numpy()


def dsc(pred_onehot, true_onehot, include_background=False):
    """
    Params
        pred_onehot: (B, C, W, H, D)
        true_onehot: (B, C, W, H, D)

    Returns
        score: (B, C)
    """
    score = monai.metrics.compute_meandice(
        pred_onehot, 
        true_onehot, 
        include_background
    )
    return score.to("cpu").numpy()


def assd(pred_onehot, true_onehot, include_background=False):
    """
    Params
        pred_onehot: (B, C, W, H, D)
        true_onehot: (B, C, W, H, D)

    Returns
        score: (B, C)
    """
    score = monai.metrics.compute_average_surface_distance(
        pred_onehot, 
        true_onehot, 
        include_background,
        symmetric=True,
    )
    return score.to("cpu").numpy()