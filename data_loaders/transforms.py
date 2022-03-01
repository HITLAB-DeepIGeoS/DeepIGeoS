import torch
import torchio as tio


class CropNonzero(tio.Transform):

    def __init__(self, modality=None):
        super().__init__()
        if modality is None:
            raise ValueError(f"modality must be specified")
        self.modality = modality

    def apply_transform(self, subject):
        images = subject.get_images(intensity_only=False)
        image_data = None
        for image in images:
            if isinstance(image, tio.ScalarImage):
                image_data = image.data
                sum_c = torch.sum(image_data[self.modality, :, :, :], dim=2)
                sum_w = torch.sum(image_data[self.modality, :, :, :], dim=0)
                xs, ys = np.where(sum_c > 0)
                _,  zs = np.where(sum_w > 0)

        if image_data is not None:
            for image in images:
                image.set_data(image.data[self.modality,
                                          min(xs):max(xs), 
                                          min(ys):max(ys),
                                          min(zs):max(zs),].unsqueeze_(dim=0))
        else:
            raise ValueError(f"subject must have ScalarImage object")

        return subject


def get_transform(split):
    transform = [tio.ToCanonical(), tio.Resample(1)]

    if split == "train":
        pass
    elif split == "valid":
        pass
    else:
        raise ValueError(f"split {split} is not supported")

    transform += [tio.RemapLabels({2:1, 3:1, 4:1}), 
                  tio.ZNormalization()]

    return tio.Compose(transform)