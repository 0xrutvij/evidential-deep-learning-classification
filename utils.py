import scipy.ndimage as nd
import torch


def rotate_img(x: torch.Tensor, deg: float) -> torch.Tensor:
    return nd.rotate(x.reshape(28, 28), deg, reshape=False).ravel()
