import numpy as np
import scipy.ndimage as nd


def rotate_img(
    x: np.ndarray, deg: float, imshape: tuple[int, ...]
) -> np.ndarray:
    return nd.rotate(x.reshape(*imshape), deg, reshape=False).ravel()
