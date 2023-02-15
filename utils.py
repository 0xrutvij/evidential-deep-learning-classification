import numpy as np
import scipy.ndimage as nd


def rotate_img(x: np.ndarray, deg: float) -> np.ndarray:
    return nd.rotate(x.reshape(28, 28), deg, reshape=False).ravel()
