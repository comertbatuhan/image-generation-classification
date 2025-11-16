# feature_extraction.py
import numpy as np
import io
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern


def extract_hog(image: np.ndarray) -> np.ndarray:
    """
    image: RGB uint8 array (H, W, 3)
    returns: 1D HOG feature vector
    """
    gray = rgb2gray(image)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(32, 32),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return features


def extract_lbp(image: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:
    """
    image: RGB uint8 array (H, W, 3)
    returns: normalized LBP histogram
    """
    gray = rgb2gray(image)
    gray_u8 = (gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_u8, P=P, R=R, method="uniform")

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, P + 3),
        range=(0, P + 2),
        density=True,
    )
    return hist
