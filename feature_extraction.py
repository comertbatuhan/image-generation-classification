# feature_extraction.py
import numpy as np
import io
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern
from skimage.transform import resize


def extract_hog(image: np.ndarray) -> np.ndarray:
    gray = rgb2gray(image)
    gray = resize(gray, (128, 128), anti_aliasing=True, preserve_range=True).astype(np.float32)

    feats = hog(
        gray,
        pixels_per_cell=(32, 32),   
        cells_per_block=(2, 2),    
        orientations=9,            
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )
    return feats

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
