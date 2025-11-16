#utils.py
import zipfile
from PIL import Image
import numpy as np
from feature_extraction import extract_hog, extract_lbp


def iter_images_from_zip(zip_path, suffix=".png"):
    """
    Generator for reading images from a zip file as RGB images.
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            if not name.lower().endswith(suffix.lower()):
                continue

            with z.open(name) as f:
                img = Image.open(f).convert("RGB")
                yield name, np.array(img)


def iter_features_from_zip(zip_path: str):
    """
    High-level generator:
    - Reads each .png from the zip
    - Extracts HOG + LBP
    - Yields (filename, feature_vector)
    """
    for path, img in iter_images_from_zip(zip_path, suffix=".png"):
        hog_feat = extract_hog(img)
        lbp_feat = extract_lbp(img)
        features = np.hstack([hog_feat, lbp_feat])
        yield path, features


def build_feature_matrix(zip_path: str):
    """
    Consumes the feature generator and builds:
    - X: numpy array (n_samples, n_features)
    - paths: list of image paths
    """
    paths = []
    feat_list = []

    for path, feats in iter_features_from_zip(zip_path):
        paths.append(path)
        feat_list.append(feats)

    X = np.vstack(feat_list)
    return X, np.array(paths)
