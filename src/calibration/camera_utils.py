from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import yaml


def load_camera_params(yaml_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cam = yaml.safe_load(f)
    K = np.array(cam["K"], dtype=np.float64)
    D = np.array(cam["D"], dtype=np.float64)
    return K, D


def preprocess_base_image(img_base, K, D, undistort_image_fn):
    """
    Canonical preprocessing for calibration:
    raw distorted image -> undistorted only
    """
    if img_base is None:
        raise ValueError("img_base is None")
    img = undistort_image_fn(K, D, img_base)
    return img


def get_processed_base_image(robot, K, D, undistort_image_fn):
    img_base, ts = robot.getImageBase()
    img = preprocess_base_image(img_base, K, D, undistort_image_fn)
    return img, ts
