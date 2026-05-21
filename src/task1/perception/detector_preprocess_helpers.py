from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def load_camera_params(param_file: str):
    import yaml
    with open(param_file, 'r') as f:
        data = yaml.safe_load(f)
    K = np.array(data['K'], dtype=np.float64)
    D = np.array(data['D'], dtype=np.float64)
    return K, D


def get_robot_param_file(base_dir: str, robot_id: int) -> str:
    return str(Path(base_dir) / f"camera-params-cr{robot_id:02d}.yaml")


def undistort_image(K: np.ndarray, D: np.ndarray, img: np.ndarray) -> np.ndarray:
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, img.shape[:2][::-1], cv2.CV_16SC2
    )
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def preprocess_base_image(
    img_base: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    img = undistort_image(K, D, img_base)
    return img


def load_and_preprocess_for_robot(img_base: np.ndarray, calibration_dir: str, robot_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    param_file = get_robot_param_file(calibration_dir, robot_id)
    K, D = load_camera_params(param_file)
    processed = preprocess_base_image(img_base, K, D)
    return processed, K, D, param_file
