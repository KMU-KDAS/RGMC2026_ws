from pathlib import Path
import sys
import os
from dotenv import load_dotenv

#ROBOT_ID = 22
#ROBOT_NAME = f"robot{ROBOT_ID}"
ROBOT_NAME=f"competition"

# =========================
# Project Path Setup
# =========================

# __file__ looks at where video.py is actually located on your disk
SCRIPT_DIR = Path(__file__).resolve().parent

# Go up one level from notebooks/ -> project root
PROJECT_ROOT = SCRIPT_DIR.parents[0]


SRC_ROOT = PROJECT_ROOT / "src"
CONFIG_ROOT = PROJECT_ROOT / "configs"
DATA_ROOT = PROJECT_ROOT / "data"
CLOUDGRIPPER_CLIENT_DIR = SRC_ROOT / "cloudgripper-api" / "client"


if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

sys.path.insert(0, str(CLOUDGRIPPER_CLIENT_DIR))

print("PROJECT_ROOT:", PROJECT_ROOT)

# =========================
# .env Load
# =========================

env_path = PROJECT_ROOT / ".env"
print("env path:", env_path)
print("ex   ists:", env_path.exists())

load_dotenv(env_path)

# 확인
if "CLOUDGRIPPER_TOKEN" not in os.environ:
    raise RuntimeError("CLOUDGRIPPER_TOKEN not found in .env")

print("Token loaded:", True)



# 1. Imports: 현재 디스크 구조 기준
import os
import time
from pathlib import Path

import cv2
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cloudgripper_client import GripperRobot
from calibration.config import calibration_config_from_yaml
from calibration.detector import YOLOGripperDetector
from calibration.collector import collect_calibration
from calibration.build_lookup_table import (
    parse_dataset,
    build_bidirectional_lookup,
    save_lookup_table,
)
from calibration.pixel_to_workspace import PixelToWorkspaceMapper
from calibration.visualize_workspace_grid import draw_grid_overlay, load_lut
from calibration.notebook_helpers import show_bgr, load_csv_preview, list_failed_points


# 2. config / output 경로 설정

#CONFIG_PATH = CONFIG_ROOT / "calibration" / f"calibration_robot{ROBOT_ID}.yaml"
OUTPUT_DIR = DATA_ROOT / "map" / ROBOT_NAME

print("ROBOT_NAME :", ROBOT_NAME)
#print("ROBOT_ID   :", ROBOT_ID)
#print("CONFIG_PATH:", CONFIG_PATH)
print("OUTPUT_DIR :", OUTPUT_DIR)

#if not CONFIG_PATH.exists():
#    raise FileNotFoundError(f"Calibration config 파일이 없습니다: {CONFIG_PATH}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 3. 토큰은 환경 변수에서만 읽기s
TOKEN = os.getenv("CLOUDGRIPPER_TOKEN")
if not TOKEN:
    raise RuntimeError("CLOUDGRIPPER_TOKEN not set")

robot = GripperRobot(ROBOT_NAME, TOKEN)

state_out = robot.get_state()
print(state_out)

import cv2

while True:
    image, timestamp = robot.getImageBaseUndistorted()
    cv2.imshow("Cloudgripper top camera stream", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
