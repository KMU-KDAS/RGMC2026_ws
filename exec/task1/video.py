from pathlib import Path
import sys
import os
import time

from dotenv import load_dotenv

# =========================
# Project Path Setup
# =========================

# video.py 위치:
# project_root/notebooks/task1/video.py
# 따라서 task1 -> notebooks -> project_root 로 두 단계 올라감
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

SRC_ROOT = PROJECT_ROOT / "src"
CONFIG_ROOT = PROJECT_ROOT / "configs"
DATA_ROOT = PROJECT_ROOT / "data"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

print("===== VSCode DIRECT RUN CONFIG =====")
print("SCRIPT_DIR  :", SCRIPT_DIR)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("SRC_ROOT    :", SRC_ROOT)
print("CONFIG_ROOT :", CONFIG_ROOT)
print("DATA_ROOT   :", DATA_ROOT)


# =========================
# .env Load
# =========================

env_path = PROJECT_ROOT / ".env"

print("\n===== ENV CHECK =====")
print("env path:", env_path)
print("exists  :", env_path.exists())

load_dotenv(env_path)

if "CLOUDGRIPPER_TOKEN" not in os.environ:
    raise RuntimeError(
        "CLOUDGRIPPER_TOKEN not found in .env\n"
        f"확인할 위치: {env_path}"
    )

print("Token loaded:", True)


# =========================
# Imports
# =========================

import cv2
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cloudgripper_api.client.cloudgripper_client import GripperRobot

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


# =========================
# Robot / Config / Output Path
# =========================

ROBOT_NAME = "robot20"

ROBOT_ID = "".join(ch for ch in ROBOT_NAME if ch.isdigit())
if not ROBOT_ID:
    raise ValueError(f"ROBOT_NAME에서 robot id를 파싱하지 못했습니다: {ROBOT_NAME!r}")

CONFIG_PATH = CONFIG_ROOT / "calibration" / f"calibration_robot{ROBOT_ID}.yaml"
OUTPUT_DIR = DATA_ROOT / "map" / ROBOT_NAME

print("\n===== ROBOT CONFIG =====")
print("ROBOT_NAME :", ROBOT_NAME)
print("ROBOT_ID   :", ROBOT_ID)
print("CONFIG_PATH:", CONFIG_PATH)
print("OUTPUT_DIR :", OUTPUT_DIR)

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Calibration config 파일이 없습니다: {CONFIG_PATH}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Robot Connect
# =========================

TOKEN = os.getenv("CLOUDGRIPPER_TOKEN")
if not TOKEN:
    raise RuntimeError("CLOUDGRIPPER_TOKEN not set")

print("\n===== ROBOT CONNECT =====")
robot = GripperRobot(ROBOT_NAME, TOKEN)

state_out = robot.get_state()
print("robot state:")
print(state_out)


# =========================
# Camera Stream
# =========================

print("\n===== CAMERA STREAM START =====")
print("카메라 창에서 q 누르면 종료됨")

try:
    while True:
        image, timestamp = robot.getImageBaseUndistorted()

        if image is None:
            print("[WARN] image is None")
            time.sleep(0.1)
            continue

        cv2.imshow("Cloudgripper top camera stream", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("q 입력됨. 종료합니다.")
            break

finally:
    cv2.destroyAllWindows()
    print("Camera stream closed.")