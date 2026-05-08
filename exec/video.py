from pathlib import Path
import sys
import os
import time
from datetime import datetime

from dotenv import load_dotenv

import cv2
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FPS = 10.0

# =========================
# User Input
# =========================

robot_id_input = input("Enter robot ID: ").strip()

if not robot_id_input.isdigit():
    raise ValueError(f"Invalid robot ID: {robot_id_input}")

ROBOT_ID = int(robot_id_input)

if ROBOT_ID == -1:
    ROBOT_NAME=f"competition"
else:
    ROBOT_NAME = f"robot{ROBOT_ID}"

print("Using ROBOT_ID  :", ROBOT_ID)
print("Using ROBOT_NAME:", ROBOT_NAME)


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

if str(CLOUDGRIPPER_CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(CLOUDGRIPPER_CLIENT_DIR))

print("PROJECT_ROOT:", PROJECT_ROOT)


# =========================
# .env Load
# =========================

env_path = PROJECT_ROOT / ".env"
print("env path:", env_path)
print("exists  :", env_path.exists())

load_dotenv(env_path)

if "CLOUDGRIPPER_TOKEN" not in os.environ:
    raise RuntimeError("CLOUDGRIPPER_TOKEN not found in .env")

print("Token loaded:", True)


# =========================
# Imports from Project
# =========================

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


# =========================
# Output Path Setup
# =========================

OUTPUT_DIR = DATA_ROOT / "videos" / ROBOT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
VIDEO_PATH = OUTPUT_DIR / f"video_{date_time_str}.mp4"

print("OUTPUT_DIR:", OUTPUT_DIR)
print("VIDEO_PATH:", VIDEO_PATH)


# =========================
# Robot Setup
# =========================

TOKEN = os.getenv("CLOUDGRIPPER_TOKEN")
if not TOKEN:
    raise RuntimeError("CLOUDGRIPPER_TOKEN not set")

robot = GripperRobot(ROBOT_NAME, TOKEN)

state_out = robot.get_state()
print("Initial robot state:")
print(state_out)


# =========================
# Video Recording Setup
# =========================



# Get first frame to determine video size
image, timestamp = robot.getImageBaseUndistorted()

if image is None:
    raise RuntimeError("Failed to get first image from robot.")

height, width = image.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(
    str(VIDEO_PATH),
    fourcc,
    FPS,
    (width, height),
)

if not video_writer.isOpened():
    raise RuntimeError(f"Failed to open video writer: {VIDEO_PATH}")

print("Recording started.")
print("Press Ctrl+C to stop and save the video.")

frame_count = 0
start_time = time.time()

try:
    while True:
        image, timestamp = robot.getImageBaseUndistorted()

        if image is None:
            print("Warning: received empty image, skipping frame.")
            continue

        # Make sure image size matches the video writer size
        if image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, (width, height))

        video_writer.write(image)
        frame_count += 1

        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f"Recorded frames: {frame_count}, elapsed time: {elapsed:.1f} s")

except KeyboardInterrupt:
    print("\nCtrl+C detected. Stopping recording...")

finally:
    video_writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print("Video saved successfully.")
    print("Path:", VIDEO_PATH)
    print("Total frames:", frame_count)
    print(f"Elapsed time: {elapsed:.2f} s")