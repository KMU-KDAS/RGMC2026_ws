from pathlib import Path
import sys
import os
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

import cv2
import numpy as np


# =========================================================
# Recording / overlay settings
# =========================================================

# The recorder will measure the real getImageBaseUndistorted() stream FPS
# before opening the video writer.
AUTO_MEASURE_STREAM_FPS = True
FPS_MEASURE_FRAMES = 60
FPS_MEASURE_MIN_FRAMES = 10
FPS_FALLBACK = 15.0

# Clamp measured FPS to avoid weird values from API jitter.
MIN_OUTPUT_FPS = 5.0
MAX_OUTPUT_FPS = 60.0

STATUS_POLL_SEC = 0.50        # eval_status() polling interval
OBJECT_POLL_SEC = 0.50        # eval_object() polling interval
TARGET_RETRY_SEC = 2.00       # retry eval_target() if target was not available
SHOW_LIVE_PREVIEW = False     # keep False if you only want to save video
LIVE_WINDOW_NAME = "CloudGripper eval overlay recorder"

# If you want the recording loop to behave as close as possible to your simple
# streaming code, set this False. Eval API calls can slightly reduce capture FPS.
ENABLE_EVAL_OVERLAY = True

# Task2 official score proxy in pixel domain:
# score = max(0, 1 - RMSE / 220), based on the Task2 notebook logic.
TASK2_RMSE_SCORE_DENOM_PX = 220.0

# BGR colors used by OpenCV
COLOR_TARGET = (0, 255, 255)       # yellow
COLOR_CURRENT = (0, 255, 0)        # green
COLOR_TEXT = (255, 255, 255)       # white
COLOR_WARN = (0, 128, 255)         # orange
COLOR_PANEL = (0, 0, 0)            # black


# =========================================================
# User Input
# =========================================================

def parse_robot_id(value: str) -> int:
    """Parse robot ID. -1 means competition mode."""
    try:
        return int(value.strip())
    except Exception:
        raise ValueError(
            "Invalid robot ID: {}. Use a number, for example 19, 22, or -1 for competition.".format(value)
        )


def parse_task_id(value: str) -> int:
    value = value.strip().lower().replace("task", "")
    if value not in {"1", "2"}:
        raise ValueError("Invalid task selection: {}. Use 1 or 2.".format(value))
    return int(value)


robot_id_input = input("Enter robot ID (-1 for competition): ").strip()
task_id_input = input("Enter task number (1 or 2): ").strip()

ROBOT_ID = parse_robot_id(robot_id_input)
TASK_ID = parse_task_id(task_id_input)
TASK_NAME = "task{}".format(TASK_ID)
COMPETITION_MODE = ROBOT_ID == -1

if COMPETITION_MODE:
    ROBOT_NAME = "competition"
else:
    ROBOT_NAME = "robot{}".format(ROBOT_ID)

print("Using TASK_ID          :", TASK_ID)
print("Using TASK_NAME        :", TASK_NAME)
print("Using ROBOT_ID         :", ROBOT_ID)
print("Using ROBOT_NAME       :", ROBOT_NAME)
print("Using COMPETITION_MODE :", COMPETITION_MODE)


# =========================================================
# Project Path Setup
# =========================================================

def find_project_root(start: Path) -> Path:
    """Find RGMC project root by walking upward from this script location."""
    start = start.resolve()

    for p in [start] + list(start.parents):
        if (p / "src").exists() and (p / "configs").exists() and (p / "data").exists():
            return p

    # Fallback to the original video.py convention.
    return start.parent


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(SCRIPT_DIR)

SRC_ROOT = PROJECT_ROOT / "src"
CONFIG_ROOT = PROJECT_ROOT / "configs"
DATA_ROOT = PROJECT_ROOT / "data"
CLOUDGRIPPER_CLIENT_DIR = SRC_ROOT / "cloudgripper-api" / "client"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if str(CLOUDGRIPPER_CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(CLOUDGRIPPER_CLIENT_DIR))

print("PROJECT_ROOT:", PROJECT_ROOT)


# =========================================================
# .env Load
# =========================================================

env_path = PROJECT_ROOT / ".env"
print("env path:", env_path)
print("exists  :", env_path.exists())

load_dotenv(env_path)

if "CLOUDGRIPPER_TOKEN" not in os.environ:
    raise RuntimeError("CLOUDGRIPPER_TOKEN not found in .env")

print("Token loaded:", True)


# =========================================================
# Imports from Project
# =========================================================

from cloudgripper_client import GripperRobot


# =========================================================
# Basic helpers
# =========================================================

def unpack_image_result(result: Any) -> Tuple[Optional[np.ndarray], Any]:
    """Handle both image-only and (image, timestamp) return formats."""
    if isinstance(result, tuple):
        if len(result) == 0:
            return None, None
        image = result[0]
        timestamp = result[1] if len(result) > 1 else None
        return image, timestamp

    return result, None


def safe_call(label: str, fn, default=None):
    """Call a robot API function without killing the recorder on transient API errors."""
    try:
        return fn()
    except Exception as e:
        print("[WARN] {} failed: {}".format(label, repr(e)))
        return default


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def get_undistorted_image(robot) -> Tuple[Optional[np.ndarray], Any]:
    """Use the same image source as your streaming code."""
    return unpack_image_result(
        safe_call(
            "getImageBaseUndistorted",
            robot.getImageBaseUndistorted,
            default=(None, None),
        )
    )


def measure_stream_fps(robot) -> Tuple[float, Optional[np.ndarray], Any]:
    """Measure the real FPS of getImageBaseUndistorted(), like your streaming code.

    Returns:
        measured_fps
        last_valid_image
        last_valid_timestamp
    """
    print("Measuring stream FPS using getImageBaseUndistorted()...")

    frames = 0
    last_image = None
    last_timestamp = None

    t0 = time.monotonic()

    while frames < FPS_MEASURE_FRAMES:
        image, timestamp = get_undistorted_image(robot)

        if image is None:
            print("[WARN] Empty image during FPS measurement.")
            time.sleep(0.05)
            continue

        last_image = image
        last_timestamp = timestamp
        frames += 1

        if frames >= FPS_MEASURE_MIN_FRAMES:
            elapsed = time.monotonic() - t0
            if elapsed >= 1.0:
                break

    elapsed = time.monotonic() - t0

    if frames <= 1 or elapsed <= 0:
        print("[WARN] Could not measure stream FPS. Using fallback:", FPS_FALLBACK)
        return FPS_FALLBACK, last_image, last_timestamp

    measured_fps = frames / elapsed
    measured_fps = clamp(measured_fps, MIN_OUTPUT_FPS, MAX_OUTPUT_FPS)

    print("Measured stream frames:", frames)
    print("Measured stream elapsed: {:.3f} s".format(elapsed))
    print("Measured stream FPS: {:.2f}".format(measured_fps))

    return measured_fps, last_image, last_timestamp


# =========================================================
# Geometry parsing helpers
# =========================================================

def _points_from_points_like(points_like: Sequence[Any]) -> np.ndarray:
    pts = []

    for p in points_like:
        if isinstance(p, dict):
            if "x" not in p or "y" not in p:
                continue
            pts.append((float(p["x"]), float(p["y"])))
        else:
            if len(p) < 2:
                continue
            pts.append((float(p[0]), float(p[1])))

    return np.asarray(pts, dtype=np.float32)


def extract_eval_geometry_points(payload: Any) -> Tuple[Optional[np.ndarray], str]:
    """Extract 2D pixel points from Task1 or Task2 eval payloads."""
    if not isinstance(payload, dict):
        return None, "invalid"

    geom = payload.get("geometry", {})

    if not isinstance(geom, dict):
        return None, "invalid_geometry"

    if "segmented_points" in geom:
        pts = _points_from_points_like(geom.get("segmented_points", []))
        return pts if len(pts) > 0 else None, "segmented_points"

    if "points" in geom:
        pts = _points_from_points_like(geom.get("points", []))
        return pts if len(pts) > 0 else None, "points"

    return None, "missing_points"


def is_polyline_geometry(task_id: int, source_name: str, n_points: int) -> bool:
    """Task2 rope is an open polyline; Task1 object target is a closed polygon."""
    if task_id == 2:
        return True

    if source_name == "segmented_points":
        return True

    return False


def scale_points(points: Optional[np.ndarray], scale_x: float, scale_y: float) -> Optional[np.ndarray]:
    if points is None:
        return None

    out = np.asarray(points, dtype=np.float32).copy()

    if out.ndim != 2 or out.shape[1] != 2:
        return None

    out[:, 0] *= float(scale_x)
    out[:, 1] *= float(scale_y)

    return out


def filter_finite_points(points: Optional[np.ndarray], width: int, height: int) -> Optional[np.ndarray]:
    if points is None:
        return None

    pts = np.asarray(points, dtype=np.float32)

    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        return None

    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]

    if len(pts) == 0:
        return None

    pts[:, 0] = np.clip(pts[:, 0], -10000, width + 10000)
    pts[:, 1] = np.clip(pts[:, 1], -10000, height + 10000)

    return pts


def draw_points_geometry(
    frame: np.ndarray,
    points: Optional[np.ndarray],
    task_id: int,
    source_name: str,
    color: Tuple[int, int, int],
    label: str,
    thickness: int = 2,
):
    """Draw Task1 polygon or Task2 rope polyline on a frame."""
    h, w = frame.shape[:2]
    pts = filter_finite_points(points, w, h)

    if pts is None or len(pts) < 1:
        return

    pts_int = np.round(pts).astype(np.int32)
    open_polyline = is_polyline_geometry(task_id, source_name, len(pts_int))

    if len(pts_int) >= 2:
        cv2.polylines(
            frame,
            [pts_int.reshape((-1, 1, 2))],
            isClosed=not open_polyline,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    radius = 3 if task_id == 2 else 4

    for idx, (u, v) in enumerate(pts_int):
        cv2.circle(frame, (int(u), int(v)), radius, color, -1, lineType=cv2.LINE_AA)

        if task_id == 2 and idx in {0, len(pts_int) - 1}:
            cv2.putText(
                frame,
                str(idx),
                (int(u) + 4, int(v) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

    anchor = pts_int[0]
    cv2.putText(
        frame,
        label,
        (int(anchor[0]) + 6, int(anchor[1]) + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        color,
        1,
        cv2.LINE_AA,
    )


def polygon_iou_px(
    poly_a: Optional[np.ndarray],
    poly_b: Optional[np.ndarray],
    image_shape: Tuple[int, int],
) -> Optional[float]:
    """Compute image-space polygon IoU for Task1 eval_target/eval_object overlays."""
    if poly_a is None or poly_b is None:
        return None

    a = np.asarray(poly_a, dtype=np.float32)
    b = np.asarray(poly_b, dtype=np.float32)

    if len(a) < 3 or len(b) < 3:
        return None

    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None

    height, width = image_shape[:2]

    mask_a = np.zeros((height, width), dtype=np.uint8)
    mask_b = np.zeros((height, width), dtype=np.uint8)

    ai = np.round(a).astype(np.int32).reshape((-1, 1, 2))
    bi = np.round(b).astype(np.int32).reshape((-1, 1, 2))

    cv2.fillPoly(mask_a, [ai], 1)
    cv2.fillPoly(mask_b, [bi], 1)

    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union <= 0:
        return None

    return float(inter) / float(union)


def task2_rmse_and_score_px(
    target_points: Optional[np.ndarray],
    current_points: Optional[np.ndarray],
) -> Tuple[Optional[float], Optional[float]]:
    """Compute Task2 pixel RMSE and score proxy from 20-point target/current observations."""
    if target_points is None or current_points is None:
        return None, None

    tgt = np.asarray(target_points, dtype=np.float32)
    cur = np.asarray(current_points, dtype=np.float32)

    if tgt.ndim != 2 or cur.ndim != 2 or tgt.shape[1] != 2 or cur.shape[1] != 2:
        return None, None

    if len(tgt) == 0 or len(cur) == 0:
        return None, None

    n = min(len(tgt), len(cur))
    tgt = tgt[:n]
    cur = cur[:n]

    mask = np.isfinite(tgt).all(axis=1) & np.isfinite(cur).all(axis=1)

    if not np.any(mask):
        return None, None

    diff = cur[mask] - tgt[mask]
    rmse = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    score = max(0.0, 1.0 - rmse / TASK2_RMSE_SCORE_DENOM_PX)

    return rmse, score


def find_first_number_by_keys(obj: Any, keys: Sequence[str]) -> Optional[float]:
    """Recursively search eval_status payload for one of several possible metric keys."""
    key_set = {str(k).lower() for k in keys}

    def _walk(x: Any) -> Optional[float]:
        if isinstance(x, dict):
            for k, v in x.items():
                if str(k).lower() in key_set:
                    try:
                        return float(v)
                    except Exception:
                        pass

                out = _walk(v)

                if out is not None:
                    return out

        elif isinstance(x, (list, tuple)):
            for item in x:
                out = _walk(item)

                if out is not None:
                    return out

        return None

    return _walk(obj)


def get_status_text(status_payload: Any) -> Optional[str]:
    """Return normalized eval_status.status text, if present."""
    if not isinstance(status_payload, dict):
        return None

    status = status_payload.get("status", None)

    if status is None:
        return None

    return str(status).strip().lower().replace(" ", "_")


def eval_run_present_from_status(status_payload: Any) -> bool:
    """Infer whether an eval/competition run currently exists."""
    if not isinstance(status_payload, dict):
        return False

    status_text = get_status_text(status_payload)

    absent_statuses = {
        "not_started",
        "not_running",
        "no_run",
        "no_active_run",
        "idle",
        "ready",
        "unknown",
        "eval_status_failed",
        "failed_to_get_eval_status",
    }

    present_statuses = {
        "running",
        "completed",
        "complete",
        "finished",
        "done",
        "active",
        "started",
        "success",
    }

    if status_text in absent_statuses:
        return False

    if status_text in present_statuses:
        return True

    metric_keys = [
        "current_iou",
        "iou",
        "iou_score",
        "current_score",
        "score",
        "final_score",
        "elapsed_time",
        "remaining_time",
        "time_remaining",
        "remain",
    ]

    if find_first_number_by_keys(status_payload, metric_keys) is not None:
        return True

    return False


def format_float(value: Any, digits: int = 3) -> str:
    try:
        return ("{:." + str(digits) + "f}").format(float(value))
    except Exception:
        return str(value)


def build_overlay_lines(
    task_id: int,
    robot_name: str,
    status_payload: Any,
    local_elapsed: float,
    target_points: Optional[np.ndarray],
    current_points: Optional[np.ndarray],
    frame_shape: Tuple[int, int, int],
    capture_fps_estimate: Optional[float] = None,
) -> List[Tuple[str, Tuple[int, int, int]]]:
    status = "?"

    if isinstance(status_payload, dict):
        status = str(status_payload.get("status", "?"))

    lines = [
        ("{} | {}".format(robot_name, "task{}".format(task_id)), COLOR_TEXT),
        ("status: {}".format(status), COLOR_TEXT),
        ("local t: {:.1f}s".format(local_elapsed), COLOR_TEXT),
    ]

    if capture_fps_estimate is not None:
        lines.append(("capture FPS: {:.1f}".format(capture_fps_estimate), COLOR_TEXT))

    if task_id == 1:
        computed_iou = polygon_iou_px(target_points, current_points, frame_shape[:2])

        if computed_iou is not None:
            lines.append(("IoU(px): {}".format(format_float(computed_iou, 3)), COLOR_TEXT))

        eval_iou = find_first_number_by_keys(
            status_payload,
            ["current_iou", "iou", "iou_score", "current_score", "score", "final_score"],
        )

        if eval_iou is not None:
            lines.append(("eval: {}".format(format_float(eval_iou, 3)), COLOR_TEXT))

    else:
        rmse_px, score_px = task2_rmse_and_score_px(target_points, current_points)

        if rmse_px is not None:
            lines.append(("RMSE(px): {}".format(format_float(rmse_px, 1)), COLOR_TEXT))

        if score_px is not None:
            lines.append(("score(px): {}".format(format_float(score_px, 3)), COLOR_TEXT))

        eval_score = find_first_number_by_keys(
            status_payload,
            ["current_score", "score", "final_score"],
        )

        if eval_score is not None:
            lines.append(("eval: {}".format(format_float(eval_score, 3)), COLOR_TEXT))

    if target_points is None:
        lines.append(("target: unavailable", COLOR_WARN))

    if current_points is None:
        lines.append(("object: unavailable", COLOR_WARN))

    return lines


def draw_corner_overlay(frame: np.ndarray, lines: List[Tuple[str, Tuple[int, int, int]]]) -> np.ndarray:
    """Draw compact metric panel in top-left corner."""
    if not lines:
        return frame

    x0 = 8
    y0 = 22
    line_h = 19
    pad = 6
    font_scale = 0.50
    thickness = 1

    max_w = 0

    for text, _ in lines:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        max_w = max(max_w, tw)

    panel_w = min(frame.shape[1] - 1, x0 + max_w + 2 * pad)
    panel_h = min(frame.shape[0] - 1, y0 + line_h * len(lines) + pad)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, (text, color) in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (x0, y0 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    return frame


# =========================================================
# Output Path Setup
# =========================================================

OUTPUT_DIR = DATA_ROOT / "videos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
VIDEO_PATH = OUTPUT_DIR / "video_{}_{}.mp4".format(TASK_NAME, date_time_str)

print("OUTPUT_DIR:", OUTPUT_DIR)
print("VIDEO_PATH:", VIDEO_PATH)


# =========================================================
# Robot Setup
# =========================================================

TOKEN = os.getenv("CLOUDGRIPPER_TOKEN")

if not TOKEN:
    raise RuntimeError("CLOUDGRIPPER_TOKEN not set")

robot = GripperRobot(ROBOT_NAME, TOKEN)

state_out = safe_call("get_state", robot.get_state, default=None)
print("Initial robot state:")
print(state_out)


# =========================================================
# Passive eval/competition run detection
# =========================================================

RUN_PRESENT = False
RUN_PRESENT_SOURCE = "not_detected"

latest_target_payload = None
latest_target_points = None
latest_target_source = "run_not_present"

latest_object_payload = None
latest_object_points = None
latest_object_source = "run_not_present"

latest_status_payload = {}

if ENABLE_EVAL_OVERLAY:
    latest_status_payload = safe_call("eval_status", robot.eval_status, default={})

    if not RUN_PRESENT and eval_run_present_from_status(latest_status_payload):
        RUN_PRESENT = True
        RUN_PRESENT_SOURCE = "detected_from_eval_status"

    if RUN_PRESENT:
        latest_target_payload = safe_call("eval_target", robot.eval_target, default=None)
        latest_target_points, latest_target_source = extract_eval_geometry_points(latest_target_payload)

        latest_object_payload = safe_call("eval_object", robot.eval_object, default=None)
        latest_object_points, latest_object_source = extract_eval_geometry_points(latest_object_payload)

        print("Eval/competition run detected. Overlay enabled. Source:", RUN_PRESENT_SOURCE)
        print(
            "Initial target source:",
            latest_target_source,
            "points:",
            None if latest_target_points is None else len(latest_target_points),
        )
        print(
            "Initial object source:",
            latest_object_source,
            "points:",
            None if latest_object_points is None else len(latest_object_points),
        )
    else:
        print("No eval/competition run detected. Target, current object, and metrics overlays are disabled.")

    print("Initial eval_status:", latest_status_payload)
else:
    print("ENABLE_EVAL_OVERLAY is False. Recording will match the simple stream loop more closely.")


# =========================================================
# Measure stream FPS and get first image
# =========================================================

if AUTO_MEASURE_STREAM_FPS:
    OUTPUT_FPS, image, timestamp = measure_stream_fps(robot)
else:
    OUTPUT_FPS = FPS_FALLBACK
    image, timestamp = get_undistorted_image(robot)

if image is None:
    raise RuntimeError(
        "Failed to get first image from robot.getImageBaseUndistorted(). "
        "Check whether the simple streaming code works for this same robot."
    )

height, width = image.shape[:2]

print("Image size:", width, "x", height)
print("Using output FPS:", OUTPUT_FPS)


# =========================================================
# Video Recording Setup
# =========================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    str(VIDEO_PATH),
    fourcc,
    OUTPUT_FPS,
    (width, height),
)

if not video_writer.isOpened():
    raise RuntimeError("Failed to open video writer: {}".format(VIDEO_PATH))

print("Recording started.")
print("Output FPS was measured from the same getImageBaseUndistorted() stream used by your preview code.")
print("Press Ctrl+C to stop and save the video.")

frame_count = 0
start_time = time.time()
last_status_poll = 0.0
last_object_poll = 0.0
last_target_retry = 0.0

try:
    while True:
        image, timestamp = get_undistorted_image(robot)

        if image is None:
            print("Warning: received empty image, skipping frame.")
            time.sleep(0.05)
            continue

        src_h, src_w = image.shape[:2]

        if src_w != width or src_h != height:
            frame = cv2.resize(image, (width, height))
            scale_x = float(width) / float(src_w)
            scale_y = float(height) / float(src_h)
        else:
            frame = image.copy()
            scale_x = 1.0
            scale_y = 1.0

        now = time.monotonic()
        elapsed = time.time() - start_time
        capture_fps_estimate = frame_count / max(elapsed, 1e-6)

        if ENABLE_EVAL_OVERLAY:
            if now - last_status_poll >= STATUS_POLL_SEC:
                st = safe_call("eval_status", robot.eval_status, default=None)

                if st is not None:
                    latest_status_payload = st
                    status_says_run_present = eval_run_present_from_status(st)

                    if status_says_run_present and not RUN_PRESENT:
                        RUN_PRESENT = True
                        RUN_PRESENT_SOURCE = "detected_from_eval_status"
                        print("Eval/competition run detected during recording. Overlay enabled.")

                    elif RUN_PRESENT and not status_says_run_present:
                        RUN_PRESENT = False
                        RUN_PRESENT_SOURCE = "not_present_from_eval_status"

                        latest_target_points = None
                        latest_object_points = None
                        latest_target_source = "run_not_present"
                        latest_object_source = "run_not_present"

                        print("Eval/competition run no longer detected. Overlay disabled.")

                last_status_poll = now

            if RUN_PRESENT and now - last_object_poll >= OBJECT_POLL_SEC:
                obj = safe_call("eval_object", robot.eval_object, default=None)

                if obj is not None:
                    pts, source = extract_eval_geometry_points(obj)
                    latest_object_payload = obj
                    latest_object_points = pts
                    latest_object_source = source

                last_object_poll = now

            if RUN_PRESENT and latest_target_points is None and now - last_target_retry >= TARGET_RETRY_SEC:
                tgt = safe_call("eval_target", robot.eval_target, default=None)

                if tgt is not None:
                    pts, source = extract_eval_geometry_points(tgt)
                    latest_target_payload = tgt
                    latest_target_points = pts
                    latest_target_source = source

                last_target_retry = now

            if RUN_PRESENT:
                draw_target_points = scale_points(latest_target_points, scale_x, scale_y)
                draw_object_points = scale_points(latest_object_points, scale_x, scale_y)

                draw_points_geometry(
                    frame,
                    draw_object_points,
                    TASK_ID,
                    latest_object_source,
                    COLOR_CURRENT,
                    "current",
                    thickness=2,
                )

                draw_points_geometry(
                    frame,
                    draw_target_points,
                    TASK_ID,
                    latest_target_source,
                    COLOR_TARGET,
                    "target",
                    thickness=2,
                )

                overlay_lines = build_overlay_lines(
                    TASK_ID,
                    ROBOT_NAME,
                    latest_status_payload,
                    elapsed,
                    draw_target_points,
                    draw_object_points,
                    frame.shape,
                    capture_fps_estimate=capture_fps_estimate,
                )

                draw_corner_overlay(frame, overlay_lines)

        video_writer.write(frame)
        frame_count += 1

        if SHOW_LIVE_PREVIEW:
            cv2.imshow(LIVE_WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("q pressed. Stopping recording...")
                break

        if frame_count % 30 == 0:
            print(
                "Recorded frames: {}, elapsed: {:.1f} s, effective capture FPS: {:.2f}, encoded FPS: {:.2f}".format(
                    frame_count,
                    elapsed,
                    capture_fps_estimate,
                    OUTPUT_FPS,
                )
            )

except KeyboardInterrupt:
    print("\nCtrl+C detected. Stopping recording...")

finally:
    video_writer.release()

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    elapsed = time.time() - start_time
    effective_capture_fps = frame_count / max(elapsed, 1e-6)
    expected_video_duration = frame_count / max(OUTPUT_FPS, 1e-6)

    print("Video saved successfully.")
    print("Path:", VIDEO_PATH)
    print("Total frames:", frame_count)
    print("Elapsed real time: {:.2f} s".format(elapsed))
    print("Effective capture FPS: {:.2f}".format(effective_capture_fps))
    print("Encoded FPS: {:.2f}".format(OUTPUT_FPS))
    print("Expected playback duration: {:.2f} s".format(expected_video_duration))

    if abs(expected_video_duration - elapsed) > 2.0:
        print(
            "[WARN] Playback duration differs from real time by {:.2f} s. "
            "If this still looks sped up, set ENABLE_EVAL_OVERLAY = False "
            "or reduce STATUS_POLL_SEC / OBJECT_POLL_SEC API overhead.".format(
                abs(expected_video_duration - elapsed)
            )
        )