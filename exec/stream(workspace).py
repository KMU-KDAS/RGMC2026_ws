from pathlib import Path
import sys
import os
import time

import cv2
import numpy as np
from dotenv import load_dotenv

# =========================
# Robot ID Input
# =========================
robot_id_input = input("Enter robot ID: ").strip()

if not robot_id_input.lstrip("-").isdigit():
    raise ValueError(f"Invalid robot ID: {robot_id_input}")

ROBOT_ID = int(robot_id_input)

if ROBOT_ID == -1:
    ROBOT_NAME = "competition"
else:
    ROBOT_NAME = f"robot{ROBOT_ID}"

print("Using ROBOT_ID  :", ROBOT_ID)
print("Using ROBOT_NAME:", ROBOT_NAME)

# =========================
# Project Path Setup
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent

# 이 파일이 exec/stream.py 같은 위치에 있다고 가정
# exec 폴더 기준 한 단계 위가 프로젝트 루트
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
# Imports after path setup
# =========================
from cloudgripper_client import GripperRobot

try:
    from pixel_to_workspace import PixelToWorkspaceMapper
except Exception:
    try:
        from calibration.pixel_to_workspace import PixelToWorkspaceMapper
    except Exception:
        PixelToWorkspaceMapper = None

# =========================
# Output dir
# =========================
OUTPUT_DIR = DATA_ROOT / "map" / ROBOT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("ROBOT_NAME :", ROBOT_NAME)
print("OUTPUT_DIR :", OUTPUT_DIR)

# =========================
# Token / Robot
# =========================
TOKEN = os.getenv("CLOUDGRIPPER_TOKEN")
if not TOKEN:
    raise RuntimeError("CLOUDGRIPPER_TOKEN not set")

robot = GripperRobot(ROBOT_NAME, TOKEN)

state_out = robot.get_state()
print("[get_state]", state_out)

# =========================================================
# Optional: evaluation start
# =========================================================
# target이 안 나오는 경우 eval_start가 필요할 수 있음.
# 이미 evaluation이 진행 중이면 에러가 날 수 있어서 try 처리.
AUTO_EVAL_START = False

if AUTO_EVAL_START:
    try:
        out = robot.eval_start()
        print("[eval_start]", out)
    except Exception as e:
        print("[WARN] eval_start failed:", repr(e))


# =========================================================
# Helper functions
# =========================================================
def unwrap_payload(payload):
    """
    robot API 결과가 tuple / dict / None 등으로 올 수 있어서 정리
    """
    if payload is None:
        return None

    if isinstance(payload, tuple):
        if len(payload) > 0:
            payload = payload[0]
        else:
            return None

    return payload


def extract_eval_points(payload):
    """
    eval_target(), eval_object() 결과에서
    geometry.points -> Nx2 pixel 좌표 배열을 뽑는다.

    예상 형태:
    {
        "coordinate_space": "undistorted_pixel_2d",
        "geometry": {
            "points": [{"x": ..., "y": ...}, ...]
        }
    }
    """
    payload = unwrap_payload(payload)
    if payload is None or not isinstance(payload, dict):
        return None

    geom = payload.get("geometry", None)
    if geom is None or not isinstance(geom, dict):
        return None

    points = geom.get("points", None)
    if points is None:
        return None

    uv = []
    for p in points:
        if isinstance(p, dict):
            x = p.get("x", None)
            y = p.get("y", None)
        else:
            try:
                x, y = p[0], p[1]
            except Exception:
                continue

        if x is None or y is None:
            continue

        uv.append([float(x), float(y)])

    if len(uv) == 0:
        return None

    return np.asarray(uv, dtype=np.float32)


def polygon_center(points):
    if points is None or len(points) == 0:
        return None
    pts = np.asarray(points, dtype=np.float32)
    return np.mean(pts, axis=0)


def draw_polygon_overlay(
    image,
    points,
    color=(0, 0, 255),
    label="target",
    closed=True,
    thickness=2,
    draw_vertices=True,
):
    """
    이미지 위에 닫힌 다각형(Task1용)을 그림
    """
    if points is None:
        return image

    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        return image

    h, w = image.shape[:2]
    pts[:, 0] = np.clip(pts[:, 0], -10000, w + 10000)
    pts[:, 1] = np.clip(pts[:, 1], -10000, h + 10000)

    pts_i = np.round(pts).astype(np.int32)

    if len(pts_i) >= 2:
        cv2.polylines(
            image,
            [pts_i.reshape((-1, 1, 2))],
            isClosed=closed,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    if draw_vertices:
        for idx, (u, v) in enumerate(pts_i):
            cv2.circle(image, (int(u), int(v)), 4, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(
                image,
                str(idx),
                (int(u) + 5, int(v) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    center = polygon_center(pts_i)
    if center is not None:
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(image, (cx, cy), 5, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (cx + 8, cy + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return image


def safe_get_target_points(robot):
    try:
        payload = robot.eval_target()
        return extract_eval_points(payload), payload
    except Exception as e:
        print("[WARN] eval_target failed:", repr(e))
        return None, None


def safe_get_current_points(robot):
    """
    현재 물체 geometry를 가져온다.
    Task1 코드에서 eval_object()가 있는 경우 그걸 사용.
    """
    try:
        payload = robot.eval_object()
        return extract_eval_points(payload), payload
    except Exception:
        return None, None


def safe_get_eval_status(robot):
    try:
        status = robot.eval_status()
        status = unwrap_payload(status)
        return status if isinstance(status, dict) else None
    except Exception:
        return None


def get_image_undistorted(robot):
    """
    getImageBaseUndistorted() 반환 형식 방어
    보통 (image, timestamp)
    """
    out = robot.getImageBaseUndistorted()

    if isinstance(out, tuple):
        if len(out) >= 2:
            return out[0], out[1]
        elif len(out) == 1:
            return out[0], None
        else:
            return None, None

    return out, None


def get_status_value(status, *keys, default=None):
    """
    eval_status() key 이름이 코드/서버 버전에 따라 조금 다를 수 있어서
    여러 key를 순서대로 확인한다.
    """
    if not isinstance(status, dict):
        return default

    for key in keys:
        value = status.get(key, None)
        if value is not None:
            return value

    return default


def format_remaining_time(seconds_value):
    """
    남은 시간을 mm:ss.s 형태로 변환.
    """
    try:
        seconds_value = float(seconds_value)
    except Exception:
        return None

    if seconds_value < 0:
        seconds_value = 0.0

    minutes = int(seconds_value // 60)
    seconds = seconds_value % 60.0
    return f"{minutes:02d}:{seconds:04.1f} ({seconds_value:.1f}s)"



# =========================================================
# Workspace overlay helpers
# =========================================================
SHOW_WORKSPACE_OVERLAY = True
SHOW_WORKSPACE_GRID = True
SHOW_ROBOT_ON_WORKSPACE = True

# normalized workspace grid lines to draw over the camera image
WORKSPACE_GRID_VALUES = [0.0, 0.25, 0.50, 0.75, 1.0]


def find_lut_path(data_root, robot_name):
    """
    data/map/<robot>/lut.pkl 을 우선 찾고, 없으면 하위 폴더에서 검색한다.
    """
    robot_map_dir = data_root / "map" / robot_name
    direct = robot_map_dir / "lut.pkl"
    if direct.exists():
        return direct

    if robot_map_dir.exists():
        candidates = list(robot_map_dir.rglob("lut.pkl"))
        if candidates:
            return candidates[0]

    candidates = list((data_root / "map").rglob("lut.pkl")) if (data_root / "map").exists() else []
    if candidates:
        # robot_name이 경로에 들어간 것을 우선
        for p in candidates:
            if robot_name.lower() in str(p).lower():
                return p
        return candidates[0]

    return None


def try_make_pixel_to_workspace_mapper():
    """
    PixelToWorkspaceMapper를 최대한 방어적으로 생성한다.
    stream 시각화 목적이므로 실패해도 카메라 overlay 자체는 계속 실행된다.
    """
    if PixelToWorkspaceMapper is None:
        print("[WORKSPACE] PixelToWorkspaceMapper import failed")
        return None

    lut_path = find_lut_path(DATA_ROOT, ROBOT_NAME)
    if lut_path is None:
        print("[WORKSPACE] lut.pkl not found under data/map")
        return None

    print("[WORKSPACE] LUT_PATH:", lut_path)

    trials = [
        ((), {
            "clamp_to_workspace": False,
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
        }),
        ((), {"clamp_to_workspace": False}),
        ((), {}),
    ]

    for args, kwargs in trials:
        try:
            mapper_obj = PixelToWorkspaceMapper(str(lut_path), *args, **kwargs)
            print("[WORKSPACE] mapper ready with kwargs:", kwargs)
            return mapper_obj
        except Exception as e:
            last_err = repr(e)

    print("[WORKSPACE] mapper init failed:", last_err)
    return None


def mapper_convert_one_safe(mapper_obj, u, v):
    try:
        x, y = mapper_obj.convert_one(float(u), float(v))
        x = float(x)
        y = float(y)
        if np.isfinite(x) and np.isfinite(y):
            return x, y
    except Exception:
        return None
    return None


def build_workspace_to_pixel_homography(mapper_obj, image_shape, grid_step_px=24):
    """
    mapper는 pixel -> workspace 변환만 제공하는 경우가 많다.
    그래서 이미지 위의 여러 pixel을 workspace로 변환한 뒤,
    workspace -> pixel homography를 근사 피팅해서
    workspace [0,1] 사각형/격자를 카메라 위에 그린다.

    주의:
    LUT가 완전 비선형이면 근사 오차가 있을 수 있지만,
    workspace boundary 확인용으로는 충분히 유용하다.
    """
    if mapper_obj is None:
        return None

    h, w = image_shape[:2]
    uv_pts = []
    xy_pts = []

    # 이미지 전체를 샘플링하되, LUT hull 밖은 convert 실패할 수 있으므로 성공한 점만 사용.
    for v in range(0, h, int(grid_step_px)):
        for u in range(0, w, int(grid_step_px)):
            xy = mapper_convert_one_safe(mapper_obj, u, v)
            if xy is None:
                continue
            x, y = xy
            # workspace 주변 점만 사용. 바깥으로 조금 튀는 점은 homography 안정화를 위해 허용.
            if -0.25 <= x <= 1.25 and -0.25 <= y <= 1.25:
                xy_pts.append([x, y])
                uv_pts.append([u, v])

    if len(xy_pts) < 8:
        print("[WORKSPACE] not enough mapper samples for homography:", len(xy_pts))
        return None

    xy_pts = np.asarray(xy_pts, dtype=np.float32)
    uv_pts = np.asarray(uv_pts, dtype=np.float32)

    H, mask = cv2.findHomography(xy_pts, uv_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if H is None:
        print("[WORKSPACE] cv2.findHomography failed")
        return None

    inliers = int(mask.sum()) if mask is not None else 0
    print(f"[WORKSPACE] homography ready: samples={len(xy_pts)}, inliers={inliers}")
    return H


def workspace_points_to_pixels(points_xy, H_ws_to_uv):
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H_ws_to_uv).reshape(-1, 2)
    return out


def draw_polyline_safe(image, points_uv, color, thickness=1, closed=False):
    if points_uv is None:
        return image

    pts = np.asarray(points_uv, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        return image

    # 너무 말도 안 되는 투영은 스킵
    if not np.all(np.isfinite(pts)):
        return image

    pts_i = np.round(pts).astype(np.int32)
    cv2.polylines(
        image,
        [pts_i.reshape((-1, 1, 2))],
        isClosed=closed,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    return image


def draw_workspace_overlay_on_camera(frame, H_ws_to_uv):
    """
    카메라 이미지 위에 normalized workspace [0,1] 사각형과 grid를 직접 그린다.
    """
    if H_ws_to_uv is None:
        return frame

    overlay = frame.copy()

    # workspace outer box
    box_xy = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float32)
    box_uv = workspace_points_to_pixels(box_xy, H_ws_to_uv)

    draw_polyline_safe(
        overlay,
        box_uv,
        color=(255, 255, 0),  # cyan/yellow-ish in BGR
        thickness=3,
        closed=True,
    )

    if SHOW_WORKSPACE_GRID:
        # x = const vertical-ish lines
        for x in WORKSPACE_GRID_VALUES:
            line_xy = np.array([[x, 0.0], [x, 1.0]], dtype=np.float32)
            line_uv = workspace_points_to_pixels(line_xy, H_ws_to_uv)
            thickness = 2 if x in (0.0, 1.0) else 1
            draw_polyline_safe(overlay, line_uv, color=(255, 180, 0), thickness=thickness, closed=False)

        # y = const horizontal-ish lines
        for y in WORKSPACE_GRID_VALUES:
            line_xy = np.array([[0.0, y], [1.0, y]], dtype=np.float32)
            line_uv = workspace_points_to_pixels(line_xy, H_ws_to_uv)
            thickness = 2 if y in (0.0, 1.0) else 1
            draw_polyline_safe(overlay, line_uv, color=(255, 180, 0), thickness=thickness, closed=False)

    # corner labels
    labels = [
        ("(0,0)", [0.0, 0.0]),
        ("(1,0)", [1.0, 0.0]),
        ("(1,1)", [1.0, 1.0]),
        ("(0,1)", [0.0, 1.0]),
    ]
    for text, xy in labels:
        uv = workspace_points_to_pixels(np.array([xy], dtype=np.float32), H_ws_to_uv)[0]
        if np.all(np.isfinite(uv)):
            u, v = int(round(uv[0])), int(round(uv[1]))
            cv2.circle(overlay, (u, v), 4, (255, 255, 0), -1, lineType=cv2.LINE_AA)
            cv2.putText(
                overlay,
                text,
                (u + 6, v - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

    # semi-transparent overlay
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)

    return frame


def unpack_robot_state_out(state_out):
    if isinstance(state_out, tuple):
        if len(state_out) >= 2:
            return state_out[0], state_out[1]
        if len(state_out) == 1:
            return state_out[0], None
    return state_out, None


def get_robot_xy_from_state_dict(state):
    if not isinstance(state, dict):
        return None

    x = state.get("x_norm", state.get("x", None))
    y = state.get("y_norm", state.get("y", None))
    if x is None or y is None:
        return None

    try:
        return np.array([float(x), float(y)], dtype=np.float32)
    except Exception:
        return None


def draw_robot_on_camera(frame, H_ws_to_uv, robot_xy):
    if H_ws_to_uv is None or robot_xy is None:
        return frame

    uv = workspace_points_to_pixels(np.asarray([robot_xy], dtype=np.float32), H_ws_to_uv)[0]
    if not np.all(np.isfinite(uv)):
        return frame

    u, v = int(round(uv[0])), int(round(uv[1]))
    cv2.circle(frame, (u, v), 7, (255, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (u, v), 10, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(
        frame,
        "robot",
        (u + 10, v + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return frame



# =========================================================
# First target check
# =========================================================
target_points, raw_target = safe_get_target_points(robot)
print("[target raw]", raw_target)
print("[target points]", target_points)

current_points, raw_current = safe_get_current_points(robot)
print("[current raw]", raw_current)
print("[current points]", current_points)

first_status = safe_get_eval_status(robot)
print("[eval_status]", first_status)


# =========================================================
# Workspace overlay init
# =========================================================
workspace_mapper = try_make_pixel_to_workspace_mapper()
H_WS_TO_UV = None
H_WS_TO_UV_IMAGE_SHAPE = None


# =========================================================
# Stream Loop
# =========================================================
WINDOW_NAME = f"CloudGripper Task1 Stream - {ROBOT_NAME}"

# Task1은 닫힌 도형
TARGET_CLOSED = True
CURRENT_CLOSED = True

# 갱신 주기
TARGET_UPDATE_SEC = 1.0
CURRENT_UPDATE_SEC = 0.25
STATUS_UPDATE_SEC = 0.25

last_target_update = 0.0
last_current_update = 0.0
last_status_update = 0.0

cached_target_points = target_points
cached_current_points = current_points
cached_status = first_status

print("\n[INFO] Streaming started. Press 'q' to quit.\n")

while True:
    image, timestamp = get_image_undistorted(robot)

    if image is None:
        print("[WARN] image is None")
        time.sleep(0.1)
        continue

    frame = image.copy()
    now = time.time()

    # -------------------------
    # workspace overlay homography build
    # -------------------------
    if SHOW_WORKSPACE_OVERLAY:
        if H_WS_TO_UV is None or H_WS_TO_UV_IMAGE_SHAPE != frame.shape[:2]:
            H_WS_TO_UV = build_workspace_to_pixel_homography(workspace_mapper, frame.shape)
            H_WS_TO_UV_IMAGE_SHAPE = frame.shape[:2]

    # -------------------------
    # target update
    # -------------------------
    if cached_target_points is None or (now - last_target_update) >= TARGET_UPDATE_SEC:
        new_target_points, _ = safe_get_target_points(robot)
        if new_target_points is not None:
            cached_target_points = new_target_points
        last_target_update = now

    # -------------------------
    # current object update
    # -------------------------
    if (now - last_current_update) >= CURRENT_UPDATE_SEC:
        new_current_points, _ = safe_get_current_points(robot)
        if new_current_points is not None:
            cached_current_points = new_current_points
        last_current_update = now

    # -------------------------
    # eval status update
    # -------------------------
    if (now - last_status_update) >= STATUS_UPDATE_SEC:
        status = safe_get_eval_status(robot)
        if status is not None:
            cached_status = status
        last_status_update = now

    # -------------------------
    # draw workspace directly on camera
    # -------------------------
    if SHOW_WORKSPACE_OVERLAY:
        frame = draw_workspace_overlay_on_camera(frame, H_WS_TO_UV)

    # -------------------------
    # draw current / target
    # -------------------------
    frame = draw_polygon_overlay(
        frame,
        cached_current_points,
        color=(0, 255, 0),   # green
        label="current",
        closed=CURRENT_CLOSED,
        thickness=2,
        draw_vertices=True,
    )

    frame = draw_polygon_overlay(
        frame,
        cached_target_points,
        color=(0, 0, 255),   # red
        label="target",
        closed=TARGET_CLOSED,
        thickness=2,
        draw_vertices=True,
    )

    if SHOW_WORKSPACE_OVERLAY and SHOW_ROBOT_ON_WORKSPACE:
        try:
            robot_state, _robot_ts = unpack_robot_state_out(robot.get_state())
            robot_xy = get_robot_xy_from_state_dict(robot_state)
            frame = draw_robot_on_camera(frame, H_WS_TO_UV, robot_xy)
        except Exception:
            pass

    # -------------------------
    # info text
    # -------------------------
    y0 = 25
    dy = 25
    text_line = y0

    cv2.putText(
        frame,
        f"Robot: {ROBOT_NAME}",
        (20, text_line),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    text_line += dy

    cv2.putText(
        frame,
        "Green=current, Red=target, Cyan=workspace, Magenta=robot, q=quit",
        (20, text_line),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    text_line += dy

    if timestamp is not None:
        cv2.putText(
            frame,
            f"Image timestamp: {timestamp}",
            (20, text_line),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        text_line += dy

    # -------------------------
    # IoU / remaining time / status 표시
    # -------------------------
    if isinstance(cached_status, dict):
        # status 표시
        status_text = get_status_value(cached_status, "status")
        if status_text is not None:
            cv2.putText(
                frame,
                f"Status: {status_text}",
                (20, text_line),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            text_line += dy

        # IoU 표시
        current_iou = get_status_value(cached_status, "current_iou", "iou")
        if current_iou is not None:
            cv2.putText(
                frame,
                f"Current IoU: {float(current_iou):.4f}",
                (20, text_line),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            text_line += dy

        # [수정] 남은 시간 표시
        # 실제 eval_status 로그에서는 "time_remaining" key로 들어오는 경우가 많음.
        # 기존 "remaining_time_sec"도 fallback으로 같이 지원.
        remaining_time = get_status_value(
            cached_status,
            "time_remaining",
            "remaining_time_sec",
            "remaining_time",
        )

        if remaining_time is not None:
            remaining_text = format_remaining_time(remaining_time)
            if remaining_text is not None:
                cv2.putText(
                    frame,
                    f"Remaining: {remaining_text}",
                    (20, text_line),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                text_line += dy

        # elapsed time 표시
        elapsed_time = get_status_value(
            cached_status,
            "time_elapsed",
            "elapsed_time_sec",
            "elapsed_time",
        )

        if elapsed_time is not None:
            try:
                elapsed_time = float(elapsed_time)
                cv2.putText(
                    frame,
                    f"Elapsed: {elapsed_time:.1f}s",
                    (20, text_line),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                text_line += dy
            except Exception:
                pass

        # success 표시
        success = get_status_value(cached_status, "success")
        if success is not None:
            cv2.putText(
                frame,
                f"Success: {success}",
                (20, text_line),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            text_line += dy

    else:
        cv2.putText(
            frame,
            "Eval status: unavailable",
            (20, text_line),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
