import cv2
import numpy as np
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


"""
Detector for CloudGripper Task1 objects.

Design intent
-------------
1. This file should do DETECTION in pixel space only.
2. Camera calibration / undistortion / image flip / rotation are expected to be
   handled OUTSIDE this file before calling the detector.
3. Pixel -> workspace conversion is also handled OUTSIDE this file through an
   injected mapper (for example calibration.pixel_to_workspace.PixelToWorkspaceMapper).
4. T-shape is normalized to the Task1 internal name "T_BASE".

Expected image input
--------------------

Canonical preprocessing for perception is:
    raw distorted image -> undistort only

"""


# ================================
# 1. Utility functions
# ================================
def load_first_frame(path: str):
    image = cv2.imread(path)
    if image is not None:
        return image

    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None
    return frame



def apply_morphology(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask



def order_points(pts: Sequence[Sequence[float]]) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)



def contour_center_from_moments(contour: np.ndarray):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)



def rectangle_theta_deg_from_box(ordered_box: np.ndarray) -> float:
    tl, tr = ordered_box[0], ordered_box[1]
    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]
    return float(np.degrees(np.arctan2(dy, dx)))



def find_bottom_edge_points(contour: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = contour.reshape(-1, 2)
    max_y = np.max(pts[:, 1])

    bottom_band = pts[pts[:, 1] >= max_y - 5]

    bl = bottom_band[np.argmin(bottom_band[:, 0])]
    br = bottom_band[np.argmax(bottom_band[:, 0])]

    return np.array(bl, dtype=np.float32), np.array(br, dtype=np.float32)



def reconstruct_square_top(bl: Sequence[float], br: Sequence[float]):
    bl = np.array(bl, dtype=np.float32)
    br = np.array(br, dtype=np.float32)

    v = br - bl
    side_len = np.linalg.norm(v)

    if side_len < 1e-6:
        return None, None

    n = np.array([-v[1], v[0]], dtype=np.float32) / side_len

    if n[1] > 0:
        n = -n

    tl = bl + n * side_len
    tr = br + n * side_len

    return tl, tr



def compute_square_center_from_bottom(bl: Sequence[float], br: Sequence[float]):
    bl = np.array(bl, dtype=np.float32)
    br = np.array(br, dtype=np.float32)

    v = br - bl
    side_len = np.linalg.norm(v)

    if side_len < 1e-6:
        return None

    mid = (bl + br) / 2.0
    n = np.array([-v[1], v[0]], dtype=np.float32) / side_len

    if n[1] > 0:
        n = -n

    center = mid + n * (side_len / 2.0)
    return center



def enclosing_circle_fill_ratio(contour: np.ndarray):
    area = cv2.contourArea(contour)
    (x, y), radius = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * (radius ** 2)

    if circle_area < 1e-6:
        return 0.0, (x, y), radius

    fill_ratio = area / circle_area
    return fill_ratio, (x, y), radius



def contour_solidity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area < 1e-6:
        return 0.0
    return float(area / hull_area)


def get_object_aligned_axes(contour: np.ndarray):
    """
    Build a stable local frame for center estimation.
    x-axis: width direction of the visible object face
    y-axis: downward direction in the image
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)

    edges = []
    for i in range(4):
        p1 = box[i]
        p2 = box[(i + 1) % 4]
        v = p2 - p1
        L = np.linalg.norm(v)
        if L > 1e-6:
            edges.append((v / L, float(L)))

    if not edges:
        return np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32), np.array(rect[0], dtype=np.float32)

    lengths = np.array([e[1] for e in edges], dtype=np.float32)
    max_len = float(np.max(lengths))
    min_len = float(np.min(lengths))

    if min_len > 1e-6 and (max_len / min_len) < 1.15:
        x_axis = max(edges, key=lambda e: abs(e[0][0]))[0].copy()
    else:
        x_axis = max(edges, key=lambda e: e[1])[0].copy()

    if x_axis[0] < 0:
        x_axis = -x_axis

    y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float32)
    if y_axis[1] < 0:
        y_axis = -y_axis
        x_axis = -x_axis

    origin = np.array(rect[0], dtype=np.float32)
    return x_axis, y_axis, origin


def project_points_to_local_frame(
    points: np.ndarray,
    origin: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
):
    rel = points.astype(np.float32) - origin.reshape(1, 2)
    x_local = rel @ x_axis
    y_local = rel @ y_axis
    return x_local, y_local


def robust_rectangle_face_center_from_contour(contour: np.ndarray):
    """
    Estimate the center of the front/working face directly in a local object-aligned frame.
    This is more robust than using the raw minAreaRect center when the top face is also visible.
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 8:
        return None

    x_axis, y_axis, origin = get_object_aligned_axes(contour)
    x_local, y_local = project_points_to_local_frame(pts, origin, x_axis, y_axis)

    width_all = float(np.max(x_local) - np.min(x_local))
    height_all = float(np.max(y_local) - np.min(y_local))
    if width_all < 1e-6 or height_all < 1e-6:
        return None

    y_q35 = np.percentile(y_local, 35)
    y_q90 = np.percentile(y_local, 90)
    side_mask = (y_local >= y_q35) & (y_local <= y_q90)
    if np.count_nonzero(side_mask) < 8:
        side_mask = y_local >= np.percentile(y_local, 25)

    x_side = x_local[side_mask]
    x_left = float(np.percentile(x_side, 5))
    x_right = float(np.percentile(x_side, 95))

    if (x_right - x_left) < 0.25 * width_all:
        x_left = float(np.percentile(x_local, 5))
        x_right = float(np.percentile(x_local, 95))

    face_width = x_right - x_left
    x_center = 0.5 * (x_left + x_right)

    center_band_half = max(6.0, 0.22 * max(face_width, 1.0))
    center_mask = np.abs(x_local - x_center) <= center_band_half
    if np.count_nonzero(center_mask) < 6:
        center_mask = np.abs(x_local - x_center) <= max(8.0, 0.30 * width_all)

    y_center_band = y_local[center_mask]
    if len(y_center_band) >= 4:
        y_top = float(np.percentile(y_center_band, 8))
    else:
        y_top = float(np.percentile(y_local, 10))

    y_bottom = float(np.percentile(y_local, 97))

    if (y_bottom - y_top) < 0.25 * height_all:
        y_top = float(np.percentile(y_local, 20))
        y_bottom = float(np.percentile(y_local, 95))

    face_height = y_bottom - y_top
    y_center = 0.5 * (y_top + y_bottom)

    center_local = np.array([x_center, y_center], dtype=np.float32)
    center_img = origin + x_axis * center_local[0] + y_axis * center_local[1]

    corners_local = np.array([
        [x_left, y_top],
        [x_right, y_top],
        [x_right, y_bottom],
        [x_left, y_bottom],
    ], dtype=np.float32)
    corners_img = origin.reshape(1, 2) + corners_local[:, [0]] * x_axis.reshape(1, 2) + corners_local[:, [1]] * y_axis.reshape(1, 2)

    theta_deg = float(np.degrees(np.arctan2(x_axis[1], x_axis[0])))

    return {
        "center_px": (int(round(center_img[0])), int(round(center_img[1]))),
        "center_px_float": (float(center_img[0]), float(center_img[1])),
        "corners_px": [(int(round(p[0])), int(round(p[1]))) for p in corners_img],
        "theta_deg": theta_deg,
        "face_width_px": float(face_width),
        "face_height_px": float(face_height),
        "x_axis": x_axis,
        "y_axis": y_axis,
        "origin": origin,
    }



# ================================
# 2. Result builders (PIXEL SPACE ONLY)
# ================================
def make_circle_result(role: str, color_name: str, center_px: Tuple[int, int], radius_px: float, contour_area: float):
    return {
        "detected": True,
        "shape": "circle",
        "role": role,
        "color": color_name,
        "center_px": tuple(center_px),
        "radius_px": float(radius_px),
        "area_px": float(contour_area),
    }



def make_rectangle_result(
    role: str,
    color_name: str,
    corners_px: Sequence[Tuple[int, int]],
    center_px: Tuple[int, int],
    theta_deg: float,
    contour_area: float,
):
    return {
        "detected": True,
        "shape": "rectangle",
        "role": role,
        "color": color_name,
        "corners_px": [tuple(p) for p in corners_px],
        "center_px": tuple(center_px),
        "theta_deg": float(theta_deg),
        "area_px": float(contour_area),
    }



def make_t_shape_result(
    role: str,
    color_name: str,
    corners_px: Sequence[Tuple[int, int]],
    center_px: Tuple[int, int],
    theta_deg: float,
    contour_area: float,
    bar_thickness_px: float,
    stem_thickness_px: float,
):
    return {
        "detected": True,
        "shape": "T_BASE",
        "role": role,
        "color": color_name,
        "corners_px": [tuple(p) for p in corners_px],
        "center_px": tuple(center_px),
        "theta_deg": float(theta_deg),
        "area_px": float(contour_area),
        "bar_thickness_px": float(bar_thickness_px),
        "stem_thickness_px": float(stem_thickness_px),
    }


# ================================
# 3. Detector functions
# ================================
def detect_yellow_gripper(image: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([10, 120, 120], dtype=np.uint8)
    upper_yellow = np.array([17, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = apply_morphology(mask, kernel_size=5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug = {
        "mask": mask,
        "contours": contours,
    }

    if len(contours) == 0:
        return None, debug

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)

    if area < 100:
        return None, debug

    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)

    result = make_circle_result(
        role="gripper",
        color_name="yellow",
        center_px=center,
        radius_px=radius,
        contour_area=area,
    )
    result["source"] = "detect_yellow_gripper"
    return result, debug



def detect_red_t_shape_template(
    contour: np.ndarray,
    contour_area: Optional[float] = None,
    bar_thickness_ratio: float = 0.2,
    stem_thickness_ratio: float = 0.2,
):
    if contour_area is None:
        contour_area = cv2.contourArea(contour)

    if contour_area < 300:
        return None

    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None

    mc_x = float(M["m10"] / M["m00"])
    mc_y = float(M["m01"] / M["m00"])

    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle_deg = rect
    angle_rad = math.radians(angle_deg)

    v_stem = np.array([cx - mc_x, cy - mc_y], dtype=np.float32)
    norm = np.linalg.norm(v_stem)

    candidates = [
        (math.cos(angle_rad), math.sin(angle_rad), w, h, angle_rad),
        (math.cos(angle_rad + math.pi / 2), math.sin(angle_rad + math.pi / 2), h, w, angle_rad + math.pi / 2),
        (math.cos(angle_rad + math.pi), math.sin(angle_rad + math.pi), w, h, angle_rad + math.pi),
        (math.cos(angle_rad + 3 * math.pi / 2), math.sin(angle_rad + 3 * math.pi / 2), h, w, angle_rad + 3 * math.pi / 2),
    ]

    if norm > 1e-6:
        v_stem = v_stem / norm
        best_candidate = max(candidates, key=lambda c: v_stem[0] * c[0] + v_stem[1] * c[1])
    else:
        best_candidate = max(candidates, key=lambda c: c[2])

    _, _, L, W, final_theta_rad = best_candidate

    bar_thickness = float(max(2.0, L * bar_thickness_ratio))
    stem_thickness = float(max(2.0, W * stem_thickness_ratio))

    pts_local = np.array([
        [-L / 2, -W / 2],
        [-L / 2 + bar_thickness, -W / 2],
        [-L / 2 + bar_thickness, -stem_thickness / 2],
        [L / 2, -stem_thickness / 2],
        [L / 2, stem_thickness / 2],
        [-L / 2 + bar_thickness, stem_thickness / 2],
        [-L / 2 + bar_thickness, W / 2],
        [-L / 2, W / 2],
    ], dtype=np.float32)

    cos_t = math.cos(final_theta_rad)
    sin_t = math.sin(final_theta_rad)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)

    rotated_pts = np.dot(pts_local, R.T) + np.array([cx, cy], dtype=np.float32)
    corners_px = [(int(round(p[0])), int(round(p[1]))) for p in rotated_pts]
    center_px = (int(round(cx)), int(round(cy)))
    theta_deg = float(math.degrees(final_theta_rad))

    result = make_t_shape_result(
        role="object",
        color_name="red",
        corners_px=corners_px,
        center_px=center_px,
        theta_deg=theta_deg,
        contour_area=contour_area,
        bar_thickness_px=bar_thickness,
        stem_thickness_px=stem_thickness,
    )
    result["source"] = "detect_red_object_T_BASE_template"
    return result



def detect_red_object(
    image: np.ndarray,
    circle_fill_threshold: float = 0.76,
    t_solidity_threshold: float = 0.88,
    bar_thickness_ratio: float = 0.3,
    stem_thickness_ratio: float = 0.3,
):
    """
    Unified red object detector.
    - If fill ratio >= threshold: circle
    - Else if solidity < threshold: T_BASE
    - Else: rectangle
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 80, 80], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)

    lower_red2 = np.array([170, 80, 80], dtype=np.uint8)
    upper_red2 = np.array([179, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = apply_morphology(mask, kernel_size=5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug = {
        "mask": mask,
        "contours": contours,
    }

    if len(contours) == 0:
        return None, debug

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)

    if area < 100:
        return None, debug

    fill_ratio, (x, y), radius = enclosing_circle_fill_ratio(contour)
    solidity = contour_solidity(contour)

    debug["fill_ratio"] = float(fill_ratio)
    debug["solidity"] = float(solidity)

    # 1) circle
    if fill_ratio >= circle_fill_threshold:
        center = (int(x), int(y))
        radius = int(radius)

        result = make_circle_result(
            role="object",
            color_name="red",
            center_px=center,
            radius_px=radius,
            contour_area=area,
        )
        result["source"] = "detect_red_object_circle"
        result["fill_ratio"] = float(fill_ratio)
        debug["classification"] = "circle"
        return result, debug

    # 2) T_BASE
    if solidity < t_solidity_threshold:
        t_result = detect_red_t_shape_template(
            contour,
            contour_area=area,
            bar_thickness_ratio=bar_thickness_ratio,
            stem_thickness_ratio=stem_thickness_ratio,
        )
        if t_result is not None:
            t_result["fill_ratio"] = float(fill_ratio)
            t_result["solidity"] = float(solidity)
            debug["classification"] = "T_BASE"
            return t_result, debug

    # 3) rectangle (center-first front-face estimation)
    rect_info = robust_rectangle_face_center_from_contour(contour)
    if rect_info is None:
        return None, debug

    center_px = rect_info["center_px"]
    theta_deg = rect_info["theta_deg"]
    corners_px = rect_info["corners_px"]

    result = make_rectangle_result(
        role="object",
        color_name="red",
        corners_px=corners_px,
        center_px=center_px,
        theta_deg=theta_deg,
        contour_area=area,
    )
    result["source"] = "detect_red_object_rectangle_center_first"
    result["fill_ratio"] = float(fill_ratio)
    result["solidity"] = float(solidity)
    result["face_width_px"] = float(rect_info["face_width_px"])
    result["face_height_px"] = float(rect_info["face_height_px"])

    debug["classification"] = "rectangle"
    debug["rectangle_method"] = "center_first_local_face_bounds"
    debug["face_width_px"] = float(rect_info["face_width_px"])
    debug["face_height_px"] = float(rect_info["face_height_px"])
    debug["rectangle_center_px_float"] = rect_info["center_px_float"]

    return result, debug


# ================================
# 4. Distance / contact functions
# ================================
def point_to_segment_closest_point(p, a, b):
    p = np.array(p, dtype=np.float32)
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    ab = b - a
    ab_len_sq = np.dot(ab, ab)

    if ab_len_sq < 1e-8:
        closest = a
        dist = np.linalg.norm(p - closest)
        return closest, dist

    t = np.dot(p - a, ab) / ab_len_sq
    t = np.clip(t, 0.0, 1.0)

    closest = a + t * ab
    dist = np.linalg.norm(p - closest)
    return closest, dist



def circle_circle_distance(circle1: Dict[str, Any], circle2: Dict[str, Any]):
    c1 = np.array(circle1["center_px"], dtype=np.float32)
    c2 = np.array(circle2["center_px"], dtype=np.float32)
    r1 = float(circle1["radius_px"])
    r2 = float(circle2["radius_px"])

    v = c2 - c1
    d_centers = np.linalg.norm(v)

    if d_centers < 1e-8:
        return -(r1 + r2), tuple(c1), tuple(c2)

    u = v / d_centers

    p1 = c1 + u * r1
    p2 = c2 - u * r2
    d = d_centers - (r1 + r2)

    return float(d), tuple(p1), tuple(p2)



def circle_polygon_distance(circle: Dict[str, Any], polygon_result: Dict[str, Any]):
    c = np.array(circle["center_px"], dtype=np.float32)
    r = float(circle["radius_px"])

    corners = [np.array(p, dtype=np.float32) for p in polygon_result["corners_px"]]
    edges = []
    n = len(corners)
    for i in range(n):
        edges.append((corners[i], corners[(i + 1) % n]))

    best_dist = float("inf")
    best_obj_pt = None

    for a, b in edges:
        obj_pt, dist_center_to_edge = point_to_segment_closest_point(c, a, b)
        edge_distance = dist_center_to_edge - r

        if edge_distance < best_dist:
            best_dist = edge_distance
            best_obj_pt = obj_pt

    v = best_obj_pt - c
    norm_v = np.linalg.norm(v)

    if norm_v < 1e-8:
        grip_pt = c
    else:
        grip_pt = c + (v / norm_v) * r

    return float(best_dist), tuple(grip_pt), tuple(best_obj_pt)



def compute_gripper_object_distance(gripper_result, object_result, threshold_px: float = 5):
    if gripper_result is None or object_result is None:
        return None

    if gripper_result["shape"] != "circle":
        raise ValueError("gripper_result must be a circle.")

    if object_result["shape"] == "circle":
        d, pg, po = circle_circle_distance(gripper_result, object_result)
    elif object_result["shape"] in ["rectangle", "T_BASE", "t_shape"]:
        d, pg, po = circle_polygon_distance(gripper_result, object_result)
    else:
        raise ValueError(f"Unsupported object shape: {object_result['shape']}")

    return {
        "distance_px": float(d),
        "contact": bool(d <= threshold_px),
        "gripper_point_px": pg,
        "object_point_px": po,
        "contact_point_px": po if d <= threshold_px else None,
        "threshold_px": float(threshold_px),
    }


# ================================
# 5. Adapter: pixel-space detection -> Task1 input
# ================================
def _mapper_convert_one(mapper: Any, u: float, v: float) -> Tuple[float, float]:
    """
    Support either:
      - mapper.convert_one(u, v)
      - mapper(u, v)
    """
    if mapper is None:
        raise ValueError("mapper is required for pixel -> workspace conversion")

    if hasattr(mapper, "convert_one"):
        x, y = mapper.convert_one(float(u), float(v))
        return float(x), float(y)

    if callable(mapper):
        x, y = mapper(float(u), float(v))
        return float(x), float(y)

    raise TypeError("mapper must provide convert_one(u, v) or be callable")



def _mapper_convert_many(mapper: Any, uv_points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return [_mapper_convert_one(mapper, u, v) for u, v in uv_points]

def _get_circle_radius_workspace(
    shape_type: str,
    task1_config: Any = None,
    circle_radius_workspace: Optional[float] = None,
) -> Optional[float]:
    if task1_config is not None and hasattr(task1_config, "get_shape_radius"):
        try:
            value = task1_config.get_shape_radius(shape_type)
            if value is not None:
                return float(value)
        except Exception:
            pass

    if circle_radius_workspace is not None:
        return float(circle_radius_workspace)

    return None


def _wrap_to_pi(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


def adapt_single_result_to_task1(
    res: Dict[str, Any],
    mapper: Any,
    target_pose: Optional[Sequence[float]] = None,
    task1_config: Any = None,
    circle_radius_workspace: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convert one detector result into Task1-friendly format.

    Returns keys:
        shape_type
        current_pose
        shape_info
        radius
        target_pose
        center_workspace
        corners_workspace (polygon shapes only)
        raw_detection

    Notes
    -----
    - For circle objects, radius is NOT taken from radius_px. Use Task1 config or
      an explicit circle_radius_workspace argument.
    - Gripper detections are ignored here because Task1 object planning should use
      the red object, not the yellow gripper.
    """
    if res is None:
        return None

    if target_pose is None:
        target_pose = [0.0, 0.0, 0.0]
    else:
        target_pose = [
            float(target_pose[0]),
            float(target_pose[1]),
            float(target_pose[2]),
        ]

    if res.get("shape") == "circle":
        if res.get("role") == "gripper":
            return None

        cx_ws, cy_ws = _mapper_convert_one(mapper, *res["center_px"])
        shape_type = "WEIGHTED_CIRCLE"
        radius = _get_circle_radius_workspace(
            shape_type=shape_type,
            task1_config=task1_config,
            circle_radius_workspace=circle_radius_workspace,
        )

        return {
            "shape_type": shape_type,
            "current_pose": [cx_ws, cy_ws, 0.0],
            "shape_info": (cx_ws, cy_ws),
            "radius": radius,
            "target_pose": target_pose,
            "center_workspace": (cx_ws, cy_ws),
            "raw_detection": dict(res),
            "debug_radius_px": float(res["radius_px"]),
        }

    if res.get("shape") == "rectangle":
        cx_ws, cy_ws = _mapper_convert_one(mapper, *res["center_px"])
        corners_ws = _mapper_convert_many(mapper, res["corners_px"])
        theta = math.radians(float(res["theta_deg"]))

        return {
            "shape_type": "WEIGHTED_SQUARE",
            "current_pose": [cx_ws, cy_ws, theta],
            "shape_info": corners_ws,
            "radius": None,
            "target_pose": target_pose,
            "center_workspace": (cx_ws, cy_ws),
            "corners_workspace": corners_ws,
            "raw_detection": dict(res),
        }

    if res.get("shape") in ["T_BASE", "t_shape"]:
        cx_ws, cy_ws = _mapper_convert_one(mapper, *res["center_px"])
        corners_ws = _mapper_convert_many(mapper, res["corners_px"])

        # detector T 기준과 planner/config T 기준의 180도 차이 보정
        theta_raw = math.radians(float(res["theta_deg"]))
        theta = _wrap_to_pi(theta_raw + math.pi / 2.0)

        return {
            "shape_type": "T_BASE",
            "current_pose": [cx_ws, cy_ws, theta],
            "shape_info": corners_ws,
            "radius": None,
            "target_pose": target_pose,
            "center_workspace": (cx_ws, cy_ws),
            "corners_workspace": corners_ws,
            "raw_detection": dict(res),
            "debug_theta_deg_raw": float(res["theta_deg"]),
            "debug_theta_deg_adjusted": float(np.degrees(theta)),
        }

    raise ValueError(f"Unsupported detection shape: {res.get('shape')}")



def convert_to_interface_format(
    results: Iterable[Dict[str, Any]],
    mapper: Any,
    target_pose: Optional[Sequence[float]] = None,
    task1_config: Any = None,
    circle_radius_workspace: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Backward-compatible batch adapter.

    Example
    -------
    from calibration.pixel_to_workspace import PixelToWorkspaceMapper
    import config

    mapper = PixelToWorkspaceMapper("lookup_table.pkl")
    task_inputs = convert_to_interface_format(
        results,
        mapper=mapper,
        target_pose=[0.7, 0.7, 0.0],
        task1_config=config,
    )
    """
    output_data = []

    for res in results:
        adapted = adapt_single_result_to_task1(
            res=res,
            mapper=mapper,
            target_pose=target_pose,
            task1_config=task1_config,
            circle_radius_workspace=circle_radius_workspace,
        )
        if adapted is not None:
            output_data.append(adapted)

    return output_data


# ================================
# 6. Visualization
# ================================
def visualize_result(image: np.ndarray, results: Iterable[Optional[Dict[str, Any]]]):
    output = image.copy()

    for res in results:
        if res is None:
            continue

        if res["shape"] == "circle":
            cx, cy = res["center_px"]
            r = int(res["radius_px"])

            if res["role"] == "gripper":
                draw_color = (0, 255, 255)
            elif res["color"] == "red":
                draw_color = (0, 255, 0)
            else:
                draw_color = (255, 255, 255)

            cv2.circle(output, (cx, cy), r, draw_color, 2)
            cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)

            cv2.putText(
                output,
                f'{res["role"]} {res["shape"]}',
                (cx + 10, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                draw_color,
                1,
            )
            cv2.putText(
                output,
                f'center={res["center_px"]}',
                (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
            cv2.putText(
                output,
                f'r={int(res["radius_px"])}',
                (cx + 10, cy + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                draw_color,
                1,
            )

        elif res["shape"] in ["rectangle", "T_BASE", "t_shape"]:
            corners = np.array(res["corners_px"], dtype=np.int32)
            cx, cy = res["center_px"]
            theta = res["theta_deg"]

            cv2.drawContours(output, [corners], 0, (255, 255, 0), 2)

            for j, p in enumerate(corners):
                cv2.circle(output, tuple(p), 5, (255, 0, 0), -1)
                cv2.putText(
                    output,
                    f"P{j}",
                    (p[0] + 5, p[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1,
                )

            cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)

            label = "rectangle object" if res["shape"] == "rectangle" else "T_BASE object"
            cv2.putText(
                output,
                label,
                (cx + 10, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )
            cv2.putText(
                output,
                f"center={res['center_px']}",
                (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
            cv2.putText(
                output,
                f"theta={theta:.2f} deg",
                (cx + 10, cy + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

    return output



def draw_contact_info(image: np.ndarray, contact_result: Optional[Dict[str, Any]]):
    if contact_result is None:
        return image

    output = image.copy()

    pg = tuple(map(int, contact_result["gripper_point_px"]))
    po = tuple(map(int, contact_result["object_point_px"]))
    d = contact_result["distance_px"]
    contact = contact_result["contact"]

    cv2.circle(output, pg, 5, (0, 255, 255), -1)
    cv2.circle(output, po, 5, (0, 0, 255), -1)
    cv2.line(output, pg, po, (255, 255, 255), 2)

    text = f"d={d:.2f}px, contact={contact}"
    cv2.putText(output, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return output


# ================================
# 7. Main (simple local debug only)
# ================================
if __name__ == "__main__":
    path = r"C:\Users\wlsdud\Desktop\T_test.png"

    image = load_first_frame(path)
    if image is None:
        print("Failed to load image or video.")
        raise SystemExit

    red_object_result, red_debug = detect_red_object(
        image,
        circle_fill_threshold=0.76,
        t_solidity_threshold=0.88,
        bar_thickness_ratio=0.35,
        stem_thickness_ratio=0.35,
    )
    yellow_gripper_result, yellow_debug = detect_yellow_gripper(image)

    results = []
    if yellow_gripper_result is not None:
        results.append(yellow_gripper_result)
    if red_object_result is not None:
        results.append(red_object_result)

    contact_result = compute_gripper_object_distance(
        yellow_gripper_result,
        red_object_result,
        threshold_px=5,
    )

    print("===== Detection Results (pixel space only) =====")
    if len(results) == 0:
        print("No object detected.")
    else:
        for i, res in enumerate(results):
            print(f"[{i}]")
            for k, v in res.items():
                print(f"  {k}: {v}")

    vis = visualize_result(image, results)
    vis = draw_contact_info(vis, contact_result)

    cv2.imshow("result", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()