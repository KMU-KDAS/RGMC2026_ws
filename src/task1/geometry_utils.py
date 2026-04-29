import math  # 파이썬 기본 수학 라이브러리 (삼각함수, pi 등 사용)
from typing import List, Optional, Sequence, Tuple  # 타입 힌트를 위한 라이브러리

import numpy as np  # 빠르고 강력한 벡터, 행렬 연산을 위한 라이브러리

# 2D 평면상의 한 점(x, y)을 나타내는 커스텀 타입(Point)
Point = Tuple[float, float]


def wrap_angle(angle: float) -> float:
    # 각도를 항상 -pi ~ pi 사이의 가장 짧은 각도로 정규화
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rotmat(theta: float) -> np.ndarray:
    # 2D 회전 변환 행렬(Rotation Matrix)
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def get_global_corners(pose: Sequence[float], local_corners: Sequence[Point]) -> List[Point]:
    # 로컬 좌표 -> 글로벌(월드) 좌표 변환
    cx, cy, theta = pose
    r = rotmat(theta)
    out = []

    for lx, ly in local_corners:
        gx, gy = np.array([cx, cy], dtype=float) + r @ np.array([lx, ly], dtype=float)
        out.append((float(gx), float(gy)))
    return out


def make_circle_polygon(center: Sequence[float], radius: float, num_points: int = 64) -> List[Point]:
    # 원형을 다각형으로 근사
    cx, cy = center
    return [
        (cx + radius * math.cos(a), cy + radius * math.sin(a))
        for a in np.linspace(0.0, 2.0 * math.pi, num_points, endpoint=False)
    ]


def polygon_centroid(points: Sequence[Point]) -> np.ndarray:
    # 다각형 도심(Centroid)
    pts = np.asarray(points, dtype=float)
    return pts.mean(axis=0)


def point_in_polygon(point: Sequence[float], polygon: Sequence[Point]) -> bool:
    # 다각형 내부 점 판별 (Ray-Casting)
    if polygon is None or len(polygon) == 0:
        return False

    x, y = point
    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-12) + x1):
            inside = not inside
    return inside


def distance_point_to_segment(point: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    # 점과 선분 사이의 최단 거리
    p = np.asarray(point, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))

    if denom < 1e-12:
        return float(np.linalg.norm(p - a))

    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def clip_segment_to_aabb(
    p0: Sequence[float],
    p1: Sequence[float],
    rect_min: Sequence[float],
    rect_max: Sequence[float],
    eps: float = 1e-12,
):
    """
    선분 p0->p1을 축 정렬 사각형(AABB) [rect_min, rect_max] 내부로 clip.
    반환:
        None                : 선분이 박스와 전혀 안 겹침
        ((x0,y0),(x1,y1))   : 박스 내부에 남는 clipped 선분
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    xmin, ymin = float(rect_min[0]), float(rect_min[1])
    xmax, ymax = float(rect_max[0]), float(rect_max[1])

    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]

    t0, t1 = 0.0, 1.0

    # Liang-Barsky clipping
    for p, q in [
        (-dx, p0[0] - xmin),
        ( dx, xmax - p0[0]),
        (-dy, p0[1] - ymin),
        ( dy, ymax - p0[1]),
    ]:
        if abs(p) < eps:
            if q < 0.0:
                return None
            continue

        r = q / p
        if p < 0.0:
            if r > t1:
                return None
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return None
            if r < t1:
                t1 = r

    if t1 < t0:
        return None

    c0 = p0 + t0 * (p1 - p0)
    c1 = p0 + t1 * (p1 - p0)
    return (tuple(c0), tuple(c1))


def _clip_polygon_with_halfplane(points, inside_fn, intersect_fn):
    if not points:
        return []

    output = []
    prev = points[-1]
    prev_inside = inside_fn(prev)

    for curr in points:
        curr_inside = inside_fn(curr)

        if curr_inside:
            if not prev_inside:
                output.append(intersect_fn(prev, curr))
            output.append(curr)
        elif prev_inside:
            output.append(intersect_fn(prev, curr))

        prev = curr
        prev_inside = curr_inside

    return output


def clip_polygon_to_aabb(
    polygon: Sequence[Point],
    rect_min: Sequence[float],
    rect_max: Sequence[float],
) -> List[Point]:
    """
    다각형을 축 정렬 사각형(AABB) 내부로 clip.
    workspace 안에 남아 있는 다각형만 반환.
    """
    if polygon is None or len(polygon) == 0:
        return []

    xmin, ymin = float(rect_min[0]), float(rect_min[1])
    xmax, ymax = float(rect_max[0]), float(rect_max[1])

    pts = [np.asarray(p, dtype=float) for p in polygon]

    def intersect_x(x_edge):
        def _fn(a, b):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            d = b - a
            if abs(d[0]) < 1e-12:
                return np.array([x_edge, a[1]], dtype=float)
            t = (x_edge - a[0]) / d[0]
            return a + t * d
        return _fn

    def intersect_y(y_edge):
        def _fn(a, b):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            d = b - a
            if abs(d[1]) < 1e-12:
                return np.array([a[0], y_edge], dtype=float)
            t = (y_edge - a[1]) / d[1]
            return a + t * d
        return _fn

    pts = _clip_polygon_with_halfplane(
        pts,
        inside_fn=lambda p: p[0] >= xmin,
        intersect_fn=intersect_x(xmin),
    )
    pts = _clip_polygon_with_halfplane(
        pts,
        inside_fn=lambda p: p[0] <= xmax,
        intersect_fn=intersect_x(xmax),
    )
    pts = _clip_polygon_with_halfplane(
        pts,
        inside_fn=lambda p: p[1] >= ymin,
        intersect_fn=intersect_y(ymin),
    )
    pts = _clip_polygon_with_halfplane(
        pts,
        inside_fn=lambda p: p[1] <= ymax,
        intersect_fn=intersect_y(ymax),
    )

    if not pts:
        return []

    # 연속 중복점 제거
    out: List[Point] = []
    for p in pts:
        tp = (float(p[0]), float(p[1]))
        if not out or np.linalg.norm(np.asarray(tp) - np.asarray(out[-1])) > 1e-9:
            out.append(tp)

    # 첫점/끝점이 사실상 같으면 마지막 제거
    if len(out) >= 2 and np.linalg.norm(np.asarray(out[0]) - np.asarray(out[-1])) <= 1e-9:
        out.pop()

    return out


def distance_point_to_polygon(point: Sequence[float], polygon: Sequence[Point]) -> float:
    # 점과 다각형 사이의 최단 거리
    if polygon is None or len(polygon) == 0:
        return float("inf")

    if len(polygon) == 1:
        return float(np.linalg.norm(np.asarray(point, dtype=float) - np.asarray(polygon[0], dtype=float)))

    if point_in_polygon(point, polygon):
        return 0.0

    return min(
        distance_point_to_segment(point, polygon[i], polygon[(i + 1) % len(polygon)])
        for i in range(len(polygon))
    )


def segment_hits_polygon(a: Sequence[float], b: Sequence[float], polygon: Sequence[Point], margin: float = 0.0) -> bool:
    # 선분과 다각형의 충돌 판정
    if polygon is None or len(polygon) == 0:
        return False

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if point_in_polygon(a, polygon) or point_in_polygon(b, polygon):
        return True

    samples = max(10, int(np.linalg.norm(b - a) / 0.0015))

    for t in np.linspace(0.0, 1.0, samples):
        p = a + t * (b - a)
        if distance_point_to_polygon(p, polygon) <= margin:
            return True

    return False


def path_length(path: Sequence[Sequence[float]]) -> float:
    # 전체 경로 길이 계산
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        total += float(np.linalg.norm(np.asarray(path[i + 1]) - np.asarray(path[i])))
    return total


def pose_error(current_pose: Sequence[float], target_pose: Sequence[float], angle_weight: float = 0.01) -> float:
    # 단순 pose 오차 계산
    dx = target_pose[0] - current_pose[0]
    dy = target_pose[1] - current_pose[1]
    dtheta = wrap_angle(target_pose[2] - current_pose[2])
    return float(np.hypot(dx, dy) + angle_weight * abs(dtheta))


def shape_alignment_error(
    shape_type: str,
    current_pose: Sequence[float],
    target_pose: Sequence[float],
    local_corners: Sequence[Point],
    radius: Optional[float] = None,
) -> float:
    # 모양 정렬 오차 계산
    if shape_type == "WEIGHTED_CIRCLE":
        return float(
            np.linalg.norm(
                np.asarray(current_pose[:2], dtype=float) - np.asarray(target_pose[:2], dtype=float)
            )
        )

    current_points = np.asarray(get_global_corners(current_pose, local_corners), dtype=float)
    target_points = np.asarray(get_global_corners(target_pose, local_corners), dtype=float)

    if len(current_points) == 0:
        return 0.0

    return float(np.mean(np.linalg.norm(current_points - target_points, axis=1)))