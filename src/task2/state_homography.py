import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def load_calibration_pairs_from_csv(
    csv_path: str | Path,
    require_ok: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    calibration.csv에서 (xy, uv) 대응쌍 로드
    기대 컬럼: x, y, u, v
    선택 컬럼: ok
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Calibration CSV not found: {csv_path}")

    xy_list = []
    uv_list = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {"x", "y", "u", "v"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise RuntimeError(
                f"CSV must contain columns {required}, got {reader.fieldnames}"
            )

        for row in reader:
            try:
                if require_ok and "ok" in row:
                    ok = int(str(row["ok"]).strip() or "0")
                    if ok != 1:
                        continue

                x = float(str(row["x"]).strip())
                y = float(str(row["y"]).strip())
                u = float(str(row["u"]).strip())
                v = float(str(row["v"]).strip())

                if not (
                    np.isfinite(x)
                    and np.isfinite(y)
                    and np.isfinite(u)
                    and np.isfinite(v)
                ):
                    continue

                xy_list.append([x, y])
                uv_list.append([u, v])

            except Exception:
                continue

    if len(xy_list) < 4:
        raise RuntimeError(
            f"Need at least 4 valid calibration pairs for homography, got {len(xy_list)}"
        )

    xy = np.asarray(xy_list, dtype=np.float32)
    uv = np.asarray(uv_list, dtype=np.float32)
    return xy, uv


def apply_homography_many(H: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"pts must be (N,2), got {pts.shape}")

    ones = np.ones((len(pts), 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])   # (N,3)

    out_h = (H @ pts_h.T).T          # (N,3)
    w = out_h[:, 2:3]

    out = np.full((len(pts), 2), np.nan, dtype=np.float64)
    valid = np.abs(w[:, 0]) > 1e-12
    out[valid] = out_h[valid, :2] / w[valid]
    return out.astype(np.float32), valid.astype(bool)


def fit_state_homography(
    xy_calib: np.ndarray,
    uv_calib: np.ndarray,
    method: int = 0,
) -> Dict[str, object]:
    """
    xy -> uv homography 적합
    inverse를 사용해서 uv -> xy_ext로 변환
    """
    H_xy_to_uv, inlier_mask = cv2.findHomography(
        srcPoints=np.asarray(xy_calib, dtype=np.float32),
        dstPoints=np.asarray(uv_calib, dtype=np.float32),
        method=method,
    )
    if H_xy_to_uv is None:
        raise RuntimeError("cv2.findHomography failed")

    H_uv_to_xy = np.linalg.inv(H_xy_to_uv)

    uv_pred, uv_valid = apply_homography_many(H_xy_to_uv, xy_calib)
    xy_pred, xy_valid = apply_homography_many(H_uv_to_xy, uv_calib)

    uv_err = (
        np.linalg.norm(uv_pred[uv_valid] - uv_calib[uv_valid], axis=1)
        if np.any(uv_valid) else np.array([])
    )
    xy_err = (
        np.linalg.norm(xy_pred[xy_valid] - xy_calib[xy_valid], axis=1)
        if np.any(xy_valid) else np.array([])
    )

    summary = {
        "n_pairs": int(len(xy_calib)),
        "n_inliers": None if inlier_mask is None else int(np.sum(inlier_mask)),
        "uv_rmse": None if len(uv_err) == 0 else float(np.sqrt(np.mean(uv_err ** 2))),
        "uv_mean_err": None if len(uv_err) == 0 else float(np.mean(uv_err)),
        "uv_max_err": None if len(uv_err) == 0 else float(np.max(uv_err)),
        "xy_rmse": None if len(xy_err) == 0 else float(np.sqrt(np.mean(xy_err ** 2))),
        "xy_mean_err": None if len(xy_err) == 0 else float(np.mean(xy_err)),
        "xy_max_err": None if len(xy_err) == 0 else float(np.max(xy_err)),
    }

    return {
        "H_xy_to_uv": H_xy_to_uv.astype(np.float64),
        "H_uv_to_xy": H_uv_to_xy.astype(np.float64),
        "summary": summary,
    }


def convert_uv_to_xy_ext(
    H_uv_to_xy: np.ndarray,
    uv_points: np.ndarray,
    x_min: float = 0.0,
    x_max: float = 0.94,
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    uv -> 확장 상태 좌표계 xy_ext
    calibration에 담긴 좌표축 정의를 그대로 따른다.
    """
    uv_points = np.asarray(uv_points, dtype=np.float32)
    xy_ext, valid = apply_homography_many(H_uv_to_xy, uv_points)

    inside_workspace_mask = (
        (xy_ext[:, 0] >= x_min) & (xy_ext[:, 0] <= x_max) &
        (xy_ext[:, 1] >= y_min) & (xy_ext[:, 1] <= y_max)
    )

    return xy_ext.astype(np.float32), valid.astype(bool), inside_workspace_mask.astype(bool)


def compare_xy_ext_vs_lut(
    xy_ext: np.ndarray,
    xy_lut: np.ndarray,
    lut_valid_mask: np.ndarray,
) -> Optional[Dict[str, float]]:
    """
    LUT로 검증 가능한 내부 노드에 대해서만
    homography 상태좌표와 LUT 좌표를 비교
    """
    xy_ext = np.asarray(xy_ext, dtype=np.float32)
    xy_lut = np.asarray(xy_lut, dtype=np.float32)
    lut_valid_mask = np.asarray(lut_valid_mask, dtype=bool)

    common_valid = lut_valid_mask.copy()
    common_valid &= np.isfinite(xy_ext[:, 0]) & np.isfinite(xy_ext[:, 1])
    common_valid &= np.isfinite(xy_lut[:, 0]) & np.isfinite(xy_lut[:, 1])

    if common_valid.sum() == 0:
        return None

    err = np.linalg.norm(xy_ext[common_valid] - xy_lut[common_valid], axis=1)

    return {
        "n_common_valid_nodes": int(common_valid.sum()),
        "mean_err": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_err": float(np.max(err)),
    }