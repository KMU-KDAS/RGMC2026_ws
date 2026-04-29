import argparse
import csv
import json
import math
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator


@dataclass
class CalibrationDataset:
    uv_points: np.ndarray
    xy_points: np.ndarray
    rows: List[Dict]


class BidirectionalLookupTable:
    def __init__(self, fx_uv_to_x, fy_uv_to_y, fu_xy_to_u, fv_xy_to_v, metadata: Dict):
        self.fx_uv_to_x = fx_uv_to_x
        self.fy_uv_to_y = fy_uv_to_y
        self.fu_xy_to_u = fu_xy_to_u
        self.fv_xy_to_v = fv_xy_to_v
        self.metadata = metadata

    def pixel_to_workspace(self, u: float, v: float) -> Tuple[float, float]:
        x = _scalarize_interp_output(self.fx_uv_to_x(u, v))
        y = _scalarize_interp_output(self.fy_uv_to_y(u, v))
        if math.isnan(x) or math.isnan(y):
            raise ValueError(f"(u={u}, v={v}) is outside the calibration hull.")
        return x, y

    def workspace_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        u = _scalarize_interp_output(self.fu_xy_to_u(x, y))
        v = _scalarize_interp_output(self.fv_xy_to_v(x, y))
        if math.isnan(u) or math.isnan(v):
            raise ValueError(f"(x={x}, y={y}) is outside the calibration hull.")
        return u, v


def _scalarize_interp_output(value) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return float("nan")
    return float(arr.reshape(-1)[0])


def load_csv_rows(csv_path: str) -> List[Dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_dataset(csv_path: str, min_confidence: float = 0.0, require_ok: bool = True) -> CalibrationDataset:
    rows = load_csv_rows(csv_path)
    filtered = []
    for row in rows:
        ok = int(str(row.get("ok", "0")).strip() or "0")
        conf = float(str(row.get("confidence", "0")).strip() or "0")
        if require_ok and ok != 1:
            continue
        if conf < min_confidence:
            continue
        filtered.append(row)
    if len(filtered) < 6:
        raise ValueError(f"Need at least 6 valid calibration rows, got {len(filtered)}")

    xy_points = np.array([[float(r["x"]), float(r["y"])] for r in filtered], dtype=np.float64)
    uv_points = np.array([[float(r["u"]), float(r["v"])] for r in filtered], dtype=np.float64)
    return CalibrationDataset(uv_points=uv_points, xy_points=xy_points, rows=filtered)


def create_interpolator(points: np.ndarray, values: np.ndarray, method: str):
    if method == "linear_nd":
        return LinearNDInterpolator(points, values)
    if method == "clough_tocher":
        return CloughTocher2DInterpolator(points, values)
    raise ValueError(f"Unsupported interpolation method: {method}")


def build_bidirectional_lookup(dataset: CalibrationDataset, method: str = "clough_tocher") -> BidirectionalLookupTable:
    uv = dataset.uv_points
    xy = dataset.xy_points

    fx_uv_to_x = create_interpolator(uv, xy[:, 0], method)
    fy_uv_to_y = create_interpolator(uv, xy[:, 1], method)
    fu_xy_to_u = create_interpolator(xy, uv[:, 0], method)
    fv_xy_to_v = create_interpolator(xy, uv[:, 1], method)

    metadata = {
        "method": method,
        "num_points": int(len(dataset.rows)),
        "workspace_bounds": {
            "x_min": float(np.min(xy[:, 0])),
            "x_max": float(np.max(xy[:, 0])),
            "y_min": float(np.min(xy[:, 1])),
            "y_max": float(np.max(xy[:, 1])),
        },
        "pixel_bounds": {
            "u_min": float(np.min(uv[:, 0])),
            "u_max": float(np.max(uv[:, 0])),
            "v_min": float(np.min(uv[:, 1])),
            "v_max": float(np.max(uv[:, 1])),
        },
    }
    return BidirectionalLookupTable(fx_uv_to_x, fy_uv_to_y, fu_xy_to_u, fv_xy_to_v, metadata)


def save_lookup_table(lut: BidirectionalLookupTable, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(lut, f)
    meta_path = os.path.splitext(out_path)[0] + "_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(lut.metadata, f, indent=2)
    return meta_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--method", default="clough_tocher", choices=["linear_nd", "clough_tocher"])
    parser.add_argument("--min_confidence", default=0.0, type=float)
    args = parser.parse_args()

    dataset = parse_dataset(args.csv_path, min_confidence=args.min_confidence, require_ok=True)
    lut = build_bidirectional_lookup(dataset, method=args.method)
    meta_path = save_lookup_table(lut, args.out_path)

    print(f"saved LUT to: {args.out_path}")
    print(f"saved metadata to: {meta_path}")
    print(json.dumps(lut.metadata, indent=2))


if __name__ == "__main__":
    main()
