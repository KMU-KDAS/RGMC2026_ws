import argparse
import json
import math
import os
import pickle

import cv2
import numpy as np


def load_lut(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_image(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return image


def safe_workspace_to_pixel(lut, x: float, y: float):
    try:
        u, v = lut.workspace_to_pixel(float(x), float(y))
        u = float(u)
        v = float(v)
        if math.isnan(u) or math.isnan(v):
            return None
        return (u, v)
    except Exception:
        return None


def draw_grid_overlay(image_bgr, lut, grid_size=11, x_min=0.05, x_max=0.95, y_min=0.05, y_max=0.95, draw_labels=True, point_radius=3, line_thickness=1):
    overlay = image_bgr.copy()
    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)

    grid_points = {}
    valid_count = 0
    invalid_count = 0

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            uv = safe_workspace_to_pixel(lut, float(x), float(y))
            grid_points[(ix, iy)] = uv
            if uv is None:
                invalid_count += 1
            else:
                valid_count += 1

    for iy, y in enumerate(ys):
        prev = None
        for ix, x in enumerate(xs):
            uv = grid_points[(ix, iy)]
            if uv is None:
                prev = None
                continue
            pt = (int(round(uv[0])), int(round(uv[1])))
            if prev is not None:
                cv2.line(overlay, prev, pt, (0, 255, 255), line_thickness, cv2.LINE_AA)
            prev = pt

    for ix, x in enumerate(xs):
        prev = None
        for iy, y in enumerate(ys):
            uv = grid_points[(ix, iy)]
            if uv is None:
                prev = None
                continue
            pt = (int(round(uv[0])), int(round(uv[1])))
            if prev is not None:
                cv2.line(overlay, prev, pt, (255, 255, 0), line_thickness, cv2.LINE_AA)
            prev = pt

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            uv = grid_points[(ix, iy)]
            if uv is None:
                continue
            u, v = uv
            pt = (int(round(u)), int(round(v)))
            cv2.circle(overlay, pt, point_radius, (0, 0, 255), -1, cv2.LINE_AA)
            if draw_labels:
                label = f"({x:.2f},{y:.2f})"
                cv2.putText(overlay, label, (pt[0] + 4, pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    info = {
        "grid_size": int(grid_size),
        "valid_projected_points": int(valid_count),
        "invalid_projected_points": int(invalid_count),
        "workspace_bounds": {
            "x_min": float(x_min),
            "x_max": float(x_max),
            "y_min": float(y_min),
            "y_max": float(y_max),
        },
    }
    return overlay, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lut_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--grid_size", type=int, default=11)
    parser.add_argument("--x_min", type=float, default=0.05)
    parser.add_argument("--x_max", type=float, default=0.95)
    parser.add_argument("--y_min", type=float, default=0.05)
    parser.add_argument("--y_max", type=float, default=0.95)
    parser.add_argument("--no_labels", action="store_true")
    args = parser.parse_args()

    lut = load_lut(args.lut_path)
    image = load_image(args.image_path)
    overlay, info = draw_grid_overlay(
        image, lut, grid_size=args.grid_size, x_min=args.x_min, x_max=args.x_max, y_min=args.y_min, y_max=args.y_max, draw_labels=not args.no_labels
    )
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    cv2.imwrite(args.out_path, overlay)

    info_path = os.path.splitext(args.out_path)[0] + "_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"saved overlay to: {args.out_path}")
    print(f"saved overlay info to: {info_path}")


if __name__ == "__main__":
    main()
