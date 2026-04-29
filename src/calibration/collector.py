import csv
import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import CalibrationConfig
from .camera_utils import get_processed_base_image


def generate_grid_points(x_min, x_max, y_min, y_max, grid_size, reverse_xy=False):
    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    points = []
    for row_idx, y in enumerate(ys):
        row_xs = xs if row_idx % 2 == 0 else xs[::-1]
        row = [(float(x), float(y)) for x in row_xs]
        points.extend(row)
    if reverse_xy:
        points = [(y, x) for x, y in points]
    return points


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _aggregate_uv(valid_preds, mode="median"):
    uv = np.array([pred["center_uv"] for pred in valid_preds], dtype=np.float64)
    if mode == "mean":
        center = uv.mean(axis=0)
    else:
        center = np.median(uv, axis=0)
    std = uv.std(axis=0)
    return center, std


def _annotate_image(image, pred, xy, idx, ts):
    vis = image.copy()
    x, y = xy
    text1 = f"idx={idx:04d} xy=({x:.3f},{y:.3f})"
    text2 = f"ts={ts}"
    cv2.putText(vis, text1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, text2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if pred.get("ok") and pred.get("center_uv") is not None:
        u, v = pred["center_uv"]
        cv2.circle(vis, (int(round(u)), int(round(v))), 10, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(vis, f"uv=({u:.2f},{v:.2f})", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(vis, "detection failed", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return vis


def _existing_completed_indices(csv_path):
    done = set()
    if not Path(csv_path).exists():
        return done
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add(int(row["idx"]))
            except Exception:
                pass
    return done


class CalibrationCollector:
    def __init__(self, robot, detector, K, D, undistort_image_fn, config: CalibrationConfig, output_dir: str):
        self.robot = robot
        self.detector = detector
        self.K = K
        self.D = D
        self.undistort_image_fn = undistort_image_fn
        self.config = config
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.masks_dir = self.output_dir / "masks"
        self.overlays_dir = self.output_dir / "overlays"
        self.debug_dir = self.output_dir / "debug"
        self.csv_path = self.output_dir / "calibration.csv"
        self.failed_csv_path = self.output_dir / "calibration_failed.csv"
        self.summary_path = self.output_dir / "summary.json"

    def prepare_dirs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.overlays_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def _init_csv_if_needed(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "idx", "x", "y", "u", "v", "ok", "confidence",
                    "std_u", "std_v", "timestamp", "n_valid",
                    "image_path", "overlay_path", "mask_path"
                ])
        if not self.failed_csv_path.exists():
            with open(self.failed_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["idx", "x", "y", "ok", "reason", "timestamp"])

    def get_points(self):
        if self.config.custom_points is not None:
            return [(float(x), float(y)) for x, y in self.config.custom_points]
        return generate_grid_points(
            self.config.x_min,
            self.config.x_max,
            self.config.y_min,
            self.config.y_max,
            self.config.grid_size,
            reverse_xy=self.config.reverse_xy,
        )

    def _capture_point(self, idx, x, y):
        self.robot.move_xy(float(x), float(y))
        time.sleep(self.config.settle_time_sec)

        frame_records = []
        last_img = None
        last_ts = None
        for frame_idx in range(self.config.frames_per_point):
            img, ts = get_processed_base_image(self.robot, self.K, self.D, self.undistort_image_fn)
            pred = self.detector.predict(img)
            frame_records.append({
                "frame_idx": frame_idx,
                "timestamp": ts,
                "ok": bool(pred.get("ok", False)),
                "center_uv": pred.get("center_uv"),
                "confidence": _safe_float(pred.get("confidence", 0.0)) or 0.0,
                "metadata": pred.get("metadata", {}),
            })
            last_img = img
            last_ts = ts
            last_pred = pred
            if frame_idx < self.config.frames_per_point - 1:
                time.sleep(self.config.inter_frame_sec)

        valid_preds = [
            {
                "center_uv": r["center_uv"],
                "confidence": r["confidence"],
            }
            for r in frame_records
            if r["ok"] and r["center_uv"] is not None
        ]

        debug_payload = {
            "idx": idx,
            "x": x,
            "y": y,
            "frames": frame_records,
            "collected_at": datetime.now().isoformat(),
        }

        if len(valid_preds) >= self.config.min_valid_frames:
            center, std = _aggregate_uv(valid_preds, mode=self.config.aggregate_mode)
            agg_pred = {
                "ok": True,
                "center_uv": (float(center[0]), float(center[1])),
            }
            return {
                "ok": True,
                "x": float(x),
                "y": float(y),
                "u": float(center[0]),
                "v": float(center[1]),
                "std_u": float(std[0]),
                "std_v": float(std[1]),
                "confidence": float(np.median([p["confidence"] for p in valid_preds])),
                "n_valid": int(len(valid_preds)),
                "timestamp": last_ts,
                "image": last_img,
                "mask": last_pred.get("mask"),
                "overlay": _annotate_image(last_img, agg_pred, (x, y), idx, last_ts),
                "debug": debug_payload,
            }
        return {
            "ok": False,
            "x": float(x),
            "y": float(y),
            "reason": f"valid_frames={len(valid_preds)}<{self.config.min_valid_frames}",
            "timestamp": last_ts,
            "image": last_img,
            "overlay": _annotate_image(last_img, {"ok": False, "center_uv": None}, (x, y), idx, last_ts) if last_img is not None else None,
            "debug": debug_payload,
        }

    def run(self):
        self.prepare_dirs()
        self._init_csv_if_needed()
        points = self.get_points()
        completed = _existing_completed_indices(self.csv_path) if self.config.resume else set()

        n_success = 0
        n_fail = 0
        started_at = datetime.now().isoformat()

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f_ok, open(self.failed_csv_path, "a", newline="", encoding="utf-8") as f_fail:
            ok_writer = csv.writer(f_ok)
            fail_writer = csv.writer(f_fail)

            for idx, (x, y) in enumerate(points):
                if idx in completed:
                    print(f"[{idx+1}/{len(points)}] skip idx={idx:04d} because it already exists in CSV")
                    continue

                print(f"[{idx+1}/{len(points)}] move to ({x:.3f}, {y:.3f})")
                result = self._capture_point(idx, x, y)

                img_path = self.images_dir / f"{idx:04d}.png"
                overlay_path = self.overlays_dir / f"{idx:04d}_overlay.png"
                mask_path = self.masks_dir / f"{idx:04d}_mask.png"
                debug_path = self.debug_dir / f"{idx:04d}.json"

                if result.get("image") is not None:
                    cv2.imwrite(str(img_path), result["image"])
                if result.get("overlay") is not None:
                    cv2.imwrite(str(overlay_path), result["overlay"])
                if self.config.save_masks and result.get("mask") is not None:
                    cv2.imwrite(str(mask_path), result["mask"])
                else:
                    mask_path = Path("")

                if self.config.write_debug_json:
                    with open(debug_path, "w", encoding="utf-8") as f:
                        json.dump(result["debug"], f, indent=2)

                if result["ok"]:
                    ok_writer.writerow([
                        idx, result["x"], result["y"], result["u"], result["v"], 1, result["confidence"],
                        result["std_u"], result["std_v"], result["timestamp"], result["n_valid"],
                        str(img_path), str(overlay_path), str(mask_path) if str(mask_path) else ""
                    ])
                    f_ok.flush()
                    n_success += 1
                    print(f"  detected uv=({result['u']:.2f}, {result['v']:.2f}) std=({result['std_u']:.2f}, {result['std_v']:.2f})")
                else:
                    fail_writer.writerow([idx, result["x"], result["y"], 0, result["reason"], result["timestamp"]])
                    f_fail.flush()
                    n_fail += 1
                    print(f"  detection failed: {result['reason']}")

        summary = {
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "output_dir": str(self.output_dir),
            "config": asdict(self.config),
            "n_total_points": len(points),
            "n_success_in_this_run": n_success,
            "n_fail_in_this_run": n_fail,
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary


def collect_calibration(robot, detector, K, D, undistort_image_fn, config: CalibrationConfig, output_dir: str):
    collector = CalibrationCollector(robot, detector, K, D, undistort_image_fn, config, output_dir)
    return collector.run()
