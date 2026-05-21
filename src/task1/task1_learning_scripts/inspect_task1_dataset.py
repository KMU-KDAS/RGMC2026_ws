# inspect_task1_dataset.py
# -*- coding: utf-8 -*-
"""
Task1 pushing dataset inspector.

사용 목적
- physics_dynamics.jsonl / real_push_dynamics.jsonl이 학습 가능한 형태인지 빠르게 확인
- 실제 로봇 데이터에서 physics_next_pose와 real_next_pose 차이 확인
- shape별 개수, 누락 필드, residual 통계 출력

권장 위치
- RGMC 프로젝트의 src/task1/ 폴더에 넣고 실행
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


SHAPE_TYPES = ("T_BASE", "WEIGHTED_SQUARE", "WEIGHTED_CIRCLE")


def wrap_angle(rad: float) -> float:
    return float((rad + math.pi) % (2.0 * math.pi) - math.pi)


def load_jsonl(path: str | Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj["_source_file"] = str(path)
                obj["_line_idx"] = line_idx
                rows.append(obj)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode failed: {path}:{line_idx} | {e}")
    return rows


def as_pose(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape[0] < 3:
        return None
    if not np.all(np.isfinite(arr[:3])):
        return None
    return arr[:3]


def get_action(obj: dict) -> Optional[dict]:
    action = obj.get("selected_action")
    if isinstance(action, dict):
        return action
    action = obj.get("action")
    if isinstance(action, dict):
        return action
    return None


def classify_row(obj: dict) -> str:
    if obj.get("data_type") == "physics":
        return "physics"
    if obj.get("valid_for_dynamics") is True or obj.get("real_next_pose") is not None:
        return "real"
    return str(obj.get("data_type", "unknown"))


def summarize_rows(rows: Iterable[dict], workspace_scale_m: float) -> None:
    rows = list(rows)
    print(f"[DATA] total rows: {len(rows)}")

    by_source = Counter(classify_row(r) for r in rows)
    by_shape = Counter(str(r.get("shape_type", "UNKNOWN")) for r in rows)
    print("[DATA] by source:", dict(by_source))
    print("[DATA] by shape :", dict(by_shape))

    missing = Counter()
    valid_real = []
    valid_physics = []

    for obj in rows:
        source = classify_row(obj)
        shape = obj.get("shape_type")
        current = as_pose(obj.get("current_pose"))
        physics = as_pose(obj.get("physics_next_pose"))
        real = as_pose(obj.get("real_next_pose"))
        target = as_pose(obj.get("target_next_pose"))
        action = get_action(obj)

        if shape not in SHAPE_TYPES:
            missing["shape_type"] += 1
        if current is None:
            missing["current_pose"] += 1
        if physics is None:
            missing["physics_next_pose"] += 1
        if action is None:
            missing["action"] += 1

        if source == "real":
            if real is None:
                missing["real_next_pose"] += 1
            elif current is not None and physics is not None:
                valid_real.append((shape, current, physics, real))
        elif source == "physics":
            if target is None and physics is None:
                missing["target_next_pose_or_physics_next_pose"] += 1
            elif current is not None and physics is not None:
                valid_physics.append((shape, current, physics, target if target is not None else physics))

    print("[DATA] missing fields:", dict(missing))
    print(f"[DATA] valid real dynamics rows   : {len(valid_real)}")
    print(f"[DATA] valid physics dynamics rows: {len(valid_physics)}")

    if valid_real:
        center_err_mm = []
        theta_err_deg = []
        residual_xy_mm = []
        residual_theta_deg = []

        by_shape_stats = defaultdict(list)
        for shape, current, physics, real in valid_real:
            dxy = np.linalg.norm(real[:2] - physics[:2]) * workspace_scale_m * 1000.0
            dth = abs(wrap_angle(float(real[2] - physics[2]))) * 180.0 / math.pi
            center_err_mm.append(dxy)
            theta_err_deg.append(dth)
            residual_xy_mm.append((real[:2] - physics[:2]) * workspace_scale_m * 1000.0)
            residual_theta_deg.append(wrap_angle(float(real[2] - physics[2])) * 180.0 / math.pi)
            by_shape_stats[shape].append((dxy, dth))

        center_err_mm = np.asarray(center_err_mm)
        theta_err_deg = np.asarray(theta_err_deg)
        residual_xy_mm = np.asarray(residual_xy_mm)
        residual_theta_deg = np.asarray(residual_theta_deg)

        print("\n[REAL vs PHYSICS baseline]")
        print(f"center residual mean / median / max [mm]: "
              f"{center_err_mm.mean():.3f} / {np.median(center_err_mm):.3f} / {center_err_mm.max():.3f}")
        print(f"theta residual  mean / median / max [deg]: "
              f"{theta_err_deg.mean():.3f} / {np.median(theta_err_deg):.3f} / {theta_err_deg.max():.3f}")
        print(f"residual dx,dy mean [mm]: {residual_xy_mm.mean(axis=0)}")
        print(f"residual theta mean [deg]: {residual_theta_deg.mean():.3f}")

        print("\n[REAL baseline by shape]")
        for shape, values in sorted(by_shape_stats.items()):
            arr = np.asarray(values, dtype=np.float32)
            print(f"  {shape:16s} | n={len(arr):4d} | center_mm={arr[:,0].mean():7.3f} | theta_deg={arr[:,1].mean():7.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=str, default=None, help="real_push_dynamics.jsonl")
    parser.add_argument("--physics", type=str, default=None, help="physics_dynamics.jsonl")
    parser.add_argument("--workspace-scale-m", type=float, default=0.15)
    args = parser.parse_args()

    rows = []
    if args.real:
        rows.extend(load_jsonl(args.real))
    if args.physics:
        rows.extend(load_jsonl(args.physics))

    if not rows:
        raise ValueError("No input rows. Pass --real and/or --physics.")

    summarize_rows(rows, workspace_scale_m=args.workspace_scale_m)


if __name__ == "__main__":
    main()
