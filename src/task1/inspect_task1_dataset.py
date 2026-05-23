# inspect_task1_dataset.py
# -*- coding: utf-8 -*-
"""
Inspect Task1 learning datasets.

목적
- physics dynamics dataset 확인
- real push dynamics dataset 확인
- 실제 데이터에서 physics_next_pose와 real_next_pose 차이 확인
- VSCode Run 버튼으로 바로 실행 가능

권장 위치
- RGMC 프로젝트의 src/task1/ 폴더
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------
# Path setup
# ------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
TASK1_DIR = THIS_FILE.parent
SRC_ROOT = TASK1_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent

for p in (str(TASK1_DIR), str(SRC_ROOT), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ============================================================
# VSCode Run-button config
# ============================================================
USE_DIRECT_RUN = True

DIRECT_RUN_CONFIG = dict(
    # physics 데이터 파일
    physics=r"src/task1/data/task1_physics_dynamics.jsonl",

    # real 데이터 파일
    # "AUTO"로 두면 debug_task1_runtime/task1_learning_dataset 안에서
    # 가장 최신 *_real_push_dynamics.jsonl 파일을 자동으로 찾음.
    real=r"AUTO",

    # 실제 작업공간 크기. 현재 config 기준 0.15m
    workspace_scale_m=0.15,

    # real 파일이 없어도 physics만 검사할지
    allow_missing_real=True,

    # physics 파일이 없어도 real만 검사할지
    allow_missing_physics=True,
)


# ============================================================
# Utility
# ============================================================
def resolve_project_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def find_latest_real_dataset() -> Optional[Path]:
    search_dir = PROJECT_ROOT / "debug_task1_runtime" / "task1_learning_dataset"
    if not search_dir.exists():
        return None

    patterns = [
        "*real_push_dynamics*.jsonl",
        "*real*dynamics*.jsonl",
    ]

    files: List[Path] = []
    for pat in patterns:
        files.extend(search_dir.glob(pat))

    files = sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    return files[0]


def direct_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    if not USE_DIRECT_RUN:
        return parser.parse_args()

    cfg = dict(DIRECT_RUN_CONFIG)

    physics = str(cfg.get("physics", "")).strip()
    real = str(cfg.get("real", "")).strip()

    if physics and physics.upper() != "NONE":
        cfg["physics"] = str(resolve_project_path(physics))
    else:
        cfg["physics"] = ""

    if real.upper() == "AUTO":
        latest = find_latest_real_dataset()
        cfg["real"] = "" if latest is None else str(latest)
    elif real and real.upper() != "NONE":
        cfg["real"] = str(resolve_project_path(real))
    else:
        cfg["real"] = ""

    args = argparse.Namespace(**cfg)

    print("===== VSCode DIRECT RUN CONFIG =====")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")
    print("====================================")

    return args


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] JSON parse failed: {path.name}:{line_idx} | {e}")
    return rows


def as_pose(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.shape[0] < 3:
        return None
    return arr[:3].astype(float)


def wrap_angle(rad: float) -> float:
    return float((rad + math.pi) % (2.0 * math.pi) - math.pi)


def pose_error_mm_deg(a, b, workspace_scale_m: float) -> Optional[Tuple[float, float, float]]:
    """
    a, b: [x, y, theta]
    return:
    - center_err_mm
    - theta_err_deg
    - xy_norm
    """
    pa = as_pose(a)
    pb = as_pose(b)
    if pa is None or pb is None:
        return None

    xy_err_norm = float(np.linalg.norm(pa[:2] - pb[:2]))
    center_err_mm = xy_err_norm * workspace_scale_m * 1000.0
    theta_err_deg = abs(math.degrees(wrap_angle(float(pa[2] - pb[2]))))
    return center_err_mm, theta_err_deg, xy_err_norm


def stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }

    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def print_stats(title: str, values: List[float], unit: str = "") -> None:
    s = stats(values)
    print(f"\n[{title}]")
    print(f"n      : {s['n']}")
    if s["n"] <= 0:
        return
    print(f"mean   : {s['mean']:.4f}{unit}")
    print(f"median : {s['median']:.4f}{unit}")
    print(f"min    : {s['min']:.4f}{unit}")
    print(f"max    : {s['max']:.4f}{unit}")
    print(f"p90    : {s['p90']:.4f}{unit}")
    print(f"p95    : {s['p95']:.4f}{unit}")


def count_by_key(rows: List[dict], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        v = str(r.get(key, "UNKNOWN"))
        out[v] = out.get(v, 0) + 1
    return out


def print_count_table(title: str, counts: Dict[str, int]) -> None:
    print(f"\n[{title}]")
    if not counts:
        print("(empty)")
        return
    for k, v in sorted(counts.items(), key=lambda kv: kv[0]):
        print(f"{k:20s}: {v}")


# ============================================================
# Physics inspection
# ============================================================
def inspect_physics(rows: List[dict], workspace_scale_m: float) -> None:
    print("\n" + "=" * 70)
    print("PHYSICS DATASET")
    print("=" * 70)

    print(f"rows: {len(rows)}")
    print_count_table("shape_type counts", count_by_key(rows, "shape_type"))

    missing_current = 0
    missing_physics_next = 0
    missing_action = 0

    stroke_lens = []
    dx_mm = []
    dy_mm = []
    dtheta_deg = []

    for r in rows:
        cur = as_pose(r.get("current_pose"))
        nxt = as_pose(r.get("physics_next_pose"))
        action = r.get("action")

        if cur is None:
            missing_current += 1
            continue
        if nxt is None:
            missing_physics_next += 1
            continue
        if not isinstance(action, dict):
            missing_action += 1
        else:
            try:
                stroke_lens.append(float(action.get("stroke_len", 0.0)) * workspace_scale_m * 1000.0)
            except Exception:
                pass

        delta = nxt - cur
        dx_mm.append(float(delta[0]) * workspace_scale_m * 1000.0)
        dy_mm.append(float(delta[1]) * workspace_scale_m * 1000.0)
        dtheta_deg.append(abs(math.degrees(wrap_angle(float(delta[2])))))

    print("\n[missing]")
    print(f"missing current_pose      : {missing_current}")
    print(f"missing physics_next_pose : {missing_physics_next}")
    print(f"missing action            : {missing_action}")

    print_stats("action stroke_len", stroke_lens, " mm")
    print_stats("physics delta x", dx_mm, " mm")
    print_stats("physics delta y", dy_mm, " mm")
    print_stats("physics delta theta abs", dtheta_deg, " deg")


# ============================================================
# Real inspection
# ============================================================
def get_real_next_pose(row: dict):
    """
    real dataset schema 차이를 흡수.
    우선순위:
    - real_next_pose
    - observed_pose_after
    - after_pose
    - pose_after
    """
    for key in ("real_next_pose", "observed_pose_after", "after_pose", "pose_after"):
        if key in row:
            p = as_pose(row.get(key))
            if p is not None:
                return p
    return None


def get_physics_next_pose(row: dict):
    for key in ("physics_next_pose", "pred_next_pose", "model_next_pose"):
        if key in row:
            p = as_pose(row.get(key))
            if p is not None:
                return p
    return None


def inspect_real(rows: List[dict], workspace_scale_m: float) -> None:
    print("\n" + "=" * 70)
    print("REAL PUSH DATASET")
    print("=" * 70)

    print(f"rows: {len(rows)}")
    print_count_table("shape_type counts", count_by_key(rows, "shape_type"))

    valid_flags = {}
    for r in rows:
        v = str(r.get("valid_for_dynamics", "MISSING"))
        valid_flags[v] = valid_flags.get(v, 0) + 1
    print_count_table("valid_for_dynamics counts", valid_flags)

    center_errs = []
    theta_errs = []
    residual_dx_mm = []
    residual_dy_mm = []
    stroke_lens = []

    usable = 0
    missing_physics = 0
    missing_real = 0

    for r in rows:
        physics_next = get_physics_next_pose(r)
        real_next = get_real_next_pose(r)

        if physics_next is None:
            missing_physics += 1
            continue
        if real_next is None:
            missing_real += 1
            continue

        err = pose_error_mm_deg(physics_next, real_next, workspace_scale_m)
        if err is None:
            continue

        usable += 1
        center_err_mm, theta_err_deg, _ = err
        center_errs.append(center_err_mm)
        theta_errs.append(theta_err_deg)

        residual = real_next - physics_next
        residual_dx_mm.append(float(residual[0]) * workspace_scale_m * 1000.0)
        residual_dy_mm.append(float(residual[1]) * workspace_scale_m * 1000.0)

        action = r.get("selected_action", r.get("action", None))
        if isinstance(action, dict):
            try:
                stroke_lens.append(float(action.get("stroke_len", 0.0)) * workspace_scale_m * 1000.0)
            except Exception:
                pass

    print("\n[usable]")
    print(f"usable physics_next vs real_next pairs : {usable}")
    print(f"missing physics_next                   : {missing_physics}")
    print(f"missing real_next                      : {missing_real}")

    print_stats("physics baseline center error", center_errs, " mm")
    print_stats("physics baseline theta error abs", theta_errs, " deg")
    print_stats("real - physics residual dx", residual_dx_mm, " mm")
    print_stats("real - physics residual dy", residual_dy_mm, " mm")
    print_stats("real selected stroke_len", stroke_lens, " mm")


# ============================================================
# Argparse / Main
# ============================================================
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect Task1 learning datasets.")
    parser.add_argument("--physics", type=str, default="")
    parser.add_argument("--real", type=str, default="")
    parser.add_argument("--workspace-scale-m", type=float, default=0.15)
    parser.add_argument("--allow-missing-real", action="store_true")
    parser.add_argument("--allow-missing-physics", action="store_true")
    return parser


def main() -> None:
    parser = build_argparser()
    args = direct_args(parser)

    workspace_scale_m = float(args.workspace_scale_m)

    physics_path = Path(str(args.physics)) if str(args.physics).strip() else None
    real_path = Path(str(args.real)) if str(args.real).strip() else None

    physics_rows: List[dict] = []
    real_rows: List[dict] = []

    print("\n===== INPUT FILES =====")

    if physics_path is not None:
        print(f"physics: {physics_path}")
        if physics_path.exists():
            physics_rows = read_jsonl(physics_path)
        else:
            msg = f"physics file not found: {physics_path}"
            if bool(args.allow_missing_physics):
                print(f"[WARN] {msg}")
            else:
                raise FileNotFoundError(msg)
    else:
        print("physics: (none)")

    if real_path is not None:
        print(f"real   : {real_path}")
        if real_path.exists():
            real_rows = read_jsonl(real_path)
        else:
            msg = f"real file not found: {real_path}"
            if bool(args.allow_missing_real):
                print(f"[WARN] {msg}")
            else:
                raise FileNotFoundError(msg)
    else:
        print("real   : (none or AUTO not found)")

    if len(physics_rows) == 0 and len(real_rows) == 0:
        print("\n[ERROR] No input rows.")
        print("확인할 것:")
        print("1. data/task1_physics_dynamics.jsonl 이 실제로 생성됐는지")
        print("2. debug_task1_runtime/task1_learning_dataset 안에 real_push_dynamics jsonl이 있는지")
        print("3. DIRECT_RUN_CONFIG의 physics / real 경로가 맞는지")
        return

    if physics_rows:
        inspect_physics(physics_rows, workspace_scale_m)

    if real_rows:
        inspect_real(real_rows, workspace_scale_m)

    print("\n[DONE] inspect complete.")


if __name__ == "__main__":
    main()