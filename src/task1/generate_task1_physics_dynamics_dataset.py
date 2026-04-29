# generate_task1_physics_dynamics_dataset.py
# -*- coding: utf-8 -*-
"""
Generate synthetic Task1 pushing dynamics data using the current physics model.

목적
- 실제 로봇 없이 현재 total_.py 물리모델만으로 많은 (s, a, physics_next) 데이터 생성
- 이후 real_push_dynamics.jsonl과 섞어서 fusion dynamics model 학습

권장 위치
- RGMC 프로젝트의 src/task1/ 폴더에 넣고 실행

VSCode 사용법
- 아래 USE_DIRECT_RUN = True 유지
- DIRECT_RUN_CONFIG 값만 수정
- VSCode에서 Run 버튼 클릭
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import List

import numpy as np


# ------------------------------------------------------------
# Path setup: 실행 위치가 달라도 src/task1 import가 되도록 처리
# ------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
TASK1_DIR = THIS_FILE.parent
SRC_ROOT = TASK1_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent

for p in (str(TASK1_DIR), str(SRC_ROOT), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from task1 import config
    from task1.brain import PushingBrain
    from task1.geometry_utils import get_global_corners, shape_alignment_error
except Exception:
    import config  # type: ignore
    from brain import PushingBrain  # type: ignore
    from geometry_utils import get_global_corners, shape_alignment_error  # type: ignore


# ============================================================
# VSCode Run-button config
# ============================================================
USE_DIRECT_RUN = True

DIRECT_RUN_CONFIG = dict(
    # 프로젝트 루트 기준 상대경로로 저장됨
    output=r"src/task1/data/task1_physics_dynamics.jsonl",

    # 만들 물리모델 데이터 개수
    num_records=50000,

    # 랜덤 seed
    seed=42,

    # 생성할 물체 종류
    shape_types="T_BASE,WEIGHTED_SQUARE,WEIGHTED_CIRCLE",

    # pose 하나에서 너무 많은 candidate를 저장하지 않도록 제한
    # 크게 하면 데이터 다양성은 늘지만 생성 시간이 증가
    max_candidates_per_pose=24,

    # 몇 개마다 진행상황 출력할지
    progress_every=1000,
)


SHAPE_TYPES = ("T_BASE", "WEIGHTED_SQUARE", "WEIGHTED_CIRCLE")


# ============================================================
# Utility
# ============================================================
def resolve_project_path(path_str: str) -> str:
    """
    상대경로는 PROJECT_ROOT 기준으로 해석.
    VSCode에서 어느 폴더 기준으로 실행해도 저장 위치가 일정해짐.
    """
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def direct_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    USE_DIRECT_RUN=True이면 argparse 명령어 인자 대신 DIRECT_RUN_CONFIG를 사용.
    USE_DIRECT_RUN=False이면 터미널 명령어 방식 사용.
    """
    if not USE_DIRECT_RUN:
        return parser.parse_args()

    cfg = dict(DIRECT_RUN_CONFIG)
    cfg["output"] = resolve_project_path(str(cfg["output"]))

    args = argparse.Namespace(**cfg)

    print("===== VSCode DIRECT RUN CONFIG =====")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")
    print("====================================")

    return args


def to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.astype(float).tolist()
    if isinstance(x, (np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.int32, np.int64)):
        return int(x)
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


def wrap_angle(rad: float) -> float:
    return float((rad + math.pi) % (2.0 * math.pi) - math.pi)


def pose_to_polygon(shape_type: str, pose: List[float]):
    if shape_type == "WEIGHTED_CIRCLE":
        return None
    return np.asarray(
        get_global_corners(pose, config.get_shape_corners(shape_type)),
        dtype=float,
    )


def pose_inside_workspace(shape_type: str, pose: List[float], margin: float = 0.02) -> bool:
    ws_min = np.asarray(config.WORKSPACE_MIN, dtype=float) + margin
    ws_max = np.asarray(config.WORKSPACE_MAX, dtype=float) - margin

    if shape_type == "WEIGHTED_CIRCLE":
        radius = float(config.get_shape_radius(shape_type) or 0.0)
        p = np.asarray(pose[:2], dtype=float)
        return bool(np.all(p - radius >= ws_min) and np.all(p + radius <= ws_max))

    poly = pose_to_polygon(shape_type, pose)
    if poly is None or len(poly) == 0:
        return False

    return bool(np.all(poly >= ws_min) and np.all(poly <= ws_max))


def sample_pose(shape_type: str, rng: random.Random, max_tries: int = 1000) -> List[float]:
    """
    물체가 workspace 내부에 들어오는 랜덤 pose 생성.
    """
    for _ in range(max_tries):
        pose = [
            rng.uniform(0.18, 0.82),
            rng.uniform(0.18, 0.82),
            rng.uniform(-math.pi, math.pi),
        ]
        if pose_inside_workspace(shape_type, pose, margin=0.015):
            return pose

    raise RuntimeError(f"failed to sample valid pose for {shape_type}")


def choose_random_candidates(candidates: List[dict], n: int, rng: random.Random) -> List[dict]:
    """
    pose 하나에서 너무 많은 후보를 저장하지 않도록 랜덤 subset 선택.
    """
    if n is None or n <= 0:
        return candidates

    if len(candidates) <= n:
        return candidates

    idxs = rng.sample(range(len(candidates)), n)
    return [candidates[i] for i in idxs]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Task1 physics dynamics dataset."
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-records", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shape-types",
        type=str,
        default="T_BASE,WEIGHTED_SQUARE,WEIGHTED_CIRCLE",
    )
    parser.add_argument("--max-candidates-per-pose", type=int, default=24)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser


# ============================================================
# Main generation
# ============================================================
def main() -> None:
    parser = build_argparser()
    args = direct_args(parser)

    rng = random.Random(int(args.seed))
    np.random.seed(int(args.seed))

    shape_types = tuple(s.strip() for s in str(args.shape_types).split(",") if s.strip())
    for s in shape_types:
        if s not in SHAPE_TYPES:
            raise ValueError(f"unknown shape_type: {s}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    brain = PushingBrain()

    num_written = 0
    num_pose_trials = 0
    num_empty_face_candidates = 0
    num_empty_action_candidates = 0

    print("[GEN] start generating physics dynamics dataset")
    print(f"[GEN] output: {out_path}")
    print(f"[GEN] target records: {args.num_records}")
    print(f"[GEN] shape_types: {shape_types}")

    with out_path.open("w", encoding="utf-8") as f:
        while num_written < int(args.num_records):
            num_pose_trials += 1

            shape_type = rng.choice(shape_types)
            current_pose = sample_pose(shape_type, rng)
            target_pose = sample_pose(shape_type, rng)

            local_corners = config.get_shape_corners(shape_type)
            radius = config.get_shape_radius(shape_type)

            if shape_type == "WEIGHTED_CIRCLE":
                shape_info = np.asarray(current_pose[:2], dtype=float)
            else:
                shape_info = np.asarray(
                    get_global_corners(current_pose, local_corners),
                    dtype=float,
                )

            base_stroke = max(
                brain._stroke_len_from_error(shape_type, current_pose, target_pose),
                config.DIRECT_MIN_STROKE_LEN,
            )

            if shape_type == "T_BASE":
                base_stroke *= config.T_BASE_STROKE_SCALE

            face_candidates = brain.generate_face_candidates(
                shape_type,
                shape_info,
                radius,
                base_stroke,
            )

            if not face_candidates:
                num_empty_face_candidates += 1
                continue

            global_err = shape_alignment_error(
                shape_type,
                current_pose,
                target_pose,
                local_corners,
                radius,
            )

            allowed_strokes = brain._allowed_strokes_from_error(shape_type, global_err)

            # generate_candidate_actions는 filtered_faces 형식을 받으므로 감싸준다.
            # 여기서는 물리 데이터 다양성을 위해 필터링 없이 모든 face candidate를 후보 seed로 사용.
            filtered_faces = [{"face_info": c} for c in face_candidates]

            candidates = brain.generate_candidate_actions(
                filtered_faces=filtered_faces,
                shape_type=shape_type,
                shape_info=shape_info,
                allowed_strokes=allowed_strokes,
                base_stroke=base_stroke,
            )

            if not candidates:
                num_empty_action_candidates += 1
                continue

            candidates = choose_random_candidates(
                candidates,
                int(args.max_candidates_per_pose),
                rng,
            )

            belief_name = brain.com_belief[shape_type]

            for cand in candidates:
                if num_written >= int(args.num_records):
                    break

                physics_next = brain._simulate_case(
                    shape_type,
                    current_pose,
                    cand,
                    belief_name,
                )

                record = {
                    "record_type": "dynamics",
                    "data_type": "physics",
                    "shape_type": shape_type,

                    "current_pose": to_jsonable(current_pose),
                    "target_pose": to_jsonable(target_pose),

                    "base_stroke": float(base_stroke),
                    "allowed_strokes": to_jsonable(allowed_strokes),

                    "action": to_jsonable(cand),

                    # 물리모델 예측값
                    "physics_next_pose": to_jsonable(physics_next),

                    # physics 데이터에서는 target도 physics_next로 둔다.
                    # 실제 데이터에서는 target_next_pose가 real_next_pose가 됨.
                    "target_next_pose": to_jsonable(physics_next),

                    "source": "simulate_push_stroke_normalized",
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                num_written += 1

                if int(args.progress_every) > 0 and num_written % int(args.progress_every) == 0:
                    print(
                        f"[GEN] wrote {num_written}/{args.num_records} records | "
                        f"pose_trials={num_pose_trials} | "
                        f"empty_face={num_empty_face_candidates} | "
                        f"empty_action={num_empty_action_candidates}",
                        flush=True,
                    )

    print("[DONE]")
    print(f"output: {out_path}")
    print(f"records: {num_written}")
    print(f"pose_trials: {num_pose_trials}")
    print(f"empty_face_candidates: {num_empty_face_candidates}")
    print(f"empty_action_candidates: {num_empty_action_candidates}")


if __name__ == "__main__":
    main()