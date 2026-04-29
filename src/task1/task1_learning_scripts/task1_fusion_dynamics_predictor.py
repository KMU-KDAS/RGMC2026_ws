# task1_fusion_dynamics_predictor.py
# -*- coding: utf-8 -*-
"""
Lightweight inference wrapper for task1_fusion_dynamics.pt.

목적
- train_task1_fusion_dynamics.py로 학습한 모델을 불러와서
  후보 action의 corrected_next_pose를 예측한다.

권장 위치
- RGMC 프로젝트의 src/task1/ 폴더

VSCode 사용법
- 단독 Run 시, DIRECT_RUN_CONFIG의 checkpoint 경로가 존재하면 간단한 smoke test 실행
- 보통은 brain.py / notebook에서 import해서 사용

사용 예시:
    from task1_fusion_dynamics_predictor import Task1FusionDynamicsPredictor

    predictor = Task1FusionDynamicsPredictor(
        r"src/task1/models/task1_fusion_dynamics.pt"
    )

    corrected = predictor.predict_corrected_next_pose(
        shape_type="T_BASE",
        current_pose=[0.4, 0.4, 0.1],
        target_pose=[0.6, 0.6, 0.0],
        action=candidate,
        physics_next_pose=[0.42, 0.41, 0.15],
    )
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


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

try:
    from task1 import config
except Exception:
    import config  # type: ignore


# ============================================================
# VSCode Run-button config
# ============================================================
USE_DIRECT_RUN = True

DIRECT_RUN_CONFIG = dict(
    checkpoint=r"src/task1/models/task1_fusion_dynamics.pt",
    device="cuda" if torch.cuda.is_available() else "cpu",
)


SHAPE_TYPES = ("T_BASE", "WEIGHTED_SQUARE", "WEIGHTED_CIRCLE")
ACTION_TYPES = ("normal", "spin_pos", "spin_neg")


# ============================================================
# Path helpers
# ============================================================
def resolve_project_path(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


# ============================================================
# Model / feature helpers
# ============================================================
class FusionDynamicsMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        sc = out[:, 2:4]
        sc = sc / sc.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.cat([out[:, 0:2], sc], dim=1)


def encode_pose(pose: Sequence[float]) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    return np.array(
        [pose[0], pose[1], math.sin(float(pose[2])), math.cos(float(pose[2]))],
        dtype=np.float32,
    )


def decode_pose_vec(vec4: Sequence[float]) -> np.ndarray:
    vec4 = np.asarray(vec4, dtype=np.float32).reshape(-1)
    return np.array(
        [vec4[0], vec4[1], math.atan2(float(vec4[2]), float(vec4[3]))],
        dtype=np.float32,
    )


def one_hot(index: int, size: int) -> np.ndarray:
    v = np.zeros((size,), dtype=np.float32)
    if 0 <= index < size:
        v[index] = 1.0
    return v


def shape_one_hot(shape_type: str) -> np.ndarray:
    if shape_type in SHAPE_TYPES:
        return one_hot(SHAPE_TYPES.index(shape_type), len(SHAPE_TYPES))
    return np.zeros((len(SHAPE_TYPES),), dtype=np.float32)


def action_type_one_hot(action_type: str) -> np.ndarray:
    if action_type in ACTION_TYPES:
        return one_hot(ACTION_TYPES.index(action_type), len(ACTION_TYPES))
    return np.zeros((len(ACTION_TYPES),), dtype=np.float32)


def face_norm(shape_type: str, face_idx: int) -> float:
    if shape_type == "T_BASE":
        denom = 7.0
    elif shape_type == "WEIGHTED_SQUARE":
        denom = 3.0
    else:
        denom = max(1.0, float(getattr(config, "DIRECT_CIRCLE_CANDIDATE_ANGLES", 16) - 1))
    return float(np.clip(face_idx / denom, 0.0, 1.0))


def vector2(value, default=(0.0, 0.0)) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float32)
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return np.asarray(default, dtype=np.float32)
    if arr.shape[0] < 2 or not np.all(np.isfinite(arr[:2])):
        return np.asarray(default, dtype=np.float32)
    return arr[:2].astype(np.float32)


def build_feature(shape_type: str, current_pose, target_pose, action: dict, physics_next_pose) -> np.ndarray:
    current_pose = np.asarray(current_pose, dtype=np.float32).reshape(-1)[:3]
    physics_next_pose = np.asarray(physics_next_pose, dtype=np.float32).reshape(-1)[:3]

    if target_pose is None:
        target_pose = np.zeros(3, dtype=np.float32)
    target_pose = np.asarray(target_pose, dtype=np.float32).reshape(-1)[:3]

    start = vector2(action.get("start"))
    end = vector2(action.get("end"))
    delta = end - start
    n_hat = vector2(action.get("n_hat"))
    t_hat = vector2(action.get("t_hat"))

    stroke_len = float(action.get("stroke_len", np.linalg.norm(delta)))
    ratio = float(action.get("ratio", 0.5))
    face_idx = int(action.get("face_idx", 0))
    action_type = str(action.get("action_type", "normal"))

    return np.concatenate(
        [
            shape_one_hot(shape_type),
            encode_pose(current_pose),
            encode_pose(target_pose),
            encode_pose(physics_next_pose),
            start.astype(np.float32),
            end.astype(np.float32),
            delta.astype(np.float32),
            n_hat.astype(np.float32),
            t_hat.astype(np.float32),
            np.array([stroke_len, ratio, face_norm(shape_type, face_idx)], dtype=np.float32),
            action_type_one_hot(action_type),
        ],
        axis=0,
    ).astype(np.float32)


# ============================================================
# Predictor
# ============================================================
class Task1FusionDynamicsPredictor:
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        ckpt = Path(checkpoint_path)
        if not ckpt.is_absolute():
            ckpt = Path(resolve_project_path(str(ckpt)))

        if not ckpt.exists():
            raise FileNotFoundError(f"fusion dynamics checkpoint not found: {ckpt}")

        self.checkpoint_path = str(ckpt)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        payload = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        self.input_dim = int(payload["input_dim"])
        self.hidden_dim = int(payload.get("hidden_dim", 128))
        self.workspace_scale_m = float(payload.get("workspace_scale_m", 0.15))
        self.payload = payload

        self.model = FusionDynamicsMLP(self.input_dim, self.hidden_dim).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict_corrected_next_pose(
        self,
        shape_type: str,
        current_pose,
        action: dict,
        physics_next_pose,
        target_pose=None,
    ) -> np.ndarray:
        x = build_feature(shape_type, current_pose, target_pose, action, physics_next_pose)
        if x.shape[0] != self.input_dim:
            raise ValueError(f"feature dim mismatch: got {x.shape[0]}, expected {self.input_dim}")

        xt = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        out = self.model(xt).squeeze(0).detach().cpu().numpy()
        return decode_pose_vec(out)

    @torch.no_grad()
    def predict_batch(
        self,
        shape_type: str,
        current_pose,
        actions: List[dict],
        physics_next_poses: List[Sequence[float]],
        target_pose=None,
    ) -> np.ndarray:
        if len(actions) != len(physics_next_poses):
            raise ValueError("actions and physics_next_poses length mismatch")

        feats = [
            build_feature(shape_type, current_pose, target_pose, a, p)
            for a, p in zip(actions, physics_next_poses)
        ]

        if not feats:
            return np.zeros((0, 3), dtype=np.float32)

        x = np.stack(feats, axis=0).astype(np.float32)
        if x.shape[1] != self.input_dim:
            raise ValueError(f"feature dim mismatch: got {x.shape[1]}, expected {self.input_dim}")

        xt = torch.tensor(x, dtype=torch.float32, device=self.device)
        out = self.model(xt).detach().cpu().numpy()
        return np.stack([decode_pose_vec(v) for v in out], axis=0).astype(np.float32)


# ============================================================
# VSCode smoke test
# ============================================================
def main() -> None:
    if not USE_DIRECT_RUN:
        print("This file is mainly an importable predictor wrapper.")
        return

    ckpt_path = resolve_project_path(str(DIRECT_RUN_CONFIG["checkpoint"]))
    device = str(DIRECT_RUN_CONFIG.get("device", "cpu"))

    print("===== VSCode DIRECT RUN CONFIG =====")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"checkpoint: {ckpt_path}")
    print(f"device    : {device}")
    print("====================================")

    if not Path(ckpt_path).exists():
        print("[WARN] checkpoint does not exist yet.")
        print("먼저 train_task1_fusion_dynamics.py를 실행해서 모델을 만들어야 함.")
        return

    predictor = Task1FusionDynamicsPredictor(ckpt_path, device=device)

    dummy_action = {
        "face_idx": 0,
        "ratio": 0.5,
        "action_type": "normal",
        "start": [0.40, 0.40],
        "end": [0.45, 0.40],
        "n_hat": [-1.0, 0.0],
        "t_hat": [0.0, 1.0],
        "stroke_len": 0.05,
    }

    corrected = predictor.predict_corrected_next_pose(
        shape_type="T_BASE",
        current_pose=[0.40, 0.40, 0.10],
        target_pose=[0.60, 0.60, 0.00],
        action=dummy_action,
        physics_next_pose=[0.42, 0.41, 0.15],
    )

    print("[OK] model loaded.")
    print("corrected_next_pose:", corrected)


if __name__ == "__main__":
    main()
