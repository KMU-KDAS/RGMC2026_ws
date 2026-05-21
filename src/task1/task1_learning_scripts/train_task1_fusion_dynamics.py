# train_task1_fusion_dynamics.py
# -*- coding: utf-8 -*-
"""
Train a clean fusion dynamics model for Task1 pushing.

핵심 아이디어
- physics 데이터: current_pose + action + physics_next_pose -> physics_next_pose
- real 데이터   : current_pose + action + physics_next_pose -> real_next_pose
- 둘을 섞어서 학습하되, real 데이터가 묻히지 않도록 real_repeat / real_weight를 크게 준다.

출력 모델
- 입력 feature를 받아 corrected_next_pose를 바로 출력
- 외부에서 physics_next + residual을 더하지 않는다.
- 단, 입력 feature에는 physics_next_pose를 포함해서 물리모델 정보를 활용한다.

VSCode 사용법
- USE_DIRECT_RUN = True 유지
- DIRECT_RUN_CONFIG 값만 수정
- VSCode에서 Run 버튼 클릭

권장 위치
- RGMC 프로젝트의 src/task1/ 폴더
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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
    # 경로는 PROJECT_ROOT 기준 상대경로.
    # 네가 원하는 통일 경로:
    physics=r"src/task1/data/task1_physics_dynamics.jsonl",

    # AUTO면 아래 순서로 최신 real 데이터를 자동 검색:
    # 1) src/task1/data/*real_push_dynamics*.jsonl
    # 2) debug_task1_runtime/task1_learning_dataset/*real_push_dynamics*.jsonl
    # 아직 real 데이터 없으면 "" 처리하고 physics만으로 학습 가능.
    real=r"AUTO",

    # 모델 저장 경로
    output=r"src/task1/models/task1_fusion_dynamics.pt",

    epochs=300,
    batch_size=128,
    hidden_dim=128,
    lr=1e-3,
    weight_decay=1e-5,
    train_ratio=0.8,
    seed=42,
    device="cuda" if torch.cuda.is_available() else "cpu",

    max_physics_records=50000,
    max_real_records=None,

    # real 데이터가 적으므로 반복해서 batch에 자주 등장시킴
    real_repeat=20,

    # real 데이터 loss를 더 강하게 반영
    real_weight=10.0,
    physics_weight=1.0,

    corner_loss_weight=0.5,
    workspace_scale_m=0.15,
    log_every=10,
)


SHAPE_TYPES = ("T_BASE", "WEIGHTED_SQUARE", "WEIGHTED_CIRCLE")
ACTION_TYPES = ("normal", "spin_pos", "spin_neg")
MAX_CORNERS = 8


# ============================================================
# Direct-run helpers
# ============================================================
def resolve_project_path(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def find_latest_real_dataset() -> Optional[Path]:
    search_dirs = [
        PROJECT_ROOT / "src" / "task1" / "data",
        PROJECT_ROOT / "debug_task1_runtime" / "task1_learning_dataset",
    ]

    patterns = [
        "*real_push_dynamics*.jsonl",
        "*real*dynamics*.jsonl",
    ]

    files: List[Path] = []
    for d in search_dirs:
        if not d.exists():
            continue
        for pat in patterns:
            files.extend(d.glob(pat))

    files = sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def direct_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    if not USE_DIRECT_RUN:
        return parser.parse_args()

    cfg = dict(DIRECT_RUN_CONFIG)

    physics = str(cfg.get("physics", "")).strip()
    real = str(cfg.get("real", "")).strip()
    output = str(cfg.get("output", "")).strip()

    cfg["physics"] = resolve_project_path(physics) if physics and physics.upper() != "NONE" else None
    cfg["output"] = resolve_project_path(output)

    if real.upper() == "AUTO":
        latest = find_latest_real_dataset()
        cfg["real"] = None if latest is None else str(latest)
    elif real and real.upper() != "NONE":
        cfg["real"] = resolve_project_path(real)
    else:
        cfg["real"] = None

    args = argparse.Namespace(**cfg)

    print("===== VSCode DIRECT RUN CONFIG =====")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")
    print("====================================")

    return args


# ============================================================
# Basic helpers
# ============================================================
def wrap_angle(rad: float) -> float:
    return float((rad + math.pi) % (2.0 * math.pi) - math.pi)


def encode_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    return np.array(
        [pose[0], pose[1], math.sin(float(pose[2])), math.cos(float(pose[2]))],
        dtype=np.float32,
    )


def decode_pose_vec(vec4: np.ndarray) -> np.ndarray:
    vec4 = np.asarray(vec4, dtype=np.float32).reshape(-1)
    return np.array(
        [vec4[0], vec4[1], math.atan2(float(vec4[2]), float(vec4[3]))],
        dtype=np.float32,
    )


def as_pose(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.shape[0] < 3:
        return None
    if not np.all(np.isfinite(arr[:3])):
        return None
    return arr[:3].astype(np.float32)


def get_action(obj: dict) -> Optional[dict]:
    a = obj.get("selected_action")
    if isinstance(a, dict):
        return a
    a = obj.get("action")
    if isinstance(a, dict):
        return a
    return None


def get_physics_next_pose(obj: dict) -> Optional[np.ndarray]:
    for key in ("physics_next_pose", "pred_next_pose", "model_next_pose"):
        p = as_pose(obj.get(key))
        if p is not None:
            return p
    return None


def get_real_next_pose(obj: dict) -> Optional[np.ndarray]:
    for key in ("real_next_pose", "observed_pose_after", "after_pose", "pose_after"):
        p = as_pose(obj.get(key))
        if p is not None:
            return p
    return None


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


def build_feature(
    shape_type: str,
    current_pose: np.ndarray,
    target_pose: Optional[np.ndarray],
    action: dict,
    physics_next_pose: np.ndarray,
) -> np.ndarray:
    if target_pose is None:
        target_pose = np.zeros(3, dtype=np.float32)

    start = vector2(action.get("start"))
    end = vector2(action.get("end"))
    delta = end - start
    n_hat = vector2(action.get("n_hat"))
    t_hat = vector2(action.get("t_hat"))

    stroke_len = float(action.get("stroke_len", np.linalg.norm(delta)))
    ratio = float(action.get("ratio", 0.5))
    face_idx = int(action.get("face_idx", 0))
    action_type = str(action.get("action_type", "normal"))

    feature = np.concatenate(
        [
            shape_one_hot(shape_type),                 # 3
            encode_pose(current_pose),                 # 4
            encode_pose(target_pose),                  # 4
            encode_pose(physics_next_pose),            # 4
            start.astype(np.float32),                  # 2
            end.astype(np.float32),                    # 2
            delta.astype(np.float32),                  # 2
            n_hat.astype(np.float32),                  # 2
            t_hat.astype(np.float32),                  # 2
            np.array([stroke_len, ratio, face_norm(shape_type, face_idx)], dtype=np.float32),  # 3
            action_type_one_hot(action_type),          # 3
        ],
        axis=0,
    )
    return feature.astype(np.float32)


def local_corners_for_shape(shape_type: str) -> Tuple[np.ndarray, np.ndarray]:
    if shape_type == "WEIGHTED_CIRCLE":
        corners = np.zeros((MAX_CORNERS, 2), dtype=np.float32)
        mask = np.zeros((MAX_CORNERS,), dtype=np.float32)
        return corners, mask

    pts = np.asarray(config.get_shape_corners(shape_type), dtype=np.float32)
    corners = np.zeros((MAX_CORNERS, 2), dtype=np.float32)
    mask = np.zeros((MAX_CORNERS,), dtype=np.float32)
    n = min(MAX_CORNERS, len(pts))
    corners[:n] = pts[:n]
    mask[:n] = 1.0
    return corners, mask


# ============================================================
# Dataset
# ============================================================
@dataclass
class FusionSample:
    x: np.ndarray
    y: np.ndarray
    physics_y: np.ndarray
    source: str
    shape_type: str
    weight: float
    corners: np.ndarray
    corner_mask: np.ndarray


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj["_source_file"] = str(path)
                obj["_line_idx"] = line_idx
                yield obj
            except json.JSONDecodeError as e:
                print(f"[WARN] skip bad json: {path}:{line_idx} | {e}")


def parse_record(obj: dict, source_hint: str, real_weight: float, physics_weight: float) -> Optional[FusionSample]:
    shape_type = obj.get("shape_type")
    if shape_type not in SHAPE_TYPES:
        return None

    current_pose = as_pose(obj.get("current_pose"))
    target_pose = as_pose(obj.get("target_pose"))
    physics_next = get_physics_next_pose(obj)
    action = get_action(obj)

    if current_pose is None or physics_next is None or action is None:
        return None

    if source_hint == "real":
        if obj.get("valid_for_dynamics") is False:
            return None
        target_next = get_real_next_pose(obj)
        if target_next is None:
            return None
        source = "real"
        weight = float(real_weight)
    else:
        target_next = as_pose(obj.get("target_next_pose"))
        if target_next is None:
            target_next = physics_next
        source = "physics"
        weight = float(physics_weight)

    x = build_feature(shape_type, current_pose, target_pose, action, physics_next)
    y = encode_pose(target_next)
    physics_y = encode_pose(physics_next)
    corners, corner_mask = local_corners_for_shape(shape_type)

    return FusionSample(
        x=x,
        y=y.astype(np.float32),
        physics_y=physics_y.astype(np.float32),
        source=source,
        shape_type=shape_type,
        weight=weight,
        corners=corners,
        corner_mask=corner_mask,
    )


def load_samples(
    physics_path: Optional[str],
    real_path: Optional[str],
    max_physics_records: Optional[int],
    max_real_records: Optional[int],
    real_repeat: int,
    real_weight: float,
    physics_weight: float,
    seed: int,
) -> List[FusionSample]:
    rng = random.Random(seed)

    physics_samples: List[FusionSample] = []
    real_samples: List[FusionSample] = []

    if physics_path:
        physics_file = Path(physics_path)
        if not physics_file.exists():
            raise FileNotFoundError(f"physics file not found: {physics_file}")

        for obj in iter_jsonl(physics_file):
            s = parse_record(obj, source_hint="physics", real_weight=real_weight, physics_weight=physics_weight)
            if s is not None:
                physics_samples.append(s)
            if max_physics_records is not None and len(physics_samples) >= int(max_physics_records):
                break
    else:
        print("[WARN] physics_path is empty")

    if real_path:
        real_file = Path(real_path)
        if not real_file.exists():
            print(f"[WARN] real file not found, skip real data: {real_file}")
        else:
            for obj in iter_jsonl(real_file):
                s = parse_record(obj, source_hint="real", real_weight=real_weight, physics_weight=physics_weight)
                if s is not None:
                    real_samples.append(s)
                if max_real_records is not None and len(real_samples) >= int(max_real_records):
                    break
    else:
        print("[WARN] real_path is empty. Training with physics data only.")

    if not physics_samples and not real_samples:
        raise RuntimeError("No valid samples loaded.")

    rng.shuffle(physics_samples)
    rng.shuffle(real_samples)

    samples: List[FusionSample] = []
    samples.extend(physics_samples)

    repeat = max(1, int(real_repeat))
    for _ in range(repeat):
        samples.extend(real_samples)

    rng.shuffle(samples)

    print(f"[DATA] physics samples: {len(physics_samples)}")
    print(f"[DATA] real samples   : {len(real_samples)} × repeat {repeat} = {len(real_samples) * repeat}")
    print(f"[DATA] total trainable rows after repeat: {len(samples)}")
    return samples


def stratified_split(samples: List[FusionSample], train_ratio: float, seed: int) -> Tuple[List[FusionSample], List[FusionSample]]:
    rng = random.Random(seed)

    groups: Dict[Tuple[str, str], List[FusionSample]] = {}
    for s in samples:
        groups.setdefault((s.source, s.shape_type), []).append(s)

    train: List[FusionSample] = []
    val: List[FusionSample] = []

    for _, group in groups.items():
        rng.shuffle(group)
        n_train = int(len(group) * float(train_ratio))
        if len(group) >= 2:
            n_train = min(max(1, n_train), len(group) - 1)
        else:
            n_train = len(group)

        train.extend(group[:n_train])
        val.extend(group[n_train:])

    rng.shuffle(train)
    rng.shuffle(val)

    if not val:
        val = train[: max(1, len(train) // 10)]

    return train, val


class FusionDataset(Dataset):
    def __init__(self, samples: List[FusionSample]):
        self.samples = samples
        if not self.samples:
            raise RuntimeError("empty dataset")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "x": torch.tensor(s.x, dtype=torch.float32),
            "y": torch.tensor(s.y, dtype=torch.float32),
            "physics_y": torch.tensor(s.physics_y, dtype=torch.float32),
            "weight": torch.tensor(s.weight, dtype=torch.float32),
            "source_is_real": torch.tensor(1.0 if s.source == "real" else 0.0, dtype=torch.float32),
            "corners": torch.tensor(s.corners, dtype=torch.float32),
            "corner_mask": torch.tensor(s.corner_mask, dtype=torch.float32),
        }


# ============================================================
# Model
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


# ============================================================
# Loss / metrics
# ============================================================
def pose_vec_to_corners(pose_vec: torch.Tensor, corners: torch.Tensor) -> torch.Tensor:
    cx = pose_vec[:, 0].view(-1, 1)
    cy = pose_vec[:, 1].view(-1, 1)
    s = pose_vec[:, 2].view(-1, 1)
    c = pose_vec[:, 3].view(-1, 1)

    lx = corners[:, :, 0]
    ly = corners[:, :, 1]

    gx = cx + c * lx - s * ly
    gy = cy + s * lx + c * ly
    return torch.stack([gx, gy], dim=-1)


def weighted_pose_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    xy_loss = ((pred[:, :2] - target[:, :2]) ** 2).sum(dim=1)
    angle_vec_loss = ((pred[:, 2:4] - target[:, 2:4]) ** 2).sum(dim=1)
    loss = xy_loss + 0.25 * angle_vec_loss
    return (loss * weight).mean()


def corner_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    corners: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    pred_c = pose_vec_to_corners(pred, corners)
    tgt_c = pose_vec_to_corners(target, corners)
    per_corner = ((pred_c - tgt_c) ** 2).sum(dim=-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    per_sample = (per_corner * mask).sum(dim=1) / denom
    return (per_sample * weight).mean()


def pose_metrics_mm_deg(pred: torch.Tensor, target: torch.Tensor, workspace_scale_m: float) -> Dict[str, float]:
    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()

    center = np.linalg.norm(pred_np[:, :2] - tgt_np[:, :2], axis=1) * workspace_scale_m * 1000.0

    pred_th = np.arctan2(pred_np[:, 2], pred_np[:, 3])
    tgt_th = np.arctan2(tgt_np[:, 2], tgt_np[:, 3])
    dth = np.abs(np.array([wrap_angle(a - b) for a, b in zip(pred_th, tgt_th)])) * 180.0 / math.pi

    return {
        "center_mm": float(np.mean(center)),
        "theta_deg": float(np.mean(dth)),
        "center_mm_median": float(np.median(center)),
        "theta_deg_median": float(np.median(dth)),
    }


def run_epoch(model, loader, device, optimizer=None, corner_loss_weight: float = 0.5) -> Dict[str, float]:
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    total_n = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        weight = batch["weight"].to(device)
        corners = batch["corners"].to(device)
        corner_mask = batch["corner_mask"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            pred = model(x)
            loss_pose = weighted_pose_loss(pred, y, weight)
            loss_corner = corner_loss(pred, y, corners, corner_mask, weight)
            loss = loss_pose + float(corner_loss_weight) * loss_corner

            if train:
                loss.backward()
                optimizer.step()

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs

    return {"loss": total_loss / max(1, total_n)}


@torch.no_grad()
def evaluate(model, loader, device, workspace_scale_m: float, corner_loss_weight: float) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_n = 0

    all_pred = []
    all_y = []
    all_physics = []
    all_real_mask = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        physics_y = batch["physics_y"].to(device)
        weight = batch["weight"].to(device)
        corners = batch["corners"].to(device)
        corner_mask = batch["corner_mask"].to(device)
        real_mask = batch["source_is_real"].to(device)

        pred = model(x)
        loss_pose = weighted_pose_loss(pred, y, weight)
        loss_corner = corner_loss(pred, y, corners, corner_mask, weight)
        loss = loss_pose + float(corner_loss_weight) * loss_corner

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs

        all_pred.append(pred.cpu())
        all_y.append(y.cpu())
        all_physics.append(physics_y.cpu())
        all_real_mask.append(real_mask.cpu())

    pred = torch.cat(all_pred, dim=0)
    y = torch.cat(all_y, dim=0)
    physics_y = torch.cat(all_physics, dim=0)
    real_mask = torch.cat(all_real_mask, dim=0) > 0.5

    learned_all = pose_metrics_mm_deg(pred, y, workspace_scale_m)
    physics_all = pose_metrics_mm_deg(physics_y, y, workspace_scale_m)

    out = {
        "loss": total_loss / max(1, total_n),
        "learned_center_mm": learned_all["center_mm"],
        "learned_theta_deg": learned_all["theta_deg"],
        "physics_center_mm": physics_all["center_mm"],
        "physics_theta_deg": physics_all["theta_deg"],
    }

    if torch.any(real_mask):
        learned_real = pose_metrics_mm_deg(pred[real_mask], y[real_mask], workspace_scale_m)
        physics_real = pose_metrics_mm_deg(physics_y[real_mask], y[real_mask], workspace_scale_m)
        out.update(
            {
                "real_learned_center_mm": learned_real["center_mm"],
                "real_learned_theta_deg": learned_real["theta_deg"],
                "real_physics_center_mm": physics_real["center_mm"],
                "real_physics_theta_deg": physics_real["theta_deg"],
            }
        )

    return out


# ============================================================
# Argparse / Main
# ============================================================
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Task1 fusion dynamics model.")
    parser.add_argument("--physics", type=str, default=None)
    parser.add_argument("--real", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--max-physics-records", type=int, default=50000)
    parser.add_argument("--max-real-records", type=int, default=None)
    parser.add_argument("--real-repeat", type=int, default=20)
    parser.add_argument("--real-weight", type=float, default=10.0)
    parser.add_argument("--physics-weight", type=float, default=1.0)
    parser.add_argument("--corner-loss-weight", type=float, default=0.5)
    parser.add_argument("--workspace-scale-m", type=float, default=0.15)
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main() -> None:
    parser = build_argparser()
    args = direct_args(parser)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = load_samples(
        physics_path=args.physics,
        real_path=args.real,
        max_physics_records=args.max_physics_records,
        max_real_records=args.max_real_records,
        real_repeat=args.real_repeat,
        real_weight=args.real_weight,
        physics_weight=args.physics_weight,
        seed=args.seed,
    )

    train_samples, val_samples = stratified_split(samples, train_ratio=args.train_ratio, seed=args.seed)
    print(f"[DATA] split train={len(train_samples)} val={len(val_samples)}")

    train_ds = FusionDataset(train_samples)
    val_ds = FusionDataset(val_samples)

    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    input_dim = int(train_ds[0]["x"].numel())
    device = torch.device(args.device)

    model = FusionDynamicsMLP(input_dim=input_dim, hidden_dim=int(args.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_score = float("inf")
    best_epoch = -1

    print("[RUN]")
    print(f"  device      : {device}")
    print(f"  input_dim   : {input_dim}")
    print(f"  hidden_dim  : {args.hidden_dim}")
    print(f"  output      : {output_path}")

    for epoch in range(1, int(args.epochs) + 1):
        tr = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            corner_loss_weight=float(args.corner_loss_weight),
        )
        va = evaluate(
            model,
            val_loader,
            device,
            workspace_scale_m=float(args.workspace_scale_m),
            corner_loss_weight=float(args.corner_loss_weight),
        )

        # 실제 데이터가 있으면 real validation 기준으로 저장.
        score = va.get("real_learned_center_mm", va["learned_center_mm"]) + 0.1 * va.get(
            "real_learned_theta_deg",
            va["learned_theta_deg"],
        )

        if score < best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dim": int(args.hidden_dim),
                    "shape_types": SHAPE_TYPES,
                    "action_types": ACTION_TYPES,
                    "max_corners": MAX_CORNERS,
                    "workspace_scale_m": float(args.workspace_scale_m),
                    "feature_version": "task1_fusion_dynamics_v1",
                    "train_args": vars(args),
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "val_metrics": va,
                },
                output_path,
            )

        if epoch == 1 or epoch % max(1, int(args.log_every)) == 0 or epoch == int(args.epochs):
            msg = (
                f"[E{epoch:04d}] train_loss={tr['loss']:.6f} val_loss={va['loss']:.6f} | "
                f"learned={va['learned_center_mm']:.2f}mm/{va['learned_theta_deg']:.2f}deg "
                f"physics={va['physics_center_mm']:.2f}mm/{va['physics_theta_deg']:.2f}deg"
            )
            if "real_learned_center_mm" in va:
                msg += (
                    f" | REAL learned={va['real_learned_center_mm']:.2f}mm/{va['real_learned_theta_deg']:.2f}deg "
                    f"physics={va['real_physics_center_mm']:.2f}mm/{va['real_physics_theta_deg']:.2f}deg"
                )
            msg += f" | best_epoch={best_epoch}"
            print(msg, flush=True)

    print(f"[DONE] best_epoch={best_epoch} best_score={best_score:.6f}")
    print(f"[DONE] saved: {output_path}")


if __name__ == "__main__":
    main()
