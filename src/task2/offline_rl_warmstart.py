import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


# ============================================================
# Compatibility constants (must match BC / teacher pipeline)
# ============================================================
DEFAULT_LENGTH_BINS = (0.03, 0.05, 0.07, 0.09, 0.12, 0.15)
DEFAULT_NUM_THETA_BINS = 40
NODE_FEATURE_DIM = 11
EPS = 1e-8
WORKSPACE_SCALE_M = 0.15


# ============================================================
# Geometry helpers
# ============================================================
def build_chain_adjacency(num_nodes: int) -> torch.Tensor:
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for i in range(num_nodes - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return adj


def _normalize_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm = np.clip(norm, eps, None)
    return v / norm


def compute_edge_tangents(points: np.ndarray) -> np.ndarray:
    edges = points[1:] - points[:-1]
    return _normalize_rows(edges)


def compute_node_tangents(points: np.ndarray) -> np.ndarray:
    num_nodes = points.shape[0]
    node_tangent = np.zeros((num_nodes, 2), dtype=np.float32)
    if num_nodes == 1:
        return node_tangent

    edge_tangent = compute_edge_tangents(points)
    node_tangent[0] = edge_tangent[0]
    node_tangent[-1] = edge_tangent[-1]
    for i in range(1, num_nodes - 1):
        t = edge_tangent[i - 1] + edge_tangent[i]
        n = np.linalg.norm(t)
        node_tangent[i] = edge_tangent[i] if n < 1e-12 else t / n
    return node_tangent.astype(np.float32)


def compute_node_curvature_abs(points: np.ndarray) -> np.ndarray:
    num_nodes = points.shape[0]
    curv = np.zeros((num_nodes,), dtype=np.float32)
    if num_nodes < 3:
        return curv

    t = compute_edge_tangents(points)
    t_prev = t[:-1]
    t_next = t[1:]
    dots = np.sum(t_prev * t_next, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    crosses = t_prev[:, 0] * t_next[:, 1] - t_prev[:, 1] * t_next[:, 0]
    phi = np.arctan2(crosses, dots)
    phi = np.clip(phi, -np.pi + 1e-4, np.pi - 1e-4)
    discrete_curvature = 2.0 * np.tan(0.5 * phi)
    curv[1:-1] = np.abs(discrete_curvature).astype(np.float32)
    return curv


def build_bc_node_features(current_points: np.ndarray, goal_points: np.ndarray) -> np.ndarray:
    """Same 11-D node feature used by BC actor.

    x_i = [cur_x, cur_y, goal_x, goal_y, delta_x, delta_y,
           idx_norm, dist_to_goal, tangent_x, tangent_y, curvature_abs]
    """
    current_points = np.asarray(current_points, dtype=np.float32)
    goal_points = np.asarray(goal_points, dtype=np.float32)
    num_nodes = current_points.shape[0]

    delta = goal_points - current_points
    idx_norm = (np.arange(num_nodes, dtype=np.float32) / max(1, num_nodes - 1)).reshape(-1, 1)
    dist_to_goal = np.linalg.norm(delta, axis=1).astype(np.float32).reshape(-1, 1)
    tangent = compute_node_tangents(current_points)
    curvature = compute_node_curvature_abs(current_points).reshape(-1, 1)

    x = np.concatenate(
        [current_points, goal_points, delta, idx_norm, dist_to_goal, tangent, curvature],
        axis=1,
    )
    return x.astype(np.float32)


# ============================================================
# Action binning helpers
# ============================================================
def canonicalize_theta(theta: float) -> float:
    theta = float(theta)
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


def make_theta_bins(num_bins: int) -> np.ndarray:
    return np.linspace(-np.pi, np.pi, num_bins, endpoint=False, dtype=np.float32)


def nearest_bin_index(value: float, bins: Sequence[float]) -> int:
    bins_arr = np.asarray(bins, dtype=np.float32)
    return int(np.argmin(np.abs(bins_arr - float(value))))


def nearest_theta_bin_index(theta: float, theta_bins: np.ndarray) -> int:
    theta = canonicalize_theta(theta)
    diff = np.angle(np.exp(1j * (theta_bins - theta)))
    return int(np.argmin(np.abs(diff)))


def action_to_indices(action: Dict[str, object], length_bins: Sequence[float], theta_bins: np.ndarray, num_nodes: int) -> Optional[Tuple[int, int, int]]:
    try:
        node_idx = int(action["node_idx"])
        length = float(action.get("length", action.get("move_norm", 0.0)))
        theta = float(action.get("theta", 0.0))
    except Exception:
        return None
    if node_idx <= 0 or node_idx >= num_nodes:
        return None
    return node_idx, nearest_bin_index(length, length_bins), nearest_theta_bin_index(theta, theta_bins)


def candidate_metric(item: Dict[str, object], source: str, fallback: float) -> float:
    """Lower metric is better."""
    try:
        if source == "mean_err_mm":
            return float(item.get("mean_err_mm", fallback))
        if source == "rmse_mm":
            return float(item.get("rmse_mm", fallback))
        if source == "total_cost":
            return float(item.get("cost", {}).get("total", fallback))
        if source == "shape_cost":
            return float(item.get("cost", {}).get("shape", fallback))
    except Exception:
        return float(fallback)
    raise ValueError(f"Unknown candidate_score_source: {source}")


# ============================================================
# Dataset parser
# ============================================================
@dataclass
class TransitionSample:
    x: np.ndarray
    next_x: np.ndarray
    node_target: int
    len_target: int
    theta_target: int
    reward: float
    done: float
    success: float

    # Candidate-aware fields, fixed length K=candidate_topn.
    cand_node: np.ndarray
    cand_len: np.ndarray
    cand_theta: np.ndarray
    cand_metric: np.ndarray          # lower is better, e.g. mean_err_mm
    cand_q_target: np.ndarray        # optional supervised critic target
    cand_mask: np.ndarray            # 1 for valid candidate


def discover_worker_jsonls(teacher_dir: str, pattern: str = 'teacher_run_worker*.jsonl') -> List[Path]:
    """
    Accept either:
      1) a folder containing teacher_run_worker*.jsonl files
      2) one merged .jsonl file, e.g. merged_teacher_run_worker_all.jsonl
    """
    source_path = Path(teacher_dir)

    if source_path.is_file():
        if source_path.suffix.lower() != ".jsonl":
            raise FileNotFoundError(f"teacher_dir is a file but not a .jsonl file: {source_path}")
        return [source_path]

    if source_path.is_dir():
        files = sorted(source_path.glob(pattern))
        if len(files) == 0:
            raise FileNotFoundError(f"No worker jsonl files found in {source_path} with pattern {pattern}")
        return files

    raise FileNotFoundError(f"teacher_dir path not found: {source_path}")


def _fallback_candidate_from_selected(step: Dict[str, object]) -> Dict[str, object]:
    return {
        "rank": 1,
        "action": step["selected_action"],
        "cost": step.get("selected_cost", {}),
        "mean_err_mm": float(step.get("selected_mean_err_mm", step.get("after_mean_err_mm", 0.0))),
        "rmse_mm": float(step.get("selected_rmse_mm", step.get("after_rmse_mm", 0.0))),
    }


def parse_candidates_for_step(
    step: Dict[str, object],
    length_bins: Sequence[float],
    theta_bins: np.ndarray,
    num_nodes: int,
    candidate_topn: int,
    candidate_score_source: str,
    candidate_q_scale: float,
    candidate_q_target_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse ranked_candidates_topn into fixed-size arrays.

    - cand_metric: lower is better.
    - cand_q_target: higher is better. Default relative mode: best candidate is 0,
      worse candidates are negative in proportion to metric gap.
    """
    ranked = step.get("ranked_candidates_topn") or []
    if not isinstance(ranked, list) or len(ranked) == 0:
        ranked = [_fallback_candidate_from_selected(step)]

    parsed = []
    fallback_metric = float(step.get("selected_mean_err_mm", step.get("after_mean_err_mm", 0.0)))
    for item in ranked:
        if not isinstance(item, dict):
            continue
        action = item.get("action")
        if not isinstance(action, dict):
            continue
        idxs = action_to_indices(action, length_bins, theta_bins, num_nodes)
        if idxs is None:
            continue
        metric = candidate_metric(item, candidate_score_source, fallback=fallback_metric)
        if not np.isfinite(metric):
            continue
        parsed.append((*idxs, float(metric)))
        if len(parsed) >= candidate_topn:
            break

    if len(parsed) == 0:
        action = step["selected_action"]
        idxs = action_to_indices(action, length_bins, theta_bins, num_nodes)
        if idxs is None:
            idxs = (1, 0, 0)
        parsed.append((*idxs, fallback_metric))

    K = int(candidate_topn)
    cand_node = np.zeros((K,), dtype=np.int64)
    cand_len = np.zeros((K,), dtype=np.int64)
    cand_theta = np.zeros((K,), dtype=np.int64)
    cand_metric = np.zeros((K,), dtype=np.float32)
    cand_mask = np.zeros((K,), dtype=np.float32)

    for k, (node_idx, len_idx, theta_idx, metric) in enumerate(parsed[:K]):
        cand_node[k] = int(node_idx)
        cand_len[k] = int(len_idx)
        cand_theta[k] = int(theta_idx)
        cand_metric[k] = float(metric)
        cand_mask[k] = 1.0

    valid_metrics = cand_metric[cand_mask > 0.5]
    best_metric = float(np.min(valid_metrics)) if len(valid_metrics) > 0 else 0.0

    if candidate_q_target_mode == "relative":
        # Best candidate gets 0. Worse candidates get negative values.
        cand_q_target = (best_metric - cand_metric) * float(candidate_q_scale)
    elif candidate_q_target_mode == "negative_metric":
        cand_q_target = -cand_metric * float(candidate_q_scale)
    else:
        raise ValueError(f"Unknown candidate_q_target_mode: {candidate_q_target_mode}")

    cand_q_target = cand_q_target.astype(np.float32)
    cand_q_target[cand_mask < 0.5] = 0.0
    return cand_node, cand_len, cand_theta, cand_metric, cand_q_target, cand_mask


def parse_worker_file_to_transitions(
    path: Path,
    length_bins: Sequence[float],
    num_theta_bins: int,
    terminal_bonus: float,
    reward_scale: float,
    reward_clip: Optional[float],
    candidate_topn: int,
    candidate_score_source: str,
    candidate_q_scale: float,
    candidate_q_target_mode: str,
    max_teacher_step_idx: Optional[int] = None,
) -> Tuple[List[TransitionSample], int]:
    theta_bins = make_theta_bins(num_theta_bins)
    transitions: List[TransitionSample] = []
    kept_steps = 0

    current_episode_steps: List[dict] = []
    current_episode_success = False

    def flush_episode():
        nonlocal transitions, current_episode_steps, current_episode_success
        if len(current_episode_steps) == 0:
            return
        for i, step in enumerate(current_episode_steps):
            current_points = np.asarray(step['before_state_xy'], dtype=np.float32)
            goal_points = np.asarray(step['goal_state_xy'], dtype=np.float32)
            next_points = np.asarray(step['after_pred_xy'], dtype=np.float32)
            action = step['selected_action']
            num_nodes = current_points.shape[0]

            selected = action_to_indices(action, length_bins, theta_bins, num_nodes)
            if selected is None:
                continue
            node_idx, len_target, theta_target = selected

            x = build_bc_node_features(current_points, goal_points)
            next_x = build_bc_node_features(next_points, goal_points)

            reward = float(step.get('improvement_mean_err_mm', 0.0)) * reward_scale
            done = 1.0 if i == len(current_episode_steps) - 1 else 0.0
            success = 1.0 if current_episode_success else 0.0
            if done > 0.5 and current_episode_success:
                reward += terminal_bonus
            if reward_clip is not None:
                reward = float(np.clip(reward, -reward_clip, reward_clip))

            cand_node, cand_len, cand_theta, cand_metric, cand_q_target, cand_mask = parse_candidates_for_step(
                step=step,
                length_bins=length_bins,
                theta_bins=theta_bins,
                num_nodes=num_nodes,
                candidate_topn=candidate_topn,
                candidate_score_source=candidate_score_source,
                candidate_q_scale=candidate_q_scale,
                candidate_q_target_mode=candidate_q_target_mode,
            )

            transitions.append(
                TransitionSample(
                    x=x,
                    next_x=next_x,
                    node_target=node_idx,
                    len_target=len_target,
                    theta_target=theta_target,
                    reward=reward,
                    done=done,
                    success=success,
                    cand_node=cand_node,
                    cand_len=cand_len,
                    cand_theta=cand_theta,
                    cand_metric=cand_metric,
                    cand_q_target=cand_q_target,
                    cand_mask=cand_mask,
                )
            )
        current_episode_steps = []
        current_episode_success = False

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rtype = obj.get('record_type')

            if rtype == 'episode_start':
                flush_episode()
                current_episode_steps = []
                current_episode_success = False
            elif rtype == 'step':
                if max_teacher_step_idx is not None and int(obj.get('step_idx', 0)) > max_teacher_step_idx:
                    continue
                current_episode_steps.append(obj)
                kept_steps += 1
            elif rtype == 'episode_end':
                current_episode_success = bool(obj.get('success', False))
                flush_episode()

    flush_episode()
    return transitions, kept_steps


def load_offline_rl_transitions(
    teacher_dir: str,
    length_bins: Sequence[float],
    num_theta_bins: int,
    terminal_bonus: float,
    reward_scale: float,
    reward_clip: Optional[float],
    candidate_topn: int,
    candidate_score_source: str,
    candidate_q_scale: float,
    candidate_q_target_mode: str,
    pattern: str = 'teacher_run_worker*.jsonl',
    max_teacher_step_idx: Optional[int] = None,
    max_records: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List[TransitionSample], int]:
    files = discover_worker_jsonls(teacher_dir, pattern=pattern)
    all_samples: List[TransitionSample] = []
    num_nodes_ref: Optional[int] = None
    total_raw_steps = 0

    if verbose:
        print(f'[DATA] worker files found: {len(files)}')
        for p in files:
            print(f'  - {p}')

    for path in files:
        file_samples, file_steps = parse_worker_file_to_transitions(
            path=path,
            length_bins=length_bins,
            num_theta_bins=num_theta_bins,
            terminal_bonus=terminal_bonus,
            reward_scale=reward_scale,
            reward_clip=reward_clip,
            candidate_topn=candidate_topn,
            candidate_score_source=candidate_score_source,
            candidate_q_scale=candidate_q_scale,
            candidate_q_target_mode=candidate_q_target_mode,
            max_teacher_step_idx=max_teacher_step_idx,
        )
        total_raw_steps += file_steps
        if len(file_samples) > 0:
            n = file_samples[0].x.shape[0]
            if num_nodes_ref is None:
                num_nodes_ref = n
            elif n != num_nodes_ref:
                raise ValueError(f'Inconsistent num_nodes: got {n}, expected {num_nodes_ref}')

        if max_records is not None:
            remain = max_records - len(all_samples)
            if remain <= 0:
                break
            all_samples.extend(file_samples[:remain])
        else:
            all_samples.extend(file_samples)

        if verbose:
            print(f'[DATA] parsed {path.name}: raw_steps={file_steps} | transitions_total={len(all_samples)}')
        if max_records is not None and len(all_samples) >= max_records:
            break

    if len(all_samples) == 0:
        raise RuntimeError('No offline RL transitions were loaded.')

    if verbose:
        rewards = np.array([s.reward for s in all_samples], dtype=np.float32)
        cand_counts = np.array([np.sum(s.cand_mask) for s in all_samples], dtype=np.float32)
        cand_metrics = np.concatenate([s.cand_metric[s.cand_mask > 0.5] for s in all_samples]).astype(np.float32)
        done_ratio = float(np.mean([s.done for s in all_samples]))
        success_ratio = float(np.mean([s.success for s in all_samples]))
        print(f'[DATA] total raw steps seen: {total_raw_steps}')
        print(f'[DATA] total transitions kept: {len(all_samples)}')
        print(f'[DATA] reward mean={rewards.mean():.4f} std={rewards.std():.4f} min={rewards.min():.4f} max={rewards.max():.4f}')
        print(f'[DATA] candidate count mean={cand_counts.mean():.2f} | metric mean={cand_metrics.mean():.4f} min={cand_metrics.min():.4f} max={cand_metrics.max():.4f}')
        print(f'[DATA] done ratio={done_ratio:.4f} | success-final ratio={success_ratio:.4f}')

    return all_samples, int(num_nodes_ref)


class OfflineRLDataset(Dataset):
    def __init__(self, samples: List[TransitionSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            'x': torch.tensor(s.x, dtype=torch.float32),
            'next_x': torch.tensor(s.next_x, dtype=torch.float32),
            'node_target': torch.tensor(s.node_target, dtype=torch.long),
            'len_target': torch.tensor(s.len_target, dtype=torch.long),
            'theta_target': torch.tensor(s.theta_target, dtype=torch.long),
            'reward': torch.tensor(s.reward, dtype=torch.float32),
            'done': torch.tensor(s.done, dtype=torch.float32),
            'success': torch.tensor(s.success, dtype=torch.float32),
            'cand_node': torch.tensor(s.cand_node, dtype=torch.long),
            'cand_len': torch.tensor(s.cand_len, dtype=torch.long),
            'cand_theta': torch.tensor(s.cand_theta, dtype=torch.long),
            'cand_metric': torch.tensor(s.cand_metric, dtype=torch.float32),
            'cand_q_target': torch.tensor(s.cand_q_target, dtype=torch.float32),
            'cand_mask': torch.tensor(s.cand_mask, dtype=torch.float32),
        }


def collate_offline_rl(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}


# ============================================================
# Model definitions
# ============================================================
class RopeMessagePassing(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        msg = self.lin(x)
        return torch.matmul(adj, msg)


class GoalConditionedBCActor(nn.Module):
    def __init__(self, node_feature_dim: int, hidden_dim: int, num_length_bins: int, num_theta_bins: int):
        super().__init__()
        self.node_embed = nn.Linear(node_feature_dim, hidden_dim)
        self.gnn1 = RopeMessagePassing(hidden_dim, hidden_dim)
        self.gnn2 = RopeMessagePassing(hidden_dim, hidden_dim)

        self.node_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.length_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_length_bins))
        self.theta_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_theta_bins))

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.node_embed(x)
        h = F.relu(self.gnn1(h, adj))
        h = F.relu(self.gnn2(h, adj))
        return h

    def forward(self, x: torch.Tensor, adj: torch.Tensor, teacher_node_idx: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(x.shape[0], -1, -1)

        h = self.encode(x, adj)
        g = h.mean(dim=1)
        g_expand = g.unsqueeze(1).expand(-1, h.shape[1], -1)
        node_feat = torch.cat([h, g_expand], dim=-1)
        node_logits = self.node_head(node_feat).squeeze(-1)
        node_logits[:, 0] = -1e9

        if teacher_node_idx is None:
            chosen_node = torch.argmax(node_logits, dim=1)
        else:
            chosen_node = teacher_node_idx

        batch_idx = torch.arange(x.shape[0], device=x.device)
        h_sel = h[batch_idx, chosen_node]
        z = torch.cat([h_sel, g], dim=-1)
        length_logits = self.length_head(z)
        theta_logits = self.theta_head(z)
        return {'node_logits': node_logits, 'length_logits': length_logits, 'theta_logits': theta_logits}


class FactorizedQCritic(nn.Module):
    """Q(s,a) ≈ Q_node(i) + Q_len(r|i) + Q_theta(theta|i)."""
    def __init__(self, node_feature_dim: int, hidden_dim: int, num_length_bins: int, num_theta_bins: int):
        super().__init__()
        self.node_embed = nn.Linear(node_feature_dim, hidden_dim)
        self.gnn1 = RopeMessagePassing(hidden_dim, hidden_dim)
        self.gnn2 = RopeMessagePassing(hidden_dim, hidden_dim)
        self.node_q_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.length_q_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_length_bins))
        self.theta_q_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_theta_bins))

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(x.shape[0], -1, -1)
        h = F.relu(self.gnn1(self.node_embed(x), adj))
        h = F.relu(self.gnn2(h, adj))
        g = h.mean(dim=1)
        return h, g

    def all_q(self, x: torch.Tensor, adj: torch.Tensor, chosen_node_idx: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        h, g = self.encode(x, adj)
        g_expand = g.unsqueeze(1).expand(-1, h.shape[1], -1)
        node_feat = torch.cat([h, g_expand], dim=-1)
        node_q = self.node_q_head(node_feat).squeeze(-1)
        node_q[:, 0] = -1e9

        if chosen_node_idx is None:
            chosen_node_idx = torch.argmax(node_q, dim=1)
        batch_idx = torch.arange(x.shape[0], device=x.device)
        h_sel = h[batch_idx, chosen_node_idx]
        z = torch.cat([h_sel, g], dim=-1)
        length_q = self.length_q_head(z)
        theta_q = self.theta_q_head(z)
        return {'node_q': node_q, 'length_q': length_q, 'theta_q': theta_q, 'chosen_node_idx': chosen_node_idx}

    def q_of_action(self, x: torch.Tensor, adj: torch.Tensor, node_idx: torch.Tensor, len_idx: torch.Tensor, theta_idx: torch.Tensor) -> torch.Tensor:
        out = self.all_q(x, adj, chosen_node_idx=node_idx)
        batch_idx = torch.arange(x.shape[0], device=x.device)
        q_node = out['node_q'][batch_idx, node_idx]
        q_len = out['length_q'][batch_idx, len_idx]
        q_theta = out['theta_q'][batch_idx, theta_idx]
        return q_node + q_len + q_theta


# ============================================================
# Loading helpers
# ============================================================
def load_bc_actor_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[GoalConditionedBCActor, Dict[str, object]]:
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state_dict' not in obj:
        raise ValueError('BC checkpoint must contain model_state_dict.')

    hidden_dim = int(obj['hidden_dim'])
    num_length_bins = len(obj.get('length_bins', list(DEFAULT_LENGTH_BINS)))
    num_theta_bins = int(obj.get('num_theta_bins', DEFAULT_NUM_THETA_BINS))
    node_feature_dim = int(obj.get('node_feature_dim', NODE_FEATURE_DIM))
    if node_feature_dim != NODE_FEATURE_DIM:
        raise ValueError(f'BC checkpoint node_feature_dim={node_feature_dim}, expected {NODE_FEATURE_DIM}')

    actor = GoalConditionedBCActor(NODE_FEATURE_DIM, hidden_dim, num_length_bins, num_theta_bins).to(device)
    actor.load_state_dict(obj['model_state_dict'])
    actor.eval()
    return actor, obj


# ============================================================
# Loss / metrics helpers
# ============================================================
def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def categorical_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets, reduction='none')


def actor_bc_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    l_node = categorical_ce(out['node_logits'], batch['node_target'])
    l_len = categorical_ce(out['length_logits'], batch['len_target'])
    l_theta = categorical_ce(out['theta_logits'], batch['theta_target'])
    per_sample = l_node + l_len + l_theta
    return per_sample.mean(), {'node': l_node, 'len': l_len, 'theta': l_theta, 'sum': per_sample}


def greedy_policy_action(actor: GoalConditionedBCActor, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    out1 = actor(x, adj, teacher_node_idx=None)
    node_idx = out1['node_logits'].argmax(dim=1)
    out2 = actor(x, adj, teacher_node_idx=node_idx)
    len_idx = out2['length_logits'].argmax(dim=1)
    theta_idx = out2['theta_logits'].argmax(dim=1)
    merged = {'node_logits': out1['node_logits'], 'length_logits': out2['length_logits'], 'theta_logits': out2['theta_logits']}
    return node_idx, len_idx, theta_idx, merged


def q_value_of_greedy_action(critic: FactorizedQCritic, actor: GoalConditionedBCActor, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    node_idx, len_idx, theta_idx, _ = greedy_policy_action(actor, x, adj)
    return critic.q_of_action(x, adj, node_idx, len_idx, theta_idx)


def target_q_value(critic_t1: FactorizedQCritic, critic_t2: FactorizedQCritic, actor: GoalConditionedBCActor, next_x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    node_idx, len_idx, theta_idx, _ = greedy_policy_action(actor, next_x, adj)
    q1 = critic_t1.q_of_action(next_x, adj, node_idx, len_idx, theta_idx)
    q2 = critic_t2.q_of_action(next_x, adj, node_idx, len_idx, theta_idx)
    return torch.minimum(q1, q2)


def q_values_for_candidates(critic: FactorizedQCritic, x: torch.Tensor, adj: torch.Tensor, cand_node: torch.Tensor, cand_len: torch.Tensor, cand_theta: torch.Tensor) -> torch.Tensor:
    B, K = cand_node.shape
    x_rep = x.unsqueeze(1).expand(B, K, x.shape[1], x.shape[2]).reshape(B * K, x.shape[1], x.shape[2])
    q_flat = critic.q_of_action(
        x_rep,
        adj,
        cand_node.reshape(-1),
        cand_len.reshape(-1),
        cand_theta.reshape(-1),
    )
    return q_flat.reshape(B, K)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(eps)


def candidate_q_supervised_loss(q1: torch.Tensor, q2: torch.Tensor, q_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = (q1 - q_target).pow(2) + (q2 - q_target).pow(2)
    return masked_mean(loss, mask)


def candidate_ranking_loss(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Assumes candidates are ordered best -> worst by MPC rank.

    Encourages Q_i > Q_j for i < j. Uses softplus(-(Q_i-Q_j)).
    """
    B, K = q_values.shape
    losses = []
    masks = []
    for i in range(K):
        for j in range(i + 1, K):
            pair_mask = mask[:, i] * mask[:, j]
            if torch.any(pair_mask > 0.5):
                losses.append(F.softplus(-(q_values[:, i] - q_values[:, j])))
                masks.append(pair_mask)
    if len(losses) == 0:
        return q_values.sum() * 0.0
    loss_t = torch.stack(losses, dim=1)
    mask_t = torch.stack(masks, dim=1)
    return masked_mean(loss_t, mask_t)


def log_prob_actions(actor: GoalConditionedBCActor, x: torch.Tensor, adj: torch.Tensor, node_idx: torch.Tensor, len_idx: torch.Tensor, theta_idx: torch.Tensor) -> torch.Tensor:
    out_node = actor(x, adj, teacher_node_idx=None)
    log_node = F.log_softmax(out_node['node_logits'], dim=1)
    selected_node_logp = log_node.gather(1, node_idx.view(-1, 1)).squeeze(1)

    out_cond = actor(x, adj, teacher_node_idx=node_idx)
    log_len = F.log_softmax(out_cond['length_logits'], dim=1)
    log_theta = F.log_softmax(out_cond['theta_logits'], dim=1)
    selected_len_logp = log_len.gather(1, len_idx.view(-1, 1)).squeeze(1)
    selected_theta_logp = log_theta.gather(1, theta_idx.view(-1, 1)).squeeze(1)
    return selected_node_logp + selected_len_logp + selected_theta_logp


def candidate_actor_soft_loss(actor: GoalConditionedBCActor, x: torch.Tensor, adj: torch.Tensor, batch: Dict[str, torch.Tensor], temp: float) -> torch.Tensor:
    B, K = batch['cand_node'].shape
    x_rep = x.unsqueeze(1).expand(B, K, x.shape[1], x.shape[2]).reshape(B * K, x.shape[1], x.shape[2])
    logp_flat = log_prob_actions(
        actor,
        x_rep,
        adj,
        batch['cand_node'].reshape(-1),
        batch['cand_len'].reshape(-1),
        batch['cand_theta'].reshape(-1),
    )
    logp = logp_flat.reshape(B, K)

    # Lower candidate metric is better -> higher soft target weight.
    scores = -batch['cand_metric'] / max(float(temp), 1e-6)
    scores = scores.masked_fill(batch['cand_mask'] < 0.5, -1e9)
    weights = F.softmax(scores, dim=1) * batch['cand_mask']
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return -(weights.detach() * logp).sum(dim=1).mean()


def batch_action_metrics(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    node_pred = out['node_logits'].argmax(dim=1)
    len_pred = out['length_logits'].argmax(dim=1)
    theta_pred = out['theta_logits'].argmax(dim=1)
    node_acc = (node_pred == batch['node_target']).float().mean().item()
    len_acc = (len_pred == batch['len_target']).float().mean().item()
    theta_acc = (theta_pred == batch['theta_target']).float().mean().item()
    triple_acc = ((node_pred == batch['node_target']) & (len_pred == batch['len_target']) & (theta_pred == batch['theta_target'])).float().mean().item()
    return {'node_acc': float(node_acc), 'len_acc': float(len_acc), 'theta_acc': float(theta_acc), 'triple_acc': float(triple_acc)}


# ============================================================
# Candidate-aware offline RL train loop
# ============================================================
def run_epoch_offline_rl(
    actor: GoalConditionedBCActor,
    critic1: FactorizedQCritic,
    critic2: FactorizedQCritic,
    target_critic1: FactorizedQCritic,
    target_critic2: FactorizedQCritic,
    loader: DataLoader,
    adj: torch.Tensor,
    device: torch.device,
    actor_opt: Optional[torch.optim.Optimizer],
    critic_opt: Optional[torch.optim.Optimizer],
    gamma: float,
    bc_coef: float,
    awac_lambda: float,
    target_update_tau: float,
    qsup_coef: float,
    rank_coef: float,
    actor_cand_coef: float,
    candidate_soft_temp: float,
    log_interval: int,
    epoch_idx: int,
    split_name: str,
) -> Dict[str, float]:
    train = actor_opt is not None and critic_opt is not None
    actor.train() if train else actor.eval()
    critic1.train() if train else critic1.eval()
    critic2.train() if train else critic2.eval()

    stat_sums = {
        'critic_loss': 0.0,
        'td_loss': 0.0,
        'qsup_loss': 0.0,
        'rank_loss': 0.0,
        'actor_loss': 0.0,
        'bc_loss': 0.0,
        'awac_loss': 0.0,
        'actor_cand_loss': 0.0,
        'avg_reward': 0.0,
        'avg_q_data': 0.0,
        'avg_td_target': 0.0,
        'avg_adv': 0.0,
        'avg_cand_q_gap': 0.0,
        'node_acc': 0.0,
        'len_acc': 0.0,
        'theta_acc': 0.0,
        'triple_acc': 0.0,
    }
    n_batches = 0
    start_t = time.time()

    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)

        with torch.set_grad_enabled(train):
            # -------------------------
            # Critic update: TD + candidate supervised/ranking losses
            # -------------------------
            q1_data = critic1.q_of_action(batch['x'], adj, batch['node_target'], batch['len_target'], batch['theta_target'])
            q2_data = critic2.q_of_action(batch['x'], adj, batch['node_target'], batch['len_target'], batch['theta_target'])
            with torch.no_grad():
                next_q = target_q_value(target_critic1, target_critic2, actor, batch['next_x'], adj)
                td_target = batch['reward'] + gamma * (1.0 - batch['done']) * next_q

            td_loss = F.mse_loss(q1_data, td_target) + F.mse_loss(q2_data, td_target)

            q1_cand = q_values_for_candidates(critic1, batch['x'], adj, batch['cand_node'], batch['cand_len'], batch['cand_theta'])
            q2_cand = q_values_for_candidates(critic2, batch['x'], adj, batch['cand_node'], batch['cand_len'], batch['cand_theta'])
            qsup_loss = candidate_q_supervised_loss(q1_cand, q2_cand, batch['cand_q_target'], batch['cand_mask'])
            q_cand_min = torch.minimum(q1_cand, q2_cand)
            rank_loss = candidate_ranking_loss(q_cand_min, batch['cand_mask'])
            critic_loss = td_loss + qsup_coef * qsup_loss + rank_coef * rank_loss

            # -------------------------
            # Actor update: AWAC selected action + BC + candidate soft target
            # -------------------------
            out = actor(batch['x'], adj, teacher_node_idx=batch['node_target'])
            bc_loss, _ = actor_bc_loss(out, batch)

            with torch.no_grad():
                q1_for_data = critic1.q_of_action(batch['x'], adj, batch['node_target'], batch['len_target'], batch['theta_target'])
                q2_for_data = critic2.q_of_action(batch['x'], adj, batch['node_target'], batch['len_target'], batch['theta_target'])
                q_data = torch.minimum(q1_for_data, q2_for_data)
                v_pi = torch.minimum(
                    q_value_of_greedy_action(critic1, actor, batch['x'], adj),
                    q_value_of_greedy_action(critic2, actor, batch['x'], adj),
                )
                adv = q_data - v_pi
                weights = torch.exp(torch.clamp(adv / max(awac_lambda, 1e-6), max=10.0))
                weights = torch.clamp(weights, 0.0, 50.0)

            l_node = categorical_ce(out['node_logits'], batch['node_target'])
            l_len = categorical_ce(out['length_logits'], batch['len_target'])
            l_theta = categorical_ce(out['theta_logits'], batch['theta_target'])
            awac_loss = ((l_node + l_len + l_theta) * weights).mean()
            actor_cand_loss = candidate_actor_soft_loss(actor, batch['x'], adj, batch, temp=candidate_soft_temp)
            actor_loss = awac_loss + bc_coef * bc_loss + actor_cand_coef * actor_cand_loss

            if train:
                critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_opt.step()

                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_opt.step()

                with torch.no_grad():
                    for p, tp in zip(critic1.parameters(), target_critic1.parameters()):
                        tp.data.mul_(1.0 - target_update_tau).add_(target_update_tau * p.data)
                    for p, tp in zip(critic2.parameters(), target_critic2.parameters()):
                        tp.data.mul_(1.0 - target_update_tau).add_(target_update_tau * p.data)

        metrics = batch_action_metrics(out, batch)
        valid_best = q_cand_min[:, 0]
        valid_last_idx = (batch['cand_mask'].sum(dim=1).long() - 1).clamp_min(0)
        q_last = q_cand_min.gather(1, valid_last_idx.view(-1, 1)).squeeze(1)
        cand_q_gap = (valid_best - q_last).mean()

        stat_sums['critic_loss'] += float(critic_loss.item())
        stat_sums['td_loss'] += float(td_loss.item())
        stat_sums['qsup_loss'] += float(qsup_loss.item())
        stat_sums['rank_loss'] += float(rank_loss.item())
        stat_sums['actor_loss'] += float(actor_loss.item())
        stat_sums['bc_loss'] += float(bc_loss.item())
        stat_sums['awac_loss'] += float(awac_loss.item())
        stat_sums['actor_cand_loss'] += float(actor_cand_loss.item())
        stat_sums['avg_reward'] += float(batch['reward'].mean().item())
        stat_sums['avg_q_data'] += float(q_data.mean().item())
        stat_sums['avg_td_target'] += float(td_target.mean().item())
        stat_sums['avg_adv'] += float(adv.mean().item())
        stat_sums['avg_cand_q_gap'] += float(cand_q_gap.item())
        for k, v in metrics.items():
            stat_sums[k] += v
        n_batches += 1

        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == len(loader)):
            elapsed = time.time() - start_t
            print(
                f'[{split_name.upper()} E{epoch_idx:03d} B{batch_idx:04d}/{len(loader):04d}] '
                f'critic={float(critic_loss.item()):.4f} td={float(td_loss.item()):.4f} '
                f'qsup={float(qsup_loss.item()):.4f} rank={float(rank_loss.item()):.4f} | '
                f'actor={float(actor_loss.item()):.4f} bc={float(bc_loss.item()):.4f} '
                f'awac={float(awac_loss.item()):.4f} cand={float(actor_cand_loss.item()):.4f} | '
                f'node_acc={metrics["node_acc"]:.3f} len_acc={metrics["len_acc"]:.3f} theta_acc={metrics["theta_acc"]:.3f} '
                f'triple={metrics["triple_acc"]:.3f} | elapsed={elapsed:.1f}s',
                flush=True,
            )

    if n_batches == 0:
        raise RuntimeError('Empty dataloader.')
    out_stats = {k: v / n_batches for k, v in stat_sums.items()}
    out_stats['epoch_time_sec'] = time.time() - start_t
    return out_stats


# ============================================================
# Training entry
# ============================================================
def train_offline_rl_from_config(args) -> Path:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    length_bins = tuple(DEFAULT_LENGTH_BINS)
    num_theta_bins = DEFAULT_NUM_THETA_BINS

    print('[RUN] configuration')
    for k, v in sorted(vars(args).items()):
        print(f'  {k}: {v}')

    samples, num_nodes = load_offline_rl_transitions(
        teacher_dir=args.teacher_dir,
        length_bins=length_bins,
        num_theta_bins=num_theta_bins,
        terminal_bonus=args.terminal_bonus,
        reward_scale=args.reward_scale,
        reward_clip=args.reward_clip,
        candidate_topn=args.candidate_topn,
        candidate_score_source=args.candidate_score_source,
        candidate_q_scale=args.candidate_q_scale,
        candidate_q_target_mode=args.candidate_q_target_mode,
        max_teacher_step_idx=args.max_teacher_step_idx,
        max_records=args.max_records,
        verbose=True,
    )
    print(f'[DATA] loaded candidate-aware offline RL transitions: {len(samples)} | num_nodes={num_nodes}')

    dataset = OfflineRLDataset(samples)
    n_train = int(len(dataset) * args.train_ratio)
    n_val = len(dataset) - n_train
    if n_train <= 0 or n_val <= 0:
        raise ValueError(f'Invalid split: total={len(dataset)}, train={n_train}, val={n_val}')

    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    print(f'[DATA] split | train={len(train_set)} | val={len(val_set)}')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_offline_rl)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_offline_rl)

    actor, bc_payload = load_bc_actor_checkpoint(args.bc_model_path, device)
    bc_hidden_dim = int(bc_payload['hidden_dim'])
    bc_num_length_bins = len(bc_payload.get('length_bins', list(DEFAULT_LENGTH_BINS)))
    bc_num_theta_bins = int(bc_payload.get('num_theta_bins', DEFAULT_NUM_THETA_BINS))
    if bc_num_length_bins != len(length_bins) or bc_num_theta_bins != num_theta_bins:
        raise ValueError('BC checkpoint action bins do not match current offline RL config.')

    critic1 = FactorizedQCritic(NODE_FEATURE_DIM, bc_hidden_dim, bc_num_length_bins, bc_num_theta_bins).to(device)
    critic2 = FactorizedQCritic(NODE_FEATURE_DIM, bc_hidden_dim, bc_num_length_bins, bc_num_theta_bins).to(device)
    target_critic1 = FactorizedQCritic(NODE_FEATURE_DIM, bc_hidden_dim, bc_num_length_bins, bc_num_theta_bins).to(device)
    target_critic2 = FactorizedQCritic(NODE_FEATURE_DIM, bc_hidden_dim, bc_num_length_bins, bc_num_theta_bins).to(device)
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())

    adj = build_chain_adjacency(num_nodes).to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, weight_decay=args.weight_decay)
    critic_opt = torch.optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=args.critic_lr, weight_decay=args.weight_decay)

    best_val = float('inf')
    best_epoch = -1
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    global_start = time.time()

    for epoch in range(1, args.epochs + 1):
        bc_coef_now = args.bc_coef_start + (args.bc_coef_end - args.bc_coef_start) * ((epoch - 1) / max(1, args.epochs - 1))
        epoch_start = time.time()
        tr = run_epoch_offline_rl(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            target_critic1=target_critic1,
            target_critic2=target_critic2,
            loader=train_loader,
            adj=adj,
            device=device,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            gamma=args.gamma,
            bc_coef=bc_coef_now,
            awac_lambda=args.awac_lambda,
            target_update_tau=args.target_update_tau,
            qsup_coef=args.qsup_coef,
            rank_coef=args.rank_coef,
            actor_cand_coef=args.actor_cand_coef,
            candidate_soft_temp=args.candidate_soft_temp,
            log_interval=args.log_interval,
            epoch_idx=epoch,
            split_name='train',
        )
        va = run_epoch_offline_rl(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            target_critic1=target_critic1,
            target_critic2=target_critic2,
            loader=val_loader,
            adj=adj,
            device=device,
            actor_opt=None,
            critic_opt=None,
            gamma=args.gamma,
            bc_coef=bc_coef_now,
            awac_lambda=args.awac_lambda,
            target_update_tau=args.target_update_tau,
            qsup_coef=args.qsup_coef,
            rank_coef=args.rank_coef,
            actor_cand_coef=args.actor_cand_coef,
            candidate_soft_temp=args.candidate_soft_temp,
            log_interval=max(1, args.log_interval),
            epoch_idx=epoch,
            split_name='val',
        )
        epoch_sec = time.time() - epoch_start

        print(
            f'[E{epoch:03d}] bc_coef={bc_coef_now:.4f} | '
            f'train critic={tr["critic_loss"]:.4f} actor={tr["actor_loss"]:.4f} | '
            f'val critic={va["critic_loss"]:.4f} actor={va["actor_loss"]:.4f} | '
            f'val td={va["td_loss"]:.4f} qsup={va["qsup_loss"]:.4f} rank={va["rank_loss"]:.4f} cand={va["actor_cand_loss"]:.4f} | '
            f'val triple_acc={va["triple_acc"]:.3f} | '
            f'val q={va["avg_q_data"]:.4f} target={va["avg_td_target"]:.4f} adv={va["avg_adv"]:.4f} cand_gap={va["avg_cand_q_gap"]:.4f} | '
            f'epoch_time={epoch_sec:.1f}s total_time={(time.time()-global_start):.1f}s',
            flush=True,
        )

        val_score = va['actor_loss'] + va['critic_loss']
        if val_score < best_val:
            best_val = val_score
            best_epoch = epoch
            payload = {
                'actor_state_dict': actor.state_dict(),
                'critic1_state_dict': critic1.state_dict(),
                'critic2_state_dict': critic2.state_dict(),
                'target_critic1_state_dict': target_critic1.state_dict(),
                'target_critic2_state_dict': target_critic2.state_dict(),
                'node_feature_dim': NODE_FEATURE_DIM,
                'hidden_dim': bc_hidden_dim,
                'num_nodes': num_nodes,
                'length_bins': list(length_bins),
                'num_theta_bins': num_theta_bins,
                'train_ratio': args.train_ratio,
                'best_epoch': best_epoch,
                'best_val_score': best_val,
                'val_metrics': va,
                'source_bc_model_path': str(args.bc_model_path),
                'offline_teacher_dir': str(args.teacher_dir),
                'gamma': args.gamma,
                'awac_lambda': args.awac_lambda,
                'bc_coef_at_best': bc_coef_now,
                'candidate_topn': args.candidate_topn,
                'candidate_score_source': args.candidate_score_source,
                'candidate_q_scale': args.candidate_q_scale,
                'candidate_q_target_mode': args.candidate_q_target_mode,
                'qsup_coef': args.qsup_coef,
                'rank_coef': args.rank_coef,
                'actor_cand_coef': args.actor_cand_coef,
                'candidate_soft_temp': args.candidate_soft_temp,
                'note': 'Candidate-aware offline RL warm-start. Uses selected_action transition plus ranked_candidates_topn cost/ranking information.',
            }
            torch.save(payload, output_path)
            print(f'[SAVE] best updated @ epoch {best_epoch}: {output_path}', flush=True)

    print(f'[DONE] best_epoch={best_epoch} | best_val_score={best_val:.6f}')
    return output_path


# ============================================================
# Argparse / direct run
# ============================================================
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Candidate-aware offline RL warm-start compatible with BC actor + teacher worker JSONLs')
    parser.add_argument('--teacher_dir', type=str, required=True, help='folder containing teacher_run_worker*.jsonl OR one merged .jsonl file')
    parser.add_argument('--bc_model_path', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--critic_lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--awac_lambda', type=float, default=0.5)
    parser.add_argument('--bc_coef_start', type=float, default=1.0)
    parser.add_argument('--bc_coef_end', type=float, default=0.1)
    parser.add_argument('--target_update_tau', type=float, default=0.01)

    parser.add_argument('--terminal_bonus', type=float, default=1.0)
    parser.add_argument('--reward_scale', type=float, default=0.1)
    parser.add_argument('--reward_clip', type=float, default=10.0)
    parser.add_argument('--max_teacher_step_idx', type=int, default=None)
    parser.add_argument('--max_records', type=int, default=None)

    parser.add_argument('--candidate_topn', type=int, default=3)
    parser.add_argument('--candidate_score_source', type=str, default='mean_err_mm', choices=['mean_err_mm', 'rmse_mm', 'total_cost', 'shape_cost'])
    parser.add_argument('--candidate_q_scale', type=float, default=0.1)
    parser.add_argument('--candidate_q_target_mode', type=str, default='relative', choices=['relative', 'negative_metric'])
    parser.add_argument('--qsup_coef', type=float, default=0.05)
    parser.add_argument('--rank_coef', type=float, default=0.05)
    parser.add_argument('--actor_cand_coef', type=float, default=0.3)
    parser.add_argument('--candidate_soft_temp', type=float, default=5.0)

    parser.add_argument('--log_interval', type=int, default=50)
    return parser


def resolve_run_path(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((Path(__file__).resolve().parent / p).resolve())


def validate_direct_run_config(cfg: Dict[str, object]) -> Dict[str, object]:
    cfg = dict(cfg)
    cfg['teacher_dir'] = resolve_run_path(str(cfg['teacher_dir']))
    cfg['bc_model_path'] = resolve_run_path(str(cfg['bc_model_path']))
    cfg['output'] = resolve_run_path(str(cfg['output']))

    missing = []
    teacher_path = Path(cfg['teacher_dir'])
    if not (teacher_path.is_dir() or teacher_path.is_file()):
        missing.append(f"teacher_dir / teacher jsonl path not found: {cfg['teacher_dir']}")
    elif teacher_path.is_file() and teacher_path.suffix.lower() != ".jsonl":
        missing.append(f"teacher_dir is a file but not .jsonl: {cfg['teacher_dir']}")

    if not Path(cfg['bc_model_path']).is_file():
        missing.append(f"bc_model_path not found: {cfg['bc_model_path']}")

    Path(cfg['output']).parent.mkdir(parents=True, exist_ok=True)
    if missing:
        raise FileNotFoundError('DIRECT_RUN_CONFIG path error\n- ' + '\n- '.join(missing))
    return cfg


def print_direct_run_config(cfg: Dict[str, object]) -> None:
    print('===== DIRECT RUN CONFIG =====')
    for k, v in cfg.items():
        print(f'{k}: {v}')
    print('=============================')


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    train_offline_rl_from_config(args)


USE_DIRECT_RUN = True

DIRECT_RUN_CONFIG = dict(
    teacher_dir=r"C:\Users\wlsdud\Desktop\Task2DYN\DERTRAIN\merged_teacher_run_worker_all.jsonl",
    bc_model_path=r"C:\Users\wlsdud\Desktop\Task2DYN\DERTRAIN\bc_actor_teacher.pt",
    output=r"C:\Users\wlsdud\Desktop\Task2DYN\DERTRAIN\offline_rl_candidate_aware_epoch10001.pt",

    epochs=1000,
    batch_size=64,
    actor_lr=1e-4,
    critic_lr=3e-4,
    weight_decay=1e-6,
    train_ratio=0.8,
    seed=42,
    device='cuda' if torch.cuda.is_available() else 'cpu',

    gamma=0.98,
    awac_lambda=0.5,
    bc_coef_start=1.0,
    bc_coef_end=0.1,
    target_update_tau=0.01,

    terminal_bonus=1.0,
    reward_scale=0.1,
    reward_clip=10.0,
    max_teacher_step_idx=None,
    max_records=None,

    # New candidate-aware settings. These match teacher JSONL ranked_candidates_topn.
    candidate_topn=3,
    candidate_score_source='mean_err_mm',
    candidate_q_scale=0.1,
    candidate_q_target_mode='relative',
    qsup_coef=0.05,
    rank_coef=0.05,
    actor_cand_coef=0.3,
    candidate_soft_temp=5.0,

    log_interval=50,
)


if __name__ == '__main__':
    if USE_DIRECT_RUN:
        from types import SimpleNamespace
        resolved_cfg = validate_direct_run_config(DIRECT_RUN_CONFIG)
        print_direct_run_config(resolved_cfg)
        train_offline_rl_from_config(SimpleNamespace(**resolved_cfg))
    else:
        main()
