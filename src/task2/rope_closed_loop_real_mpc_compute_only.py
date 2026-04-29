import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from der_nonmouse import TabletopSoftRope


# ============================================================
# Feature dimensions (must match current exp2 train/vis setup)
# ============================================================

NODE_FEATURE_DIM = 11
ACTION_DIM = 8
WORKSPACE_SCALE_M = 0.15  # normalized coord -> meters
EPS = 1e-8


# ============================================================
# Model (compatible with current checkpoint)
# ============================================================

class RopeMessagePassing(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        msg = self.lin(x)
        return torch.matmul(adj, msg)


class GraphDLO(nn.Module):
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.node_embed = nn.Linear(NODE_FEATURE_DIM, hidden_dim)
        self.action_embed = nn.Linear(ACTION_DIM, hidden_dim)
        self.gnn1 = RopeMessagePassing(hidden_dim, hidden_dim)
        self.gnn2 = RopeMessagePassing(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, action: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        node_feat = self.node_embed(x)
        action_feat = self.action_embed(action).unsqueeze(1).repeat(1, x.shape[1], 1)
        h = node_feat + action_feat
        h = torch.relu(self.gnn1(h, adj))
        h = torch.relu(self.gnn2(h, adj))
        return self.mlp(h)


# ============================================================
# Feature builders (same as current exp2 train/vis setup)
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



def build_node_features(before: np.ndarray, der_base: np.ndarray, node_idx: int) -> np.ndarray:
    num_nodes = before.shape[0]
    grasp_flag = np.zeros((num_nodes, 1), dtype=np.float32)
    grasp_flag[node_idx, 0] = 1.0
    node_idx_norm = (np.arange(num_nodes, dtype=np.float32) / max(1, num_nodes - 1)).reshape(-1, 1)
    graph_dist_to_grasp_norm = (
        np.abs(np.arange(num_nodes, dtype=np.float32) - float(node_idx)) / max(1, num_nodes - 1)
    ).reshape(-1, 1)
    grasp_pt = before[node_idx]
    euclid_dist_to_grasp = np.linalg.norm(before - grasp_pt[None, :], axis=1).astype(np.float32).reshape(-1, 1)
    node_tangent = compute_node_tangents(before)
    node_curvature_abs = compute_node_curvature_abs(before).reshape(-1, 1)
    x = np.concatenate(
        [
            before.astype(np.float32),
            der_base.astype(np.float32),
            grasp_flag,
            node_idx_norm,
            graph_dist_to_grasp_norm,
            euclid_dist_to_grasp,
            node_tangent.astype(np.float32),
            node_curvature_abs.astype(np.float32),
        ],
        axis=1,
    )
    return x.astype(np.float32)



def build_action_features(
    num_nodes: int,
    node_idx: int,
    grasp_xy: np.ndarray,
    target_xy: np.ndarray,
    move_norm: float,
) -> np.ndarray:
    delta = target_xy - grasp_xy
    return np.array(
        [
            node_idx / max(1, num_nodes - 1),
            grasp_xy[0],
            grasp_xy[1],
            target_xy[0],
            target_xy[1],
            delta[0],
            delta[1],
            move_norm,
        ],
        dtype=np.float32,
    )


# ============================================================
# Loading / metrics
# ============================================================


def load_residual_model(model_path: str, device: torch.device) -> nn.Module:
    obj = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(obj, nn.Module):
        model = obj
    else:
        hidden_dim = obj.get("hidden_dim", 1024)
        expected_nf = obj.get("node_feature_dim")
        expected_af = obj.get("action_dim")
        if expected_nf not in (None, NODE_FEATURE_DIM):
            raise ValueError(f"Checkpoint node_feature_dim={expected_nf}, expected {NODE_FEATURE_DIM}")
        if expected_af not in (None, ACTION_DIM):
            raise ValueError(f"Checkpoint action_dim={expected_af}, expected {ACTION_DIM}")
        model = GraphDLO(hidden_dim=hidden_dim)
        model.load_state_dict(obj["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model



def mean_node_error_mm(points_a: np.ndarray, points_b: np.ndarray, scale_m: float = WORKSPACE_SCALE_M) -> float:
    d = np.linalg.norm(points_a - points_b, axis=1)
    return float(np.mean(d) * scale_m * 1000.0)



def rmse_mm(points_a: np.ndarray, points_b: np.ndarray, scale_m: float = WORKSPACE_SCALE_M) -> float:
    return float(np.sqrt(np.mean((points_a - points_b) ** 2)) * scale_m * 1000.0)


# ============================================================
# Hybrid predictor: DER + GNN residual
# ============================================================


def run_der_rollout(
    sample: dict,
    kb_opt: float,
    damping_opt: float,
    stick_speed: float = 0.02,
    hole_anchor: Tuple[float, float] = (0.5, -0.09),
    total_length: float = 0.8333,
    dt: float = 0.01,
    wait_steps: int = 30,
) -> np.ndarray:
    state_before = np.array(sample["state_before_xy_ext_full"], dtype=np.float32)
    grasp_xy = np.array(sample["grasp_xy"], dtype=np.float32)
    target_xy = np.array(sample["target_xy"], dtype=np.float32)
    move_norm = float(sample["move_norm"])
    robot_speed_norm = 0.0933 / 0.15
    drag_time = move_norm / robot_speed_norm
    drag_steps = max(1, int(round(drag_time / dt)))

    rope = TabletopSoftRope(
        num_nodes=state_before.shape[0],
        total_length=total_length,
        bending_stiffness=kb_opt,
        damping=damping_opt,
        stick_speed=stick_speed,
        hole_anchor=hole_anchor,
        dt=dt,
    )
    rope.x = np.copy(state_before)
    rope.v = np.zeros_like(rope.x)
    rope.set_grip_by_index_and_center(int(sample["node_idx"]), grasp_xy)

    for t in range(1, drag_steps + 1):
        alpha = t / drag_steps
        current_center = (1.0 - alpha) * grasp_xy + alpha * target_xy
        rope.move_grip_center(current_center, project=True)
        rope.step()

    rope.release_grip()
    for _ in range(wait_steps):
        rope.step()

    return rope.x.copy().astype(np.float32)



def gnn_residual_predict(
    model: nn.Module,
    before_state: np.ndarray,
    after_der: np.ndarray,
    node_idx: int,
    grasp_xy: np.ndarray,
    target_xy: np.ndarray,
    move_norm: float,
    adj: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    num_nodes = before_state.shape[0]
    x_input = build_node_features(before=before_state, der_base=after_der, node_idx=node_idx)
    action = build_action_features(
        num_nodes=num_nodes,
        node_idx=node_idx,
        grasp_xy=grasp_xy,
        target_xy=target_xy,
        move_norm=move_norm,
    )
    x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(device)
    action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
    adj = adj.to(device)
    with torch.no_grad():
        pred_res = model(x_tensor, action_tensor, adj)
    return pred_res.squeeze(0).cpu().numpy()



def hybrid_predict_next_shape(
    model: nn.Module,
    current_points: np.ndarray,
    node_idx: int,
    target_xy: np.ndarray,
    kb_opt: float,
    damping_opt: float,
    adj: torch.Tensor,
    device: torch.device,
    total_length: float = 0.8333,
    dt: float = 0.01,
    stick_speed: float = 0.02,
    hole_anchor: Tuple[float, float] = (0.5, -0.09),
    wait_steps: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grasp_xy = current_points[node_idx].astype(np.float32)
    target_xy = np.asarray(target_xy, dtype=np.float32)
    move_norm = float(np.linalg.norm(target_xy - grasp_xy))

    sample = {
        "state_before_xy_ext_full": current_points.astype(np.float32),
        "grasp_xy": grasp_xy,
        "target_xy": target_xy,
        "move_norm": move_norm,
        "node_idx": int(node_idx),
    }

    after_der = run_der_rollout(
        sample=sample,
        kb_opt=kb_opt,
        damping_opt=damping_opt,
        total_length=total_length,
        dt=dt,
        stick_speed=stick_speed,
        hole_anchor=hole_anchor,
        wait_steps=wait_steps,
    )
    pred_res = gnn_residual_predict(
        model=model,
        before_state=current_points.astype(np.float32),
        after_der=after_der.astype(np.float32),
        node_idx=int(node_idx),
        grasp_xy=grasp_xy,
        target_xy=target_xy,
        move_norm=move_norm,
        adj=adj,
        device=device,
    )
    after_final = after_der + pred_res
    return after_final.astype(np.float32), after_der.astype(np.float32), pred_res.astype(np.float32)


# ============================================================
# Data / pair sampling
# ============================================================


def load_all_valid_samples(jsonl_path: str) -> List[dict]:
    valid_samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("valid_sample") is True:
                valid_samples.append(obj)
    if len(valid_samples) == 0:
        raise RuntimeError("valid_sample=True 항목이 없습니다.")
    return valid_samples



def sample_random_pair(
    samples: List[dict],
    rng: random.Random,
    max_trials: int = 200,
    min_initial_mean_err_mm: float = 5.0,
    max_initial_mean_err_mm: float = 200.0,
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    n = len(samples)
    for _ in range(max_trials):
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        current_points = np.asarray(samples[i]["state_before_xy_ext_full"], dtype=np.float32)
        goal_points = np.asarray(samples[j]["state_after_xy_ext_full"], dtype=np.float32)
        err = mean_node_error_mm(current_points, goal_points)
        if min_initial_mean_err_mm <= err <= max_initial_mean_err_mm:
            return i, j, current_points, goal_points

    i = rng.randrange(n)
    j = (i + 1 + rng.randrange(n - 1)) % n
    current_points = np.asarray(samples[i]["state_before_xy_ext_full"], dtype=np.float32)
    goal_points = np.asarray(samples[j]["state_after_xy_ext_full"], dtype=np.float32)
    return i, j, current_points, goal_points


# ============================================================
# Greedy 1-step MPC with cheap pre-filter
# ============================================================

@dataclass
class MPCAction:
    node_idx: int
    length: float
    theta: float
    grasp_xy: np.ndarray
    target_xy: np.ndarray
    raw_target_xy: np.ndarray
    move_norm: float
    feasible_margin: float
    cosine_align: float
    overshoot: float
    cheap_score: float


class GreedyRopeMPC:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        kb_opt: float,
        damping_opt: float,
        num_nodes: int = 20,
        total_length: float = 0.8333,
        dt: float = 0.01,
        stick_speed: float = 0.02,
        hole_anchor: Tuple[float, float] = (0.5, -0.09),
        length_bins: Sequence[float] = (0.03, 0.06, 0.09),
        num_global_dir_bins: int = 40,
        cheap_alpha: float = 2.0,
        cheap_beta: float = 0.5,
        cheap_gamma: float = 0.1,
        shortlist_k: int = 20,
        shape_weight: float = 1.0,
        key_weight: float = 2.0,
        tangent_weight: float = 0.35,
        curvature_weight: float = 0.15,
        keypoint_indices: Optional[Sequence[int]] = None,
        candidate_start_idx: int = 1,
        candidate_step: int = 1,
        prune_opposite_half: bool = True,
        use_projected_target: bool = True,
        dedup_tol: float = 1e-4,
        wait_steps: int = 30,
        topk_default: int = 5,
    ):
        self.model = model
        self.device = device
        self.kb_opt = float(kb_opt)
        self.damping_opt = float(damping_opt)
        self.num_nodes = int(num_nodes)
        self.total_length = float(total_length)
        self.dt = float(dt)
        self.stick_speed = float(stick_speed)
        self.hole_anchor = tuple(hole_anchor)
        self.length_bins = list(length_bins)
        self.num_global_dir_bins = int(num_global_dir_bins)
        self.cheap_alpha = float(cheap_alpha)
        self.cheap_beta = float(cheap_beta)
        self.cheap_gamma = float(cheap_gamma)
        self.shortlist_k = int(shortlist_k)
        self.shape_weight = float(shape_weight)
        self.key_weight = float(key_weight)
        self.tangent_weight = float(tangent_weight)
        self.curvature_weight = float(curvature_weight)
        self.candidate_start_idx = int(candidate_start_idx)
        self.candidate_step = int(candidate_step)
        self.prune_opposite_half = bool(prune_opposite_half)
        self.use_projected_target = bool(use_projected_target)
        self.dedup_tol = float(dedup_tol)
        self.wait_steps = int(wait_steps)
        self.topk_default = int(topk_default)

        if keypoint_indices is None:
            self.keypoint_indices = None
        else:
            self.keypoint_indices = np.array(sorted(set(int(i) for i in keypoint_indices)), dtype=int)

        self.adj = build_chain_adjacency(self.num_nodes)

    def make_rope_from_state(self, current_points: np.ndarray) -> TabletopSoftRope:
        rope = TabletopSoftRope(
            num_nodes=current_points.shape[0],
            total_length=self.total_length,
            bending_stiffness=self.kb_opt,
            damping=self.damping_opt,
            stick_speed=self.stick_speed,
            hole_anchor=self.hole_anchor,
            dt=self.dt,
        )
        rope.x = np.asarray(current_points, dtype=float).copy()
        rope.v = np.zeros_like(rope.x)
        return rope

    def get_candidate_nodes(self, current_points: np.ndarray) -> List[int]:
        n = current_points.shape[0]
        return [i for i in range(1, n)]

    def get_keypoint_indices(self, n: int) -> np.ndarray:
        if self.keypoint_indices is not None:
            return self.keypoint_indices
        idx = sorted(set([0, n // 2, n - 1]))
        return np.array(idx, dtype=int)

    def get_global_direction_bins(self) -> np.ndarray:
        return np.linspace(-np.pi, np.pi, self.num_global_dir_bins, endpoint=False, dtype=np.float32)

    def prune_direction_bins_by_goal(self, theta_bins: np.ndarray, desired_vec: np.ndarray) -> np.ndarray:
        if not self.prune_opposite_half:
            return theta_bins
        nrm = np.linalg.norm(desired_vec)
        if nrm < 1e-8:
            return theta_bins
        u = desired_vec / nrm
        kept = []
        for theta in theta_bins:
            d = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
            if float(np.dot(d, u)) >= 0.0:
                kept.append(theta)
        if len(kept) == 0:
            return theta_bins
        return np.asarray(kept, dtype=np.float32)

    def cosine_align(self, action_vec: np.ndarray, desired_vec: np.ndarray) -> float:
        return float(np.dot(action_vec, desired_vec) / (np.linalg.norm(action_vec) * np.linalg.norm(desired_vec) + EPS))

    def overshoot(self, action_vec: np.ndarray, desired_vec: np.ndarray) -> float:
        return float(max(-0.03, np.linalg.norm(action_vec) - np.linalg.norm(desired_vec)))

    def cheap_score(self, cosine_align: float, overshoot: float, feasible_margin: float) -> float:
        return float(self.cheap_alpha * cosine_align - self.cheap_beta * overshoot + self.cheap_gamma * feasible_margin)

    def _candidate_for_one_node(self, current_points: np.ndarray, goal_points: np.ndarray, node_idx: int) -> List[MPCAction]:
        rope = self.make_rope_from_state(current_points)
        grasp_xy = current_points[node_idx].astype(np.float32)
        rope.set_grip_by_index_and_center(node_idx, grasp_xy)

        theta_bins = self.get_global_direction_bins()
        desired_vec = goal_points[node_idx] - current_points[node_idx]
        theta_bins = self.prune_direction_bins_by_goal(theta_bins, desired_vec)

        actions: List[MPCAction] = []
        seen = set()
        for r in self.length_bins:
            for theta in theta_bins:
                action_vec = np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)
                raw_target = grasp_xy + action_vec

                if self.use_projected_target:
                    target_xy = rope.project_center_to_feasible(raw_target)
                    margin = float(rope.compute_rigid_grip_margin(target_xy))
                else:
                    if not rope.is_rigid_grip_feasible(raw_target):
                        continue
                    target_xy = raw_target
                    margin = float(rope.compute_rigid_grip_margin(target_xy))

                key = tuple(np.round(target_xy / max(self.dedup_tol, 1e-8)).astype(np.int64).tolist())
                if key in seen:
                    continue
                seen.add(key)

                final_action_vec = target_xy.astype(np.float32) - grasp_xy.astype(np.float32)
                move_norm = float(np.linalg.norm(final_action_vec))
                ca = self.cosine_align(final_action_vec, desired_vec)
                oa = self.overshoot(final_action_vec, desired_vec)
                sc = self.cheap_score(ca, oa, margin)

                actions.append(
                    MPCAction(
                        node_idx=int(node_idx),
                        length=float(move_norm),
                        theta=float(np.arctan2(final_action_vec[1], final_action_vec[0] + 0.0)),
                        grasp_xy=grasp_xy.astype(np.float32).copy(),
                        target_xy=np.asarray(target_xy, dtype=np.float32).copy(),
                        raw_target_xy=raw_target.astype(np.float32).copy(),
                        move_norm=move_norm,
                        feasible_margin=margin,
                        cosine_align=ca,
                        overshoot=oa,
                        cheap_score=sc,
                    )
                )
        return actions

    def build_action_candidates(self, current_points: np.ndarray, goal_points: np.ndarray) -> List[MPCAction]:
        actions: List[MPCAction] = []
        for node_idx in self.get_candidate_nodes(current_points):
            node_actions = self._candidate_for_one_node(current_points, goal_points, node_idx)
            actions.extend(node_actions)
        if len(actions) == 0:
            raise RuntimeError("No feasible MPC candidates were generated.")
        return actions

    def shortlist_candidates(self, actions: List[MPCAction], shortlist_k: Optional[int] = None) -> List[MPCAction]:
        k = self.shortlist_k if shortlist_k is None else int(shortlist_k)
        actions_sorted = sorted(actions, key=lambda a: a.cheap_score, reverse=True)
        return actions_sorted[:max(1, min(k, len(actions_sorted)))]

    def shape_distance(self, pred_points: np.ndarray, goal_points: np.ndarray) -> float:
        diff = pred_points - goal_points
        return float(np.mean(np.sum(diff * diff, axis=1)))

    def keypoint_distance(self, pred_points: np.ndarray, goal_points: np.ndarray) -> float:
        key_idx = self.get_keypoint_indices(len(pred_points))
        diff = pred_points[key_idx] - goal_points[key_idx]
        return float(np.mean(np.sum(diff * diff, axis=1)))

    def tangent_distance(self, pred_points: np.ndarray, goal_points: np.ndarray) -> float:
        t_pred = compute_node_tangents(pred_points)
        t_goal = compute_node_tangents(goal_points)
        cos = np.sum(t_pred * t_goal, axis=1)
        cos = np.clip(cos, -1.0, 1.0)
        return float(np.mean(1.0 - cos))

    def curvature_distance(self, pred_points: np.ndarray, goal_points: np.ndarray) -> float:
        k_pred = compute_node_curvature_abs(pred_points)
        k_goal = compute_node_curvature_abs(goal_points)
        diff = k_pred - k_goal
        return float(np.mean(diff * diff))

    def total_cost(self, pred_points: np.ndarray, goal_points: np.ndarray) -> Dict[str, float]:
        d_shape = self.shape_distance(pred_points, goal_points)
        d_key = self.keypoint_distance(pred_points, goal_points)
        d_tan = self.tangent_distance(pred_points, goal_points)
        d_curv = self.curvature_distance(pred_points, goal_points)

        total = (
            self.shape_weight * d_shape
            + self.key_weight * d_key
            + self.tangent_weight * d_tan
            + self.curvature_weight * d_curv
        )
        return {
            "total": float(total),
            "shape": float(d_shape),
            "key": float(d_key),
            "tangent": float(d_tan),
            "curvature": float(d_curv),
        }

    def evaluate_action(self, current_points: np.ndarray, goal_points: np.ndarray, action: MPCAction) -> Dict[str, object]:
        pred_next, after_der, pred_res = hybrid_predict_next_shape(
            model=self.model,
            current_points=current_points,
            node_idx=action.node_idx,
            target_xy=action.target_xy,
            kb_opt=self.kb_opt,
            damping_opt=self.damping_opt,
            adj=self.adj,
            device=self.device,
            total_length=self.total_length,
            dt=self.dt,
            stick_speed=self.stick_speed,
            hole_anchor=self.hole_anchor,
            wait_steps=self.wait_steps,
        )
        cost_info = self.total_cost(pred_next, goal_points)
        return {
            "action": action,
            "pred_next": pred_next,
            "after_der": after_der,
            "pred_res": pred_res,
            "cost": cost_info,
            "mean_err_mm": mean_node_error_mm(pred_next, goal_points),
            "rmse_mm": rmse_mm(pred_next, goal_points),
        }

    def select_action(
        self,
        current_points: np.ndarray,
        goal_points: np.ndarray,
        topk: Optional[int] = None,
        shortlist_k: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, object]:
        current_points = np.asarray(current_points, dtype=np.float32)
        goal_points = np.asarray(goal_points, dtype=np.float32)
        topk = self.topk_default if topk is None else int(topk)
        shortlist_k = self.shortlist_k if shortlist_k is None else int(shortlist_k)

        candidates = self.build_action_candidates(current_points, goal_points)
        if verbose:
            print(f"[MPC] total feasible candidates: {len(candidates)}", flush=True)

        shortlist = self.shortlist_candidates(candidates, shortlist_k=shortlist_k)
        if verbose:
            print(f"[MPC] shortlist by cheap score: {len(shortlist)} / {len(candidates)}", flush=True)
            print("[MPC] cheap top summary:")
            for rank, a in enumerate(shortlist[:min(10, len(shortlist))], start=1):
                print(
                    f"  #{rank} | node={a.node_idx:2d} | len={a.length:.4f} | "
                    f"theta_deg={np.rad2deg(a.theta):7.2f} | cheap={a.cheap_score:.4f} | "
                    f"cos={a.cosine_align:.4f} | over={a.overshoot:.4f} | margin={a.feasible_margin:.4f}"
                )

        evaluated: List[Dict[str, object]] = []
        for idx, action in enumerate(shortlist, start=1):
            item = self.evaluate_action(current_points, goal_points, action)
            evaluated.append(item)
            if verbose and (idx % 5 == 0 or idx == len(shortlist)):
                print(f"[MPC]   expensive eval {idx:3d}/{len(shortlist)} done", flush=True)

        evaluated.sort(key=lambda d: d["cost"]["total"])
        ranked = evaluated[:max(1, min(topk, len(evaluated)))]

        if verbose:
            print("[MPC] final top-k summary:")
            for rank, item in enumerate(ranked, start=1):
                a = item["action"]
                print(
                    f"  #{rank} | node={a.node_idx:2d} | len={a.length:.4f} | "
                    f"theta_deg={np.rad2deg(a.theta):7.2f} | cost={item['cost']['total']:.6f} | "
                    f"shape={item['cost']['shape']:.5f} | key={item['cost']['key']:.5f} | "
                    f"tan={item['cost']['tangent']:.5f} | curv={item['cost']['curvature']:.5f} | "
                    f"mean_err_mm={item['mean_err_mm']:.3f} | rmse_mm={item['rmse_mm']:.3f}"
                )

        return {
            "all_candidates": candidates,
            "shortlist": shortlist,
            "ranked": ranked,
            "best": ranked[0],
        }

    def rollout_to_goal(
        self,
        initial_points: np.ndarray,
        goal_points: np.ndarray,
        topk: Optional[int] = None,
        shortlist_k: Optional[int] = None,
        max_steps: int = 30,
        stop_mean_error_mm: float = 3.0,
        verbose: bool = True,
    ) -> Dict[str, object]:
        current = np.asarray(initial_points, dtype=np.float32).copy()
        goal = np.asarray(goal_points, dtype=np.float32).copy()
        history: List[Dict[str, object]] = []
        topk = self.topk_default if topk is None else int(topk)
        shortlist_k = self.shortlist_k if shortlist_k is None else int(shortlist_k)

        init_mean_mm = mean_node_error_mm(current, goal)
        init_rmse_mm = rmse_mm(current, goal)
        if verbose:
            print(f"[PLAN] start | mean_err_mm={init_mean_mm:.3f} | rmse_mm={init_rmse_mm:.3f}")

        success = False
        for step_idx in range(1, max_steps + 1):
            sel = self.select_action(current, goal, topk=topk, shortlist_k=shortlist_k, verbose=verbose)
            best = sel["best"]
            next_state = best["pred_next"].copy()

            mean_mm = mean_node_error_mm(next_state, goal)
            rmse_now_mm = rmse_mm(next_state, goal)

            step_record = {
                "step_idx": step_idx,
                "before": current.copy(),
                "goal": goal.copy(),
                "after_der": best["after_der"].copy(),
                "after_pred": next_state.copy(),
                "selection": sel,
                "best": best,
                "mean_err_mm": mean_mm,
                "rmse_mm": rmse_now_mm,
            }
            history.append(step_record)

            if verbose:
                a = best["action"]
                print(
                    f"[PLAN] step={step_idx:02d} | choose node={a.node_idx:2d} | len={a.length:.4f} | "
                    f"theta_deg={np.rad2deg(a.theta):7.2f} | mean_err_mm={mean_mm:.3f} | rmse_mm={rmse_now_mm:.3f}",
                    flush=True,
                )

            current = next_state
            if mean_mm <= stop_mean_error_mm:
                success = True
                if verbose:
                    print(f"[PLAN] stop: mean node error <= {stop_mean_error_mm:.3f} mm")
                break

        if not success and verbose:
            print(f"[PLAN] stop: reached max_steps={max_steps}")

        return {
            "success": success,
            "history": history,
            "initial_points": np.asarray(initial_points, dtype=np.float32),
            "final_points": current.copy(),
            "goal_points": goal.copy(),
            "final_mean_err_mm": mean_node_error_mm(current, goal),
            "final_rmse_mm": rmse_mm(current, goal),
        }


# ============================================================
# Visualization
# ============================================================


def visualize_rollout(result: Dict[str, object], show_all_steps: bool = True) -> None:
    history = result["history"]
    initial_points = result["initial_points"]
    goal_points = result["goal_points"]
    final_points = result["final_points"]

    if len(history) == 0:
        raise RuntimeError("History is empty.")

    mean_errs = [mean_node_error_mm(initial_points, goal_points)] + [step["mean_err_mm"] for step in history]
    rmses = [rmse_mm(initial_points, goal_points)] + [step["rmse_mm"] for step in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    ax.plot(initial_points[:, 0], initial_points[:, 1], 'o--', color='lightgray', label='Initial')
    ax.plot(goal_points[:, 0], goal_points[:, 1], 'o-', color='royalblue', linewidth=2, label='Goal')
    ax.plot(final_points[:, 0], final_points[:, 1], 'x-', color='crimson', linewidth=2, label='Final')

    if show_all_steps:
        for step in history:
            p = step["after_pred"]
            ax.plot(p[:, 0], p[:, 1], '-', color='tomato', alpha=0.18)

    for step in history:
        a = step["best"]["action"]
        ax.annotate(
            '', xy=a.target_xy, xytext=a.grasp_xy,
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5, alpha=0.65)
        )

    ax.set_title(
        f"Rollout result\n"
        f"success={result['success']} | final mean={result['final_mean_err_mm']:.2f} mm | final rmse={result['final_rmse_mm']:.2f} mm"
    )
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':')
    ax.legend(loc='lower right')

    ax = axes[1]
    ax.plot(range(len(mean_errs)), mean_errs, marker='o', label='Mean node error (mm)')
    ax.axhline(3.0, color='red', linestyle='--', label='3 mm threshold')
    ax.set_xlabel('planning step')
    ax.set_ylabel('mm')
    ax.set_title('Mean node error over steps')
    ax.grid(True, linestyle=':')
    ax.legend()

    ax = axes[2]
    ax.plot(range(len(rmses)), rmses, marker='o', label='RMSE (mm)')
    ax.set_xlabel('planning step')
    ax.set_ylabel('mm')
    ax.set_title('RMSE over steps')
    ax.grid(True, linestyle=':')
    ax.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# Compute-only wrapper (merged; notebook-compatible)
# ============================================================

EXPECTED_NUM_NODES = 20
DEFAULT_GRASPABLE_INDEX_MIN = 3
DEFAULT_GRASPABLE_INDEX_MAX = 19
ACTION_X_MIN, ACTION_X_MAX = 0.10, 0.84
ACTION_Y_MIN, ACTION_Y_MAX = 0.00, 1.00


class RobotRangeGreedyRopeMPC(GreedyRopeMPC):
    """
    compute-only planner
    - robot I/O 없음
    - eval_object / eval_target 없음
    - 현재 state, goal state를 입력받아 best action만 계산
    - 실제 로봇 action workspace box도 planner 단계에서 반영
    """

    def __init__(
        self,
        *args,
        graspable_index_min: int = DEFAULT_GRASPABLE_INDEX_MIN,
        graspable_index_max: int = DEFAULT_GRASPABLE_INDEX_MAX,
        action_x_min: float = ACTION_X_MIN,
        action_x_max: float = ACTION_X_MAX,
        action_y_min: float = ACTION_Y_MIN,
        action_y_max: float = ACTION_Y_MAX,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.graspable_index_min = int(graspable_index_min)
        self.graspable_index_max = int(graspable_index_max)

        self.action_x_min = float(action_x_min)
        self.action_x_max = float(action_x_max)
        self.action_y_min = float(action_y_min)
        self.action_y_max = float(action_y_max)

    def get_candidate_nodes(self, current_points: np.ndarray):
        n = current_points.shape[0]
        lo = max(1, self.graspable_index_min)
        hi = min(n - 1, self.graspable_index_max)
        return [i for i in range(lo, hi + 1)]

    def clip_target_xy(self, xy: np.ndarray) -> np.ndarray:
        x = float(np.clip(xy[0], self.action_x_min, self.action_x_max))
        y = float(np.clip(xy[1], self.action_y_min, self.action_y_max))
        return np.array([x, y], dtype=np.float32)

    def _candidate_for_one_node(
        self,
        current_points: np.ndarray,
        goal_points: np.ndarray,
        node_idx: int,
    ):
        rope = self.make_rope_from_state(current_points)
        grasp_xy = current_points[node_idx].astype(np.float32)
        rope.set_grip_by_index_and_center(node_idx, grasp_xy)

        theta_bins = self.get_global_direction_bins()
        desired_vec = goal_points[node_idx] - current_points[node_idx]
        theta_bins = self.prune_direction_bins_by_goal(theta_bins, desired_vec)

        actions = []
        seen = set()

        for r in self.length_bins:
            for theta in theta_bins:
                action_vec = np.array(
                    [r * np.cos(theta), r * np.sin(theta)],
                    dtype=np.float32,
                )
                raw_target = grasp_xy + action_vec

                # 1) rope feasibility projection
                if self.use_projected_target:
                    target_xy = rope.project_center_to_feasible(raw_target)
                else:
                    if not rope.is_rigid_grip_feasible(raw_target):
                        continue
                    target_xy = raw_target

                # 2) robot workspace clipping
                target_xy = self.clip_target_xy(target_xy)

                # 3) clipped target feasibility re-check only when strict mode is used
                if (not self.use_projected_target) and (not rope.is_rigid_grip_feasible(target_xy)):
                    continue

                margin = float(rope.compute_rigid_grip_margin(target_xy))

                key = tuple(
                    np.round(target_xy / max(self.dedup_tol, 1e-8))
                    .astype(np.int64)
                    .tolist()
                )
                if key in seen:
                    continue
                seen.add(key)

                final_action_vec = target_xy.astype(np.float32) - grasp_xy.astype(np.float32)
                move_norm = float(np.linalg.norm(final_action_vec))
                ca = self.cosine_align(final_action_vec, desired_vec)
                oa = self.overshoot(final_action_vec, desired_vec)
                sc = self.cheap_score(ca, oa, margin)

                actions.append(
                    MPCAction(
                        node_idx=int(node_idx),
                        length=float(move_norm),
                        theta=float(np.arctan2(final_action_vec[1], final_action_vec[0])),
                        grasp_xy=grasp_xy.astype(np.float32).copy(),
                        target_xy=np.asarray(target_xy, dtype=np.float32).copy(),
                        raw_target_xy=raw_target.astype(np.float32).copy(),
                        move_norm=move_norm,
                        feasible_margin=margin,
                        cosine_align=ca,
                        overshoot=oa,
                        cheap_score=sc,
                    )
                )

        return actions


def build_compute_only_planner(
    model_path: str,
    kb_opt: float = 3.225057e-05,
    damping_opt: float = 0.7077,
    num_nodes: int = EXPECTED_NUM_NODES,
    shortlist_k: int = 20,
    topk: int = 5,
    num_global_dir_bins: int = 40,
    graspable_index_min: int = DEFAULT_GRASPABLE_INDEX_MIN,
    graspable_index_max: int = DEFAULT_GRASPABLE_INDEX_MAX,
    action_x_min: float = ACTION_X_MIN,
    action_x_max: float = ACTION_X_MAX,
    action_y_min: float = ACTION_Y_MIN,
    action_y_max: float = ACTION_Y_MAX,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_residual_model(model_path, device)

    planner = RobotRangeGreedyRopeMPC(
        model=model,
        device=device,
        kb_opt=kb_opt,
        damping_opt=damping_opt,
        num_nodes=num_nodes,
        total_length=0.8333,
        dt=0.01,
        stick_speed=0.02,
        hole_anchor=(0.5, -0.09),
        length_bins=(0.07, 0.15, 0.28, 0.35),
        num_global_dir_bins=num_global_dir_bins,
        cheap_alpha=2.0,
        cheap_beta=0.35,
        cheap_gamma=0.1,
        shortlist_k=shortlist_k,
        shape_weight=1.0,
        key_weight=2.0,
        tangent_weight=0.35,
        curvature_weight=0.15,
        keypoint_indices=None,
        candidate_start_idx=1,
        candidate_step=1,
        prune_opposite_half=True,
        use_projected_target=True,
        topk_default=topk,
        graspable_index_min=graspable_index_min,
        graspable_index_max=graspable_index_max,
        action_x_min=action_x_min,
        action_x_max=action_x_max,
        action_y_min=action_y_min,
        action_y_max=action_y_max,
    )
    return planner



def compute_mpc_action(
    planner: RobotRangeGreedyRopeMPC,
    current_points: np.ndarray,
    goal_points: np.ndarray,
    topk: Optional[int] = None,
    shortlist_k: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    current_points = np.asarray(current_points, dtype=np.float32)
    goal_points = np.asarray(goal_points, dtype=np.float32)

    if current_points.ndim != 2 or current_points.shape[1] != 2:
        raise ValueError(f"current_points shape must be [N,2], got {current_points.shape}")
    if goal_points.ndim != 2 or goal_points.shape[1] != 2:
        raise ValueError(f"goal_points shape must be [N,2], got {goal_points.shape}")
    if current_points.shape != goal_points.shape:
        raise ValueError(
            f"current_points shape {current_points.shape} != goal_points shape {goal_points.shape}"
        )

    sel = planner.select_action(
        current_points=current_points,
        goal_points=goal_points,
        topk=topk,
        shortlist_k=shortlist_k,
        verbose=verbose,
    )
    best = sel["best"]
    action = best["action"]

    return {
        "node_idx": int(action.node_idx),
        "grasp_xy": np.asarray(action.grasp_xy, dtype=np.float32).copy(),
        "target_xy": np.asarray(action.target_xy, dtype=np.float32).copy(),
        "predicted_next": np.asarray(best["pred_next"], dtype=np.float32).copy(),
        "after_der": np.asarray(best["after_der"], dtype=np.float32).copy(),
        "pred_res": np.asarray(best["pred_res"], dtype=np.float32).copy(),
        "predicted_mean_err_mm": float(mean_node_error_mm(best["pred_next"], goal_points)),
        "predicted_rmse_mm": float(rmse_mm(best["pred_next"], goal_points)),
        "cost": dict(best["cost"]),
        "selection": sel,
        "best": best,
    }



def compute_closed_loop_plan_only(
    planner: RobotRangeGreedyRopeMPC,
    initial_points: np.ndarray,
    goal_points: np.ndarray,
    max_steps: int = 10,
    stop_mean_error_mm: float = 4.0,
    topk: Optional[int] = None,
    shortlist_k: Optional[int] = None,
    verbose: bool = True,
):
    current = np.asarray(initial_points, dtype=np.float32).copy()
    goal = np.asarray(goal_points, dtype=np.float32).copy()
    history = []

    for step_idx in range(1, max_steps + 1):
        out = compute_mpc_action(
            planner=planner,
            current_points=current,
            goal_points=goal,
            topk=topk,
            shortlist_k=shortlist_k,
            verbose=verbose,
        )
        history.append({
            "step_idx": int(step_idx),
            "before": current.copy(),
            "node_idx": int(out["node_idx"]),
            "target_xy": np.asarray(out["target_xy"], dtype=np.float32).copy(),
            "predicted_next": np.asarray(out["predicted_next"], dtype=np.float32).copy(),
            "predicted_mean_err_mm": float(out["predicted_mean_err_mm"]),
        })
        current = np.asarray(out["predicted_next"], dtype=np.float32).copy()

        if out["predicted_mean_err_mm"] <= stop_mean_error_mm:
            break

    return {
        "history": history,
        "final_points": current.copy(),
        "final_mean_err_mm": float(mean_node_error_mm(current, goal)),
        "final_rmse_mm": float(rmse_mm(current, goal)),
    }


__all__ = [
    "RobotRangeGreedyRopeMPC",
    "build_compute_only_planner",
    "compute_mpc_action",
    "compute_closed_loop_plan_only",
    "mean_node_error_mm",
    "rmse_mm",
]
