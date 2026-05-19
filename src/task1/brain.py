from typing import Dict, List

import numpy as np

from task1 import config
from geometry_utils import (
    clip_polygon_to_aabb,
    clip_segment_to_aabb,
    distance_point_to_polygon,
    get_global_corners,
    make_circle_polygon,
    shape_alignment_error,
    wrap_angle,
)
from motion_planner import XYMotionPlanner
from perf_utils import perf_timer
from total_ import get_circle_normal, simulate_push_stroke_normalized


class PushingBrain:
    def __init__(self):
        self.com_belief = {
            shape_name: list(shape_data["cases"].keys())[0]
            for shape_name, shape_data in config.SHAPE_DB.items()
        }

        self.vote_box = {
            shape_name: {k: 0 for k in shape_data["cases"].keys()}
            for shape_name, shape_data in config.SHAPE_DB.items()
        }

        self.motion_planner = XYMotionPlanner(
            workspace_min=config.WORKSPACE_MIN,
            workspace_max=config.WORKSPACE_MAX,
            grid_resolution=config.PATH_GRID_RES,
            obstacle_margin=config.PUSHER_RADIUS + config.PATH_OBSTACLE_MARGIN,
        )

        self._last_plan_failure_reason = None
        self._last_progress_debug_signature = None
        self._last_top_feasible_debug_signature = None
        self._last_no_plan_debug_signature = None

    @staticmethod
    def _debug_log(message):
        if config.DEBUG_BRAIN:
            print(f"[BRAIN] {message}")

    def _workspace_rect(self):
        return (
            np.asarray(config.WORKSPACE_MIN, dtype=float),
            np.asarray(config.WORKSPACE_MAX, dtype=float),
        )

    def _clip_polygon_to_workspace(self, polygon):
        ws_min, ws_max = self._workspace_rect()
        return clip_polygon_to_aabb(polygon, ws_min, ws_max)

    def _visible_polygon_face_segments(self, polygon_points):
        ws_min, ws_max = self._workspace_rect()
        segments = []

        n = len(polygon_points)
        for face_idx in range(n):
            p1 = np.asarray(polygon_points[face_idx], dtype=float)
            p2 = np.asarray(polygon_points[(face_idx + 1) % n], dtype=float)

            edge_vec = p2 - p1
            edge_len = np.linalg.norm(edge_vec)
            if edge_len < 1e-8:
                continue

            clipped = clip_segment_to_aabb(p1, p2, ws_min, ws_max)
            if clipped is None:
                continue

            c1 = np.asarray(clipped[0], dtype=float)
            c2 = np.asarray(clipped[1], dtype=float)
            if np.linalg.norm(c2 - c1) < 1e-8:
                continue

            t_hat = edge_vec / edge_len
            if np.dot(c2 - c1, t_hat) < 0.0:
                c1, c2 = c2, c1

            n_hat = np.array([t_hat[1], -t_hat[0]], dtype=float)

            segments.append({
                "face_idx": face_idx,
                "p1": c1,
                "p2": c2,
                "t_hat": t_hat,
                "n_hat": n_hat,
            })

        return segments

    def _face_contact_is_in_workspace(self, start_xy, n_hat):
        start_xy = np.asarray(start_xy, dtype=float)
        n_hat = np.asarray(n_hat, dtype=float)

        if not self.motion_planner._inside_bounds(start_xy):
            return False

        approach_xy = start_xy + self.motion_planner.obstacle_margin * 1.0 * n_hat
        return self.motion_planner._inside_bounds(approach_xy)

    def _candidate_motion_is_in_workspace(self, start_xy, end_xy, n_hat, retreat_distance):
        start_xy = np.asarray(start_xy, dtype=float)
        end_xy = np.asarray(end_xy, dtype=float)
        n_hat = np.asarray(n_hat, dtype=float)

        approach_xy = start_xy + self.motion_planner.obstacle_margin * 1.0 * n_hat
        retreat_xy = end_xy + retreat_distance * n_hat

        return (
            self.motion_planner._inside_bounds(start_xy)
            and self.motion_planner._inside_bounds(end_xy)
            and self.motion_planner._inside_bounds(approach_xy)
            and self.motion_planner._inside_bounds(retreat_xy)
        )

    def _report_plan_failure(self, reason):
        if self._last_plan_failure_reason == reason:
            return

        self._last_plan_failure_reason = reason
        reason_messages = {
            "no_feasible_candidates": "plan failed: no feasible candidates",
            "progress_cutoff": "plan failed: all feasible candidates rejected by progress cutoff",
            "score_cutoff": "plan failed: all feasible candidates rejected by score cutoff",
        }
        print(reason_messages[reason])

    def _report_no_plan_context(self, current_local_err, global_shape_err, lookahead, local_target):
        signature = (
            round(float(current_local_err), 6),
            round(float(global_shape_err), 6),
            int(lookahead),
            tuple(round(float(value), 6) for value in local_target),
        )
        if self._last_no_plan_debug_signature == signature:
            return

        self._last_no_plan_debug_signature = signature
        print(
            "[BRAIN] No Plan context: "
            f"current_local_err={float(current_local_err):.5f}, "
            f"global_shape_err={float(global_shape_err):.5f}, "
            f"lookahead={int(lookahead)}, "
            f"local_target={self._format_vector(local_target)}"
        )

    def _report_progress_cutoff_debug(self, feasible_items, current_local_err):
        if not config.DEBUG_BRAIN_PROGRESS or not feasible_items:
            return

        progress_values = sorted(
            (float(item["progress"]) for item in feasible_items),
            reverse=True,
        )
        top_progress = progress_values[:5]
        best_next_local_err = min(float(item["next_shape_err"]) for item in feasible_items)

        signature = (
            tuple(round(value, 6) for value in top_progress),
            round(float(current_local_err), 6),
            round(best_next_local_err, 6),
        )
        if self._last_progress_debug_signature == signature:
            return

        self._last_progress_debug_signature = signature
        formatted_progress = ", ".join(f"{value:.5f}" for value in top_progress)
        print(f"[BRAIN] top feasible progress values: [{formatted_progress}]")
        print(
            "[BRAIN] "
            f"current_local_err={float(current_local_err):.4f}, "
            f"best_next_local_err={best_next_local_err:.4f}"
        )

    @staticmethod
    def _format_vector(values, precision=5):
        return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"

    def _report_top_feasible_candidates_debug(
        self,
        feasible_items,
        current_pose,
        local_target,
        current_local_err,
    ):
        if not config.DEBUG_BRAIN_TOP_CANDIDATES or not feasible_items:
            return

        top_items = sorted(
            feasible_items,
            key=lambda item: float(item["next_shape_err"]),
        )[:3]

        signature = tuple(
            (
                int(item["candidate"]["index"]),
                round(float(item["next_shape_err"]), 6),
                round(float(item["progress"]), 6),
                round(float(item["candidate"]["stroke_len"]), 6),
            )
            for item in top_items
        )
        if self._last_top_feasible_debug_signature == signature:
            return

        self._last_top_feasible_debug_signature = signature
        for rank, item in enumerate(top_items, start=1):
            cand = item["candidate"]
            print(
                f"[BRAIN][TOP{rank}] "
                f"face={cand['face_idx']}, action={cand['action_type']}, ratio={cand['ratio']:.3f}"
            )
            print(
                "  "
                f"start={self._format_vector(cand['start'][:2])}, "
                f"end={self._format_vector(cand['end'][:2])}"
            )
            print(f"  current_pose={self._format_vector(current_pose)}")
            print(f"  next_pose={self._format_vector(item['next_pose'])}")
            print(f"  local_target={self._format_vector(local_target)}")
            print(
                "  "
                f"progress={float(item['progress']):.5f}, "
                f"align={float(item['align']):.5f}, "
                f"stroke_len={float(cand['stroke_len']):.5f}"
            )
            print(
                "  "
                f"current_local_err={float(current_local_err):.5f}, "
                f"next_local_err={float(item['next_shape_err']):.5f}"
            )

    @staticmethod
    def _min_progress_cutoff():
        return 0.0

    @staticmethod
    def _negative_score_cutoff():
        return -1e9

    def build_reference_path(self, current_pose, target_pose, steps=config.REF_PATH_STEPS):
        x0, y0, th0 = current_pose
        xf, yf, thf = target_pose
        path = []

        for i in range(steps):
            t = i / max(1, steps - 1)
            s = 10 * (t ** 3) - 15 * (t ** 4) + 6 * (t ** 5)
            d_th = wrap_angle(thf - th0)

            path.append([
                float(x0 + (xf - x0) * s),
                float(y0 + (yf - y0) * s),
                float(wrap_angle(th0 + d_th * s)),
            ])
        return path

    def _choose_lookahead(self, global_shape_err):
        if global_shape_err >= config.REF_PATH_LOOKAHEAD_FAR_ERR_THRESH:
            lookahead = config.REF_PATH_LOOKAHEAD_FAR
        elif global_shape_err >= config.REF_PATH_LOOKAHEAD_MID_ERR_THRESH:
            lookahead = config.REF_PATH_LOOKAHEAD_MID
        elif global_shape_err >= config.REF_PATH_LOOKAHEAD_CLOSE_ERR_THRESH:
            lookahead = config.REF_PATH_LOOKAHEAD_CLOSE
        else:
            lookahead = config.REF_PATH_LOOKAHEAD_MIN

        return int(np.clip(lookahead, config.REF_PATH_LOOKAHEAD_MIN, config.REF_PATH_LOOKAHEAD_MAX))

    @staticmethod
    def _should_retry_with_farther_target(failure_reason, global_shape_err, chosen_lookahead):
        _ = global_shape_err
        return (
            failure_reason == "no_feasible_candidates"
            and chosen_lookahead < config.REF_PATH_LOOKAHEAD_FALLBACK_MAX
        )

    @staticmethod
    def _next_retry_lookahead(chosen_lookahead):
        return min(
            int(chosen_lookahead) + int(config.REF_PATH_LOOKAHEAD_FALLBACK_DELTA),
            int(config.REF_PATH_LOOKAHEAD_FALLBACK_MAX),
        )

    def pick_local_waypoint(self, current_pose, ref_path, lookahead=config.REF_PATH_LOOKAHEAD):
        if not ref_path:
            return list(current_pose)

        cx, cy = current_pose[0], current_pose[1]
        closest_idx = 0
        min_dist = float("inf")

        for idx, pt in enumerate(ref_path):
            dist = (pt[0] - cx) ** 2 + (pt[1] - cy) ** 2
            if dist < min_dist:
                min_dist = dist
                closest_idx = idx

        target_idx = min(closest_idx + lookahead, len(ref_path) - 1)
        return ref_path[target_idx]

    def compute_lateral_error(self, current_pose, ref_path):
        if not ref_path:
            return 0.0
        point = np.asarray(current_pose[:2], dtype=float)
        return min(float(np.linalg.norm(point - np.asarray(pt[:2], dtype=float))) for pt in ref_path)

    def generate_face_candidates(self, shape_type, shape_info, radius, base_stroke):
        with perf_timer("brain.generate_face_candidates"):
            candidates: List[Dict] = []
            if base_stroke <= 1e-10:
                return candidates

            if shape_type == "circle":
                cx, cy = shape_info
                angles = np.linspace(
                    0.0,
                    2.0 * np.pi,
                    config.DIRECT_CIRCLE_CANDIDATE_ANGLES,
                    endpoint=False,
                )
                for face_idx, angle in enumerate(angles):
                    surface_x = cx + radius * np.cos(angle)
                    surface_y = cy + radius * np.sin(angle)

                    n_hat = get_circle_normal(cx, cy, surface_x, surface_y, radius)
                    t_hat = np.array([-n_hat[1], n_hat[0]], dtype=float)

                    start = np.array([
                        surface_x + config.PUSHER_RADIUS * n_hat[0],
                        surface_y + config.PUSHER_RADIUS * n_hat[1],
                    ], dtype=float)

                    if not self._face_contact_is_in_workspace(start, n_hat):
                        continue

                    end_normal = start - base_stroke * n_hat
                    end_spin_pos = end_normal + (base_stroke * config.DIRECT_SPIN_TANGENT_SCALE) * t_hat
                    end_spin_neg = end_normal - (base_stroke * config.DIRECT_SPIN_TANGENT_SCALE) * t_hat

                    candidates.extend([
                        {
                            "face_idx": face_idx,
                            "ratio": 0.5,
                            "start": start.copy(),
                            "end": end_normal,
                            "n_hat": n_hat,
                            "t_hat": t_hat,
                            "stroke_len": float(np.linalg.norm(end_normal - start)),
                            "action_type": "normal",
                        },
                        {
                            "face_idx": face_idx,
                            "ratio": 0.5,
                            "start": start.copy(),
                            "end": end_spin_pos,
                            "n_hat": n_hat,
                            "t_hat": t_hat,
                            "stroke_len": float(np.linalg.norm(end_spin_pos - start)),
                            "action_type": "spin_pos",
                        },
                        {
                            "face_idx": face_idx,
                            "ratio": 0.5,
                            "start": start.copy(),
                            "end": end_spin_neg,
                            "n_hat": n_hat,
                            "t_hat": t_hat,
                            "stroke_len": float(np.linalg.norm(end_spin_neg - start)),
                            "action_type": "spin_neg",
                        },
                    ])
                return candidates

            segments = self._visible_polygon_face_segments(shape_info)

            for seg in segments:
                face_idx = seg["face_idx"]
                p1 = seg["p1"]
                p2 = seg["p2"]

                edge_vec = p2 - p1
                edge_len = np.linalg.norm(edge_vec)
                if edge_len < 1e-8:
                    continue

                t_hat = seg["t_hat"]
                n_hat = seg["n_hat"]

                ratios = config.T_EDGE_RATIOS if shape_type == "t" else config.DIRECT_EDGE_RATIOS

                for ratio in ratios:
                    surface_start = p1 + ratio * edge_vec
                    push_offset = config.PUSHER_RADIUS * n_hat
                    start = surface_start + push_offset

                    if not self._face_contact_is_in_workspace(start, n_hat):
                        continue

                    end_normal = start - base_stroke * n_hat
                    end_spin_pos = end_normal + (base_stroke * config.DIRECT_SPIN_TANGENT_SCALE) * t_hat
                    end_spin_neg = end_normal - (base_stroke * config.DIRECT_SPIN_TANGENT_SCALE) * t_hat

                    candidates.extend([
                        {
                            "face_idx": face_idx,
                            "ratio": float(ratio),
                            "start": start.copy(),
                            "end": end_normal,
                            "n_hat": n_hat,
                            "t_hat": t_hat,
                            "stroke_len": float(np.linalg.norm(end_normal - start)),
                            "action_type": "normal",
                            "segment_p1": p1.copy(),
                            "segment_p2": p2.copy(),
                        },
                        {
                            "face_idx": face_idx,
                            "ratio": float(ratio),
                            "start": start.copy(),
                            "end": end_spin_pos,
                            "n_hat": n_hat,
                            "t_hat": t_hat,
                            "stroke_len": float(np.linalg.norm(end_spin_pos - start)),
                            "action_type": "spin_pos",
                            "segment_p1": p1.copy(),
                            "segment_p2": p2.copy(),
                        },
                        {
                            "face_idx": face_idx,
                            "ratio": float(ratio),
                            "start": start.copy(),
                            "end": end_spin_neg,
                            "n_hat": n_hat,
                            "t_hat": t_hat,
                            "stroke_len": float(np.linalg.norm(end_spin_neg - start)),
                            "action_type": "spin_neg",
                            "segment_p1": p1.copy(),
                            "segment_p2": p2.copy(),
                        },
                    ])
            return candidates

    def _is_quickly_reachable(self, start_xy, n_hat, obstacle_polygon):
        start_xy = np.asarray(start_xy, dtype=float)
        n_hat = np.asarray(n_hat, dtype=float)

        if not self.motion_planner._inside_bounds(start_xy):
            return False

        approach_xy = start_xy + self.motion_planner.obstacle_margin * 0.3 * n_hat
        if not self.motion_planner._inside_bounds(approach_xy):
            return False

        clearance = distance_point_to_polygon(approach_xy, obstacle_polygon)
        return clearance > max(self.motion_planner.obstacle_margin * 0.25, 1e-6)

    def filter_faces_by_dynamics(
        self,
        face_candidates,
        shape_type,
        current_pose,
        local_target,
        local_corners,
        radius,
        robot_xy,
        belief_name,
        obstacle_polygon,
    ):
        with perf_timer("brain.filter_faces_by_dynamics"):
            _ = robot_xy
            scored_faces = []

            for face in face_candidates:
                quickly_reachable = self._is_quickly_reachable(face["start"], face["n_hat"], obstacle_polygon)
                next_pose = self._simulate_case(shape_type, current_pose, face, belief_name)
                err = shape_alignment_error(shape_type, next_pose, local_target, local_corners, radius)

                scored_faces.append({
                    "face_info": face,
                    "error": float(err),
                    "next_pose": next_pose,
                    "quickly_reachable": bool(quickly_reachable),
                })

            scored_faces.sort(
                key=lambda item: (
                    item["error"],
                    0 if item["quickly_reachable"] else 1,
                )
            )
            topk = config.T_FACE_FILTER_TOPK if shape_type == "t" else config.FACE_FILTER_TOPK
            return scored_faces[:topk]

    def generate_candidate_actions(self, filtered_faces, shape_type, shape_info, allowed_strokes, base_stroke):
        with perf_timer("brain.generate_candidate_actions"):
            candidates = []
            action_idx = 0

            if base_stroke <= 1e-10:
                return candidates

            for face_data in filtered_faces:
                face = face_data["face_info"]
                face_idx = face["face_idx"]
                n_hat = face["n_hat"]
                t_hat = face["t_hat"]

                # filter_faces_by_dynamics()에서 살아남은 ratio만 사용
                selected_ratio = float(face.get("ratio", 0.5))

                if shape_type == "circle":
                    ratios = [0.5]

                elif shape_type == "t":
                    p1 = np.array(face.get("segment_p1", shape_info[face_idx]), dtype=float)
                    p2 = np.array(face.get("segment_p2", shape_info[(face_idx + 1) % len(shape_info)]), dtype=float)
                    edge_vec = p2 - p1

                    if np.linalg.norm(edge_vec) < 1e-8:
                        continue

                    # 기존: ratios = config.T_EDGE_RATIOS
                    # 수정: filter에서 선택된 ratio만 사용
                    ratios = [selected_ratio]

                else:
                    p1 = np.array(face.get("segment_p1", shape_info[face_idx]), dtype=float)
                    p2 = np.array(face.get("segment_p2", shape_info[(face_idx + 1) % len(shape_info)]), dtype=float)
                    edge_vec = p2 - p1

                    if np.linalg.norm(edge_vec) < 1e-8:
                        continue

                    # 기존: ratios = config.DIRECT_EDGE_RATIOS
                    # 수정: filter에서 선택된 ratio만 사용
                    ratios = [selected_ratio]

                for ratio in ratios:
                    if shape_type == "circle":
                        start_pt = face["start"].copy()
                    else:
                        surface_start = p1 + ratio * edge_vec
                        start_pt = surface_start + config.PUSHER_RADIUS * n_hat

                    for stroke_mult in allowed_strokes:
                        stroke_len = min(base_stroke * stroke_mult, config.MAX_STROKE_LEN * stroke_mult)
                        end_normal = start_pt - stroke_len * n_hat

                        if self.motion_planner._inside_bounds(start_pt) and self.motion_planner._inside_bounds(end_normal):
                            candidates.append({
                                "index": action_idx,
                                "face_idx": face_idx,
                                "ratio": float(ratio),
                                "start": start_pt.copy(),
                                "end": end_normal,
                                "n_hat": n_hat,
                                "t_hat": t_hat,
                                "stroke_len": float(stroke_len),
                                "action_type": "normal",
                            })
                            action_idx += 1

                        if config.DIRECT_SPIN_TANGENT_SCALE > 0.0:
                            end_spin_pos = end_normal + (stroke_len * config.DIRECT_SPIN_TANGENT_SCALE) * t_hat

                            if self.motion_planner._inside_bounds(start_pt) and self.motion_planner._inside_bounds(end_spin_pos):
                                candidates.append({
                                    "index": action_idx,
                                    "face_idx": face_idx,
                                    "ratio": float(ratio),
                                    "start": start_pt.copy(),
                                    "end": end_spin_pos,
                                    "n_hat": n_hat,
                                    "t_hat": t_hat,
                                    "stroke_len": float(np.linalg.norm(end_spin_pos - start_pt)),
                                    "action_type": "spin_pos",
                                })
                                action_idx += 1

                            end_spin_neg = end_normal - (stroke_len * config.DIRECT_SPIN_TANGENT_SCALE) * t_hat

                            if self.motion_planner._inside_bounds(start_pt) and self.motion_planner._inside_bounds(end_spin_neg):
                                candidates.append({
                                    "index": action_idx,
                                    "face_idx": face_idx,
                                    "ratio": float(ratio),
                                    "start": start_pt.copy(),
                                    "end": end_spin_neg,
                                    "n_hat": n_hat,
                                    "t_hat": t_hat,
                                    "stroke_len": float(np.linalg.norm(end_spin_neg - start_pt)),
                                    "action_type": "spin_neg",
                                })
                                action_idx += 1

            return candidates

    def _shape_polygon(self, shape_type, current_pose, local_corners, radius):
        if shape_type == "circle":
            return make_circle_polygon(current_pose[:2], radius, config.CIRCLE_POLY_POINTS)
        return get_global_corners(current_pose, local_corners)

    def _simulate_case(self, shape_type, current_pose, candidate, case_name):
        shape_data = config.SHAPE_DB[shape_type]
        case_info = shape_data["cases"][case_name]

        _, _, theta_next, next_body_center_norm = simulate_push_stroke_normalized(
            m=case_info["m"],
            i_body=case_info["I"],
            com_x=case_info["com"][0],
            com_y=case_info["com"][1],
            body_center_norm=np.array(current_pose[:2], dtype=float),
            theta=current_pose[2],
            v=np.array([0.0, 0.0]),
            w=0.0,
            start_x_norm=candidate["start"][0],
            start_y_norm=candidate["start"][1],
            end_x_norm=candidate["end"][0],
            end_y_norm=candidate["end"][1],
            n_hat=candidate["n_hat"],
            t_hat=candidate["t_hat"],
            config_dict=config.__dict__,
            eff_r=shape_data["eff_r"],
        )
        return [float(next_body_center_norm[0]), float(next_body_center_norm[1]), float(theta_next)]

    def _goal_vector(self, shape_type, current_pose, target_pose):
        dx = target_pose[0] - current_pose[0]
        dy = target_pose[1] - current_pose[1]
        dtheta = 0.0 if shape_type == "circle" else wrap_angle(target_pose[2] - current_pose[2])

        return np.array([
            config.DIRECT_TRANSLATION_GAIN * dx,
            config.DIRECT_TRANSLATION_GAIN * dy,
            config.DIRECT_ROTATION_GAIN * dtheta,
        ], dtype=float)

    def _motion_vector(self, shape_type, current_pose, next_pose):
        dx = next_pose[0] - current_pose[0]
        dy = next_pose[1] - current_pose[1]
        dtheta = 0.0 if shape_type == "circle" else wrap_angle(next_pose[2] - current_pose[2])

        return np.array([
            config.DIRECT_TRANSLATION_GAIN * dx,
            config.DIRECT_TRANSLATION_GAIN * dy,
            config.DIRECT_ROTATION_GAIN * dtheta,
        ], dtype=float)

    @staticmethod
    def _cosine_score(a, b):
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _stroke_len_from_error(self, shape_type, current_pose, target_pose):
        err = shape_alignment_error(
            shape_type,
            current_pose,
            target_pose,
            config.get_shape_corners(shape_type),
            config.get_shape_radius(shape_type),
        )
        if err <= config.DIRECT_NEAR_GOAL_THRESH:
            return float(max(
                config.MAX_STROKE_LEN * config.DIRECT_NEAR_GOAL_STROKE_SCALE,
                config.DIRECT_MIN_STROKE_LEN,
            ))
        return float(max(config.MAX_STROKE_LEN, config.DIRECT_MIN_STROKE_LEN))

    def _allowed_strokes_from_error(self, shape_type, global_err):
        """
        현재 목표까지의 global_err에 따라 사용할 stroke 배율 목록을 고른다.
        각 값은 절대 길이가 아니라 base_stroke에 곱해지는 multiplier다.
        """
        if shape_type == "t":
            far = config.T_ALLOWED_STROKES_FAR
            mid = config.T_ALLOWED_STROKES_MID
            near = config.T_ALLOWED_STROKES_NEAR
        elif shape_type == "circle":
            far = getattr(config, "CIRCLE_ALLOWED_STROKES_FAR", [1.2, 1.6])
            mid = getattr(config, "CIRCLE_ALLOWED_STROKES_MID", [0.7, 0.9])
            near = getattr(config, "CIRCLE_ALLOWED_STROKES_NEAR", [0.2, 0.5])
        elif shape_type == "square":
            far = getattr(config, "SQUARE_ALLOWED_STROKES_FAR", [1.2, 1.6])
            mid = getattr(config, "SQUARE_ALLOWED_STROKES_MID", [0.7, 0.9])
            near = getattr(config, "SQUARE_ALLOWED_STROKES_NEAR", [0.2, 0.5])
        else:
            far = [1.2, 1.6]
            mid = [0.7, 0.9]
            near = [0.2, 0.5]

        if global_err > config.STROKE_DIST_FAR:
            return list(far)
        if global_err > config.STROKE_DIST_MID:
            return list(mid)
        return list(near)

    def _evaluate_candidate_without_path(
        self,
        shape_type,
        current_pose,
        local_target,
        local_corners,
        radius,
        belief_name,
        candidate,
        current_local_err,
        goal_vec,
    ):
        next_pose = self._simulate_case(shape_type, current_pose, candidate, belief_name)

        next_local_err = shape_alignment_error(
            shape_type,
            next_pose,
            local_target,
            local_corners,
            radius,
        )

        progress = current_local_err - next_local_err
        motion_vec = self._motion_vector(shape_type, current_pose, next_pose)
        align = self._cosine_score(goal_vec, motion_vec)
        cheap_score = align + progress

        return {
            "candidate": candidate,
            "next_pose": next_pose,
            "current_shape_err": float(current_local_err),
            "next_shape_err": float(next_local_err),
            "progress": float(progress),
            "align": float(align),
            "cheap_score": float(cheap_score),
        }

    def score_candidate_actions(
        self,
        candidate_actions,
        shape_type,
        current_pose,
        local_target,
        local_corners,
        radius,
        belief_name,
    ):
        with perf_timer("brain.score_candidate_actions"):
            current_local_err = shape_alignment_error(
                shape_type,
                current_pose,
                local_target,
                local_corners,
                radius,
            )
            goal_vec = self._goal_vector(shape_type, current_pose, local_target)

            scored = []

            for candidate in candidate_actions:
                item = self._evaluate_candidate_without_path(
                    shape_type,
                    current_pose,
                    local_target,
                    local_corners,
                    radius,
                    belief_name,
                    candidate,
                    current_local_err,
                    goal_vec,
                )
                scored.append(item)

            scored.sort(key=lambda item: item["cheap_score"], reverse=True)
            return scored

    def shortlist_candidate_actions(
        self,
        candidate_actions,
        shape_type,
        current_pose,
        local_target,
        local_corners,
        radius,
        belief_name,
    ):
        with perf_timer("brain.shortlist_candidate_actions"):
            scored = self.score_candidate_actions(
                candidate_actions,
                shape_type,
                current_pose,
                local_target,
                local_corners,
                radius,
                belief_name,
            )
            return scored[:config.DIRECT_ACTION_SHORTLIST_TOPK]

    def _attach_feasible_candidates(self, robot_xy, scored_actions, obstacle_polygon, shape_type):
        """
        후보 action들에 대해 접근 경로를 붙인다.

        기존 방식:
            MAX_APPROACH_WAYPOINTS 하나로만 검사
            -> 5개 제한에서 실패하면 긴 경로 후보를 전부 버림

        수정 방식:
            5개 제한으로 먼저 검사
            없으면 8개
            없으면 10개
            없으면 12개
            순서로 점진적으로 완화

        장점:
            - 짧은 경로가 있으면 무조건 짧은 경로를 우선 사용
            - 진짜 막힌 경우에만 긴 A* 경로 허용
        """
        feasible = {}
        attach_fail_count = 0

        waypoint_limits = list(
            getattr(
                config,
                "APPROACH_WAYPOINT_LIMIT_SEQUENCE",
                [getattr(config, "MAX_APPROACH_WAYPOINTS", 5)],
            )
        )

        if len(waypoint_limits) == 0:
            waypoint_limits = [getattr(config, "MAX_APPROACH_WAYPOINTS", 5)]

        original_max_waypoints = getattr(config, "MAX_APPROACH_WAYPOINTS", None)

        total_attempts = 0
        last_fail_count = 0

        try:
            for limit in waypoint_limits:
                config.MAX_APPROACH_WAYPOINTS = int(limit)

                feasible_this_limit = {}
                fail_this_limit = 0
                attempt_this_limit = 0

                for item in scored_actions:
                    candidate = item["candidate"]

                    retreat_distance = (
                        config.T_RETREAT_DISTANCE
                        if shape_type == "t"
                        else config.RETREAT_DISTANCE
                    )

                    if not self._candidate_motion_is_in_workspace(
                        candidate["start"],
                        candidate["end"],
                        candidate["n_hat"],
                        retreat_distance,
                    ):
                        fail_this_limit += 1
                        attempt_this_limit += 1
                        continue

                    motion = self.motion_planner.attach_approach(
                        robot_xy,
                        candidate["start"],
                        candidate["end"],
                        candidate["n_hat"],
                        obstacle_polygon,
                        retreat_distance=retreat_distance,
                    )

                    attempt_this_limit += 1

                    if motion is not None:
                        feasible_this_limit[candidate["index"]] = {
                            "motion": motion,
                            **item,
                        }
                    else:
                        fail_this_limit += 1

                total_attempts += attempt_this_limit
                last_fail_count = fail_this_limit

                # 이 limit에서 feasible이 하나라도 나오면 바로 종료
                # 즉, 5개로 가능하면 8/10/12까지 절대 안 감
                if feasible_this_limit:
                    feasible = feasible_this_limit

                    if getattr(config, "DEBUG_BRAIN", False):
                        print(
                            f"[BRAIN] attach success with "
                            f"MAX_APPROACH_WAYPOINTS={int(limit)}, "
                            f"feasible={len(feasible)}"
                        )

                    return feasible, attempt_this_limit, fail_this_limit

            # 모든 limit에서 실패한 경우
            attach_fail_count = last_fail_count
            return feasible, len(scored_actions), attach_fail_count

        finally:
            # 다른 코드에 영향 안 주도록 원래 값 복구
            if original_max_waypoints is not None:
                config.MAX_APPROACH_WAYPOINTS = original_max_waypoints

    def generate_candidate_bundle(
        self,
        shape_type,
        current_pose,
        target_pose,
        local_corners,
        radius,
        robot_pose=None,
        lookahead_override=None,
    ):
        robot_xy = np.array(robot_pose[:2] if robot_pose is not None else config.ROBOT_START_XY, dtype=float)
        shape_info = current_pose[:2] if shape_type == "circle" else get_global_corners(current_pose, local_corners)

        ref_path = self.build_reference_path(current_pose, target_pose)

        global_err = shape_alignment_error(
            shape_type,
            current_pose,
            target_pose,
            local_corners,
            radius,
        )

        chosen_lookahead = (
            int(lookahead_override)
            if lookahead_override is not None
            else self._choose_lookahead(global_err)
        )

        local_target = self.pick_local_waypoint(current_pose, ref_path, lookahead=chosen_lookahead)

        current_local_err = shape_alignment_error(
            shape_type,
            current_pose,
            local_target,
            local_corners,
            radius,
        )

        full_shape_polygon = self._shape_polygon(shape_type, current_pose, local_corners, radius)
        obstacle_polygon = self._clip_polygon_to_workspace(full_shape_polygon)

        base_stroke = max(
            self._stroke_len_from_error(shape_type, current_pose, target_pose),
            config.DIRECT_MIN_STROKE_LEN,
        )
        if shape_type == "t":
            base_stroke *= config.T_BASE_STROKE_SCALE

        allowed_strokes = self._allowed_strokes_from_error(shape_type, global_err)

        face_candidates = self.generate_face_candidates(
            shape_type,
            shape_info,
            radius,
            base_stroke,
        )

        belief_name = self.com_belief[shape_type]

        filtered_faces = self.filter_faces_by_dynamics(
            face_candidates,
            shape_type,
            current_pose,
            local_target,
            local_corners,
            radius,
            robot_xy,
            belief_name,
            obstacle_polygon,
        )

        candidate_actions = self.generate_candidate_actions(
            filtered_faces,
            shape_type,
            shape_info,
            allowed_strokes,
            base_stroke,
        )

        scored_candidate_actions = self.score_candidate_actions(
            candidate_actions,
            shape_type,
            current_pose,
            local_target,
            local_corners,
            radius,
            belief_name,
        )

        shortlisted_actions = scored_candidate_actions[:config.DIRECT_ACTION_SHORTLIST_TOPK]

        feasible, attach_attempt_count, attach_fail_count = self._attach_feasible_candidates(
            robot_xy,
            shortlisted_actions,
            obstacle_polygon,
            shape_type,
        )

        exhaustive_attach_used = False
        if not feasible and len(scored_candidate_actions) > len(shortlisted_actions):
            exhaustive_attach_used = True
            remaining_actions = scored_candidate_actions[len(shortlisted_actions):]
            fallback_feasible, fallback_attempts, fallback_failures = self._attach_feasible_candidates(
                robot_xy,
                remaining_actions,
                obstacle_polygon,
                shape_type,
            )
            feasible.update(fallback_feasible)
            attach_attempt_count += fallback_attempts
            attach_fail_count += fallback_failures

        return {
            "ref_path": ref_path,
            "local_target": local_target,
            "current_local_err": float(current_local_err),
            "global_shape_err": float(global_err),
            "chosen_lookahead": int(chosen_lookahead),
            "shape_info": shape_info,
            "full_shape_polygon": full_shape_polygon,
            "obstacle_polygon": obstacle_polygon,
            "base_stroke": float(base_stroke),
            "filtered_faces": filtered_faces,
            "allowed_strokes": allowed_strokes,
            "candidate_actions": candidate_actions,
            "scored_candidate_actions": scored_candidate_actions,
            "shortlisted_actions": shortlisted_actions,
            "feasible": feasible,
            "attach_attempt_count": attach_attempt_count,
            "attach_fail_count": attach_fail_count,
            "exhaustive_attach_used": exhaustive_attach_used,
        }

    def get_best_plan(
        self,
        shape_type,
        current_pose,
        target_pose,
        local_corners,
        planner=None,
        radius=None,
        robot_pose=None,
        candidate_data=None,
        allow_farther_retry=True,
    ):
        with perf_timer("brain.get_best_plan"):
            _ = planner

            min_progress_cutoff = self._min_progress_cutoff()
            negative_score_cutoff = self._negative_score_cutoff()

            candidate_data = candidate_data if candidate_data is not None else self.generate_candidate_bundle(
                shape_type,
                current_pose,
                target_pose,
                local_corners,
                radius,
                robot_pose=robot_pose,
            )

            local_target = candidate_data["local_target"]
            current_local_err = float(candidate_data.get("current_local_err", shape_alignment_error(
                shape_type,
                current_pose,
                local_target,
                local_corners,
                radius,
            )))
            global_shape_err = float(candidate_data.get("global_shape_err", shape_alignment_error(
                shape_type,
                current_pose,
                target_pose,
                local_corners,
                radius,
            )))
            chosen_lookahead = int(candidate_data.get("chosen_lookahead", config.REF_PATH_LOOKAHEAD))

            candidate_actions_count = len(candidate_data["candidate_actions"])
            shortlist_count = len(candidate_data.get("shortlisted_actions", []))
            feasible_count = len(candidate_data["feasible"])
            attach_attempt_count = candidate_data.get("attach_attempt_count", shortlist_count)
            attach_fail_count = candidate_data.get("attach_fail_count", max(0, attach_attempt_count - feasible_count))

            self._debug_log(
                f"get_best_plan counts: candidates={candidate_actions_count}, "
                f"shortlist={shortlist_count}, attach_attempts={attach_attempt_count}, "
                f"attach_failures={attach_fail_count}, feasible={feasible_count}, "
                f"relaxed_cutoff={config.DEBUG_RELAX_CUTOFF}, "
                f"exhaustive_attach_used={candidate_data.get('exhaustive_attach_used', False)}"
            )

            scored = []
            progress_pass_count = 0
            final_score_pass_count = 0

            for item in candidate_data["feasible"].values():
                cand = item["candidate"]
                motion = item["motion"]
                next_pose = item["next_pose"]

                push_len = float(np.linalg.norm(cand["end"] - cand["start"]))
                path_cost = motion["approach_length"] + push_len

                next_local_err = item["next_shape_err"]
                progress = item["progress"]
                align = item["align"]

                if progress >= min_progress_cutoff:
                    progress_pass_count += 1

                score = align + progress # - config.DIRECT_PATH_PENALTY * path_cost

                if score >= negative_score_cutoff:
                    final_score_pass_count += 1

                scored.append({
                    "candidate": cand,
                    "motion": motion,
                    "next_pose": next_pose,
                    "current_shape_err": current_local_err,
                    "next_shape_err": next_local_err,
                    "score": float(score),
                    "path_cost": float(path_cost),
                    "progress": float(progress),
                    "align": float(align),
                })

            self._debug_log(
                f"get_best_plan cutoffs: progress_pass={progress_pass_count}, "
                f"final_score_pass={final_score_pass_count}, "
                f"min_progress_cutoff={min_progress_cutoff:.4f}, "
                f"score_cutoff={negative_score_cutoff:.4f}"
            )

            if not scored:
                self._last_progress_debug_signature = None
                self._last_top_feasible_debug_signature = None
                failure_reason = "no_feasible_candidates"

                if allow_farther_retry and self._should_retry_with_farther_target(
                    failure_reason,
                    global_shape_err,
                    chosen_lookahead,
                ):
                    retry_lookahead = self._next_retry_lookahead(chosen_lookahead)

                    if retry_lookahead > chosen_lookahead:
                        self._debug_log(
                            f"retrying with farther local target: lookahead {chosen_lookahead} -> {retry_lookahead}"
                        )
                        retry_candidate_data = self.generate_candidate_bundle(
                            shape_type,
                            current_pose,
                            target_pose,
                            local_corners,
                            radius,
                            robot_pose=robot_pose,
                            lookahead_override=retry_lookahead,
                        )
                        retry_plan = self.get_best_plan(
                            shape_type,
                            current_pose,
                            target_pose,
                            local_corners,
                            planner=planner,
                            radius=radius,
                            robot_pose=robot_pose,
                            candidate_data=retry_candidate_data,
                            allow_farther_retry=(retry_lookahead < config.REF_PATH_LOOKAHEAD_FALLBACK_MAX),
                        )
                        if retry_plan is not None:
                            return retry_plan
                        return None

                self._report_plan_failure(failure_reason)
                self._report_no_plan_context(
                    current_local_err,
                    global_shape_err,
                    chosen_lookahead,
                    local_target,
                )
                return None

            self._last_plan_failure_reason = None
            self._last_progress_debug_signature = None
            self._last_top_feasible_debug_signature = None
            self._last_no_plan_debug_signature = None

            return max(scored, key=lambda item: item["score"])

    def get_best_action(self, shape_type, current_pose, target_pose, shape_info, planner=None, radius=None):
        if shape_type == "circle":
            local_corners = [(0.0, 0.0)]
        else:
            cx, cy, theta = current_pose
            local_corners = []
            c = np.cos(theta)
            s = np.sin(theta)
            r_t = np.array([[c, s], [-s, c]], dtype=float)
            for gx, gy in shape_info:
                lx, ly = r_t @ (np.array([gx, gy], dtype=float) - np.array([cx, cy], dtype=float))
                local_corners.append((float(lx), float(ly)))

        best = self.get_best_plan(
            shape_type,
            current_pose,
            target_pose,
            local_corners,
            planner=planner,
            radius=radius,
            robot_pose=config.ROBOT_START_XY,
        )

        if best is None:
            return None

        cand = best["candidate"]
        motion = best["motion"]
        return (
            float(cand["start"][0]),
            float(cand["start"][1]),
            float(cand["end"][0]),
            float(cand["end"][1]),
            cand["n_hat"],
            cand["t_hat"],
            float(motion["retreat_xy"][0]),
            float(motion["retreat_xy"][1]),
        )

    def update_com_belief(self, real_delta_theta, shape_type, pre_push_pose, candidate):
        shape_data = config.SHAPE_DB[shape_type]
        if len(shape_data["cases"]) <= 1:
            return

        best_candidate = self.com_belief[shape_type]
        min_error = float("inf")

        for cand_name in shape_data["cases"].keys():
            pred_pose = self._simulate_case(shape_type, pre_push_pose, candidate, cand_name)
            pred_delta_theta = wrap_angle(pred_pose[2] - pre_push_pose[2])
            err = abs(wrap_angle(real_delta_theta - pred_delta_theta))

            if err < min_error:
                min_error = err
                best_candidate = cand_name

        if min_error <= config.SYSID_ANGLE_DEADZONE:
            self.vote_box[shape_type][best_candidate] += 1

            if self.vote_box[shape_type][best_candidate] >= config.SYSID_VOTE_THRESHOLD:
                self.com_belief[shape_type] = best_candidate
                self.vote_box[shape_type] = {k: 0 for k in self.vote_box[shape_type]}

                if config.SYSID_VERBOSE:
                    print(f"[System ID] {shape_type} COM belief -> {best_candidate}")