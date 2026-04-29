import numpy as np
import matplotlib.pyplot as plt


class TabletopSoftRope:
    """
    Soft 2D rope on a tabletop.

    Constraints:
    - node 0 is fixed at hole_anchor
    - node 1 is constrained vertically above node 0
    - all nodes must satisfy y >= hole_anchor_y
    - gripper center x must stay in [grip_x_min, grip_x_max]

    Grip behavior:
    - when grasp starts, grip direction is aligned with local rope tangent
    - during dragging, that grasp direction is maintained
    """

    def __init__(
        self,
        num_nodes=20,
        total_length=1.0,
        total_mass=0.006,
        dt=0.01,
        damping=1.0,
        gravity=(0.0, 0.0),
        bending_stiffness=0.0002,
        constraint_iterations=150,
        stick_speed=0.0,
        grip_node_ratio=0.05,
        hole_anchor=(0.5, -0.131),
        grip_x_min=0.0,
        grip_x_max=1.1,
    ):
        if num_nodes < 2:
            raise ValueError("num_nodes must be at least 2")

        self.num_nodes = int(num_nodes)
        self.num_edges = self.num_nodes - 1

        self.total_mass = float(total_mass)
        self.node_mass = self.total_mass / self.num_nodes

        self.dt = float(dt)
        self.damping = float(damping)
        self.gravity = np.asarray(gravity, dtype=float)
        self.kb = float(bending_stiffness)
        self.constraint_iterations = int(constraint_iterations)
        self.stick_speed = float(stick_speed)

        self.hole_anchor = np.asarray(hole_anchor, dtype=float)
        self.hole_floor_y = float(self.hole_anchor[1])
        self.grip_x_min = float(grip_x_min)
        self.grip_x_max = float(grip_x_max)

        dx = total_length / self.num_edges

        self.rest_positions = np.array(
            [[self.hole_anchor[0], self.hole_anchor[1] + i * dx] for i in range(self.num_nodes)],
            dtype=float,
        )
        self.x = self.rest_positions.copy()
        self.v = np.zeros_like(self.x)
        self.rest_lengths = np.linalg.norm(self.x[1:] - self.x[:-1], axis=1)

        self.fixed = np.zeros(self.num_nodes, dtype=bool)
        self.fixed[0] = True
        self.fixed_position = self.hole_anchor.copy()

        self.dragged_idx = None
        self.mouse_world = None
        self.grip_direction = None

        self.grip_node_ratio = float(grip_node_ratio)
        self.grip_half_width = max(1, int(self.num_nodes * self.grip_node_ratio))

        self.last_feasibility_margin = 0.0
        self.last_blocked_reason = ""
        self.last_action_status = "ready"

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    def edges(self, x=None):
        if x is None:
            x = self.x
        return x[1:] - x[:-1]

    def edge_lengths(self, x=None):
        return np.linalg.norm(self.edges(x), axis=1)

    def tangents(self, x=None):
        e = self.edges(x)
        lengths = np.linalg.norm(e, axis=1, keepdims=True)
        lengths = np.clip(lengths, 1e-12, None)
        return e / lengths

    def turning_angles(self, x=None):
        if self.num_nodes < 3:
            return np.zeros(0, dtype=float)

        t = self.tangents(x)
        t_prev = t[:-1]
        t_next = t[1:]

        dots = np.sum(t_prev * t_next, axis=1)
        dots = np.clip(dots, -1.0, 1.0)

        crosses = t_prev[:, 0] * t_next[:, 1] - t_prev[:, 1] * t_next[:, 0]
        return np.arctan2(crosses, dots)

    def discrete_curvature(self, x=None):
        phi = self.turning_angles(x)
        phi = np.clip(phi, -np.pi + 1e-4, np.pi - 1e-4)
        return 2.0 * np.tan(0.5 * phi)

    def voronoi_lengths(self, x=None):
        lengths = self.edge_lengths(x)
        if len(lengths) < 2:
            return np.zeros(0, dtype=float)
        return 0.5 * (lengths[:-1] + lengths[1:])

    def get_grip_indices(self):
        if self.dragged_idx is None:
            return []

        left = max(0, self.dragged_idx - self.grip_half_width)
        right = min(self.num_nodes - 1, self.dragged_idx + self.grip_half_width)
        return list(range(left, right + 1))

    def arc_offset_from_center(self, j, center_k=None):
        if center_k is None:
            center_k = self.dragged_idx
        if center_k is None:
            return 0.0

        if j > center_k:
            return float(np.sum(self.rest_lengths[center_k:j]))
        if j < center_k:
            return -float(np.sum(self.rest_lengths[j:center_k]))
        return 0.0

    def get_rope_tangent_direction(self, center_k=None):
        if center_k is None:
            center_k = self.dragged_idx

        if center_k is None:
            return np.array([0.0, 1.0], dtype=float)

        center_k = int(np.clip(center_k, 0, self.num_nodes - 1))

        if center_k <= 0:
            d = self.x[1] - self.x[0]
        elif center_k >= self.num_nodes - 1:
            d = self.x[-1] - self.x[-2]
        else:
            d = self.x[center_k + 1] - self.x[center_k - 1]

        norm = np.linalg.norm(d)
        if norm < 1e-12:
            return np.array([0.0, 1.0], dtype=float)

        return d / norm

    def get_fixed_grip_direction(self, center=None):
        if self.grip_direction is not None:
            d = np.asarray(self.grip_direction, dtype=float)
            n = np.linalg.norm(d)
            if n > 1e-12:
                return d / n

        return self.get_rope_tangent_direction(self.dragged_idx)

    # ------------------------------------------------------------------
    # feasibility
    # ------------------------------------------------------------------
    def rigid_grip_targets_for_center(self, center=None):
        if self.dragged_idx is None:
            return {}

        if center is None:
            center = self.mouse_world
        if center is None:
            return {}

        d = self.get_fixed_grip_direction(center)
        center_k = self.dragged_idx
        grip_indices = self.get_grip_indices()

        targets = {}
        for j in grip_indices:
            s = self.arc_offset_from_center(j, center_k)
            targets[j] = center + s * d
        return targets

    def clamp_grip_center(self, center):
        center = np.asarray(center, dtype=float).copy()
        center[0] = np.clip(center[0], self.grip_x_min, self.grip_x_max)
        center[1] = max(center[1], self.hole_floor_y)
        return center

    def compute_rigid_grip_margin(self, center=None):
        if self.dragged_idx is None:
            return np.inf

        if center is None:
            center = self.mouse_world
        if center is None:
            return np.inf

        center = self.clamp_grip_center(center)
        targets = self.rigid_grip_targets_for_center(center=center)
        x0 = self.fixed_position
        margins = []

        for j, target in targets.items():
            if target[1] < self.hole_floor_y:
                margins.append(-1.0)
                continue

            if j == 0:
                margins.append(-np.linalg.norm(target - x0))
                continue

            max_reach = float(np.sum(self.rest_lengths[:j]))
            dist = float(np.linalg.norm(target - x0))
            margins.append(max_reach - dist)

        if len(margins) == 0:
            return np.inf

        return float(min(margins))

    def is_rigid_grip_feasible(self, center=None, tol=1e-9):
        margin = self.compute_rigid_grip_margin(center=center)
        self.last_feasibility_margin = margin
        return margin >= -tol

    def project_center_to_feasible(self, center):
        if self.dragged_idx is None:
            return self.clamp_grip_center(center)

        center = self.clamp_grip_center(center)

        if self.is_rigid_grip_feasible(center=center):
            return center.copy()

        x0 = self.fixed_position
        rel = center - x0
        r = np.linalg.norm(rel)

        if r < 1e-12:
            return x0.copy()

        u = rel / r
        lo, hi = 0.0, r

        for _ in range(50):
            mid = 0.5 * (lo + hi)
            test_center = x0 + mid * u
            test_center = self.clamp_grip_center(test_center)
            if self.is_rigid_grip_feasible(center=test_center):
                lo = mid
            else:
                hi = mid

        return self.clamp_grip_center(x0 + lo * u)

    # ------------------------------------------------------------------
    # energy / forces
    # ------------------------------------------------------------------
    def bending_energy(self, x=None):
        if self.kb <= 0.0 or self.num_nodes < 3:
            return 0.0

        kappa = self.discrete_curvature(x)
        lv = self.voronoi_lengths(x)
        lv = np.clip(lv, 1e-12, None)
        return 0.5 * self.kb * np.sum((kappa ** 2) / lv)

    def bending_forces_fd(self, eps=1e-6):
        if self.kb <= 0.0 or self.num_nodes < 3:
            return np.zeros_like(self.x)

        f = np.zeros_like(self.x)
        base_x = self.x.copy()

        for i in range(self.num_nodes):
            if self.fixed[i]:
                continue
            for d in range(2):
                x_plus = base_x.copy()
                x_minus = base_x.copy()
                x_plus[i, d] += eps
                x_minus[i, d] -= eps
                e_plus = self.bending_energy(x_plus)
                e_minus = self.bending_energy(x_minus)
                grad = (e_plus - e_minus) / (2.0 * eps)
                f[i, d] = -grad

        return f

    def external_forces(self):
        f = np.zeros_like(self.x)
        for i in range(self.num_nodes):
            if self.fixed[i]:
                continue
            f[i] += self.node_mass * self.gravity
        return f

    # ------------------------------------------------------------------
    # constraints
    # ------------------------------------------------------------------
    def project_inextensibility(self):
        for _ in range(self.constraint_iterations):
            for i in range(self.num_edges):
                j = i + 1
                delta = self.x[j] - self.x[i]
                dist = np.linalg.norm(delta)

                if dist < 1e-12:
                    continue

                rest = self.rest_lengths[i]
                diff = dist - rest
                direction = delta / dist

                if self.fixed[i] and self.fixed[j]:
                    continue
                elif self.fixed[i]:
                    self.x[j] -= diff * direction
                elif self.fixed[j]:
                    self.x[i] += diff * direction
                else:
                    correction = 0.5 * diff * direction
                    self.x[i] += correction
                    self.x[j] -= correction

            self.x[self.fixed] = self.fixed_position
            self.enforce_hole_guide()
            self.enforce_floor_constraint()

    def enforce_hole_guide(self):
    # 두 번째 노드를 더 이상 guide node로 강제하지 않음
        self.x[0] = self.fixed_position.copy()

    def enforce_floor_constraint(self):
        self.x[:, 1] = np.maximum(self.x[:, 1], self.hole_floor_y)

    def enforce_velocity_consistency(self, x_prev):
        self.v = (self.x - x_prev) / self.dt
        self.v[self.fixed] = 0.0

    def apply_stick_condition(self):
        if self.dragged_idx is not None:
            return

        if self.stick_speed <= 0.0:
            return

        speeds = np.linalg.norm(self.v, axis=1)
        mask = (~self.fixed) & (speeds < self.stick_speed)
        self.v[mask] = 0.0

    def apply_rigid_grip(self):
        if self.dragged_idx is None or self.mouse_world is None:
            return
        if self.fixed[self.dragged_idx]:
            return

        self.mouse_world = self.clamp_grip_center(self.mouse_world)

        d = self.get_fixed_grip_direction(self.mouse_world)
        center_k = self.dragged_idx
        grip_indices = self.get_grip_indices()

        for j in grip_indices:
            if self.fixed[j]:
                continue

            s = self.arc_offset_from_center(j, center_k)
            target = self.mouse_world + s * d
            target[1] = max(target[1], self.hole_floor_y)
            self.x[j] = target
            self.v[j] = 0.0

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    def step(self):
        x_prev = self.x.copy()

        f = self.external_forces() + self.bending_forces_fd()
        a = f / self.node_mass
        a[self.fixed] = 0.0

        self.v += self.dt * a
        self.v *= self.damping
        self.v[self.fixed] = 0.0
        self.x += self.dt * self.v

        self.x[self.fixed] = self.fixed_position
        self.enforce_hole_guide()
        self.enforce_floor_constraint()

        self.project_inextensibility()

        if self.dragged_idx is not None and self.mouse_world is not None:
            self.mouse_world = self.project_center_to_feasible(self.mouse_world)
            self.apply_rigid_grip()

        self.project_inextensibility()

        if self.dragged_idx is not None and self.mouse_world is not None:
            self.apply_rigid_grip()

        self.x[self.fixed] = self.fixed_position
        self.enforce_hole_guide()
        self.enforce_floor_constraint()

        self.enforce_velocity_consistency(x_prev)
        self.apply_stick_condition()

    # ------------------------------------------------------------------
    # coordinate-based grip API
    # ------------------------------------------------------------------
    def set_grip_by_index_and_center(self, idx, center):
        idx = int(idx)
        if idx <= 0 or idx >= self.num_nodes:
            raise ValueError("Grip index must be in [1, num_nodes-1] for this model.")

        self.dragged_idx = idx
        self.grip_direction = self.get_rope_tangent_direction(idx)
        self.mouse_world = self.project_center_to_feasible(np.asarray(center, dtype=float))
        self.last_action_status = "grip selected by coordinate"
        self.last_blocked_reason = ""

    def set_grip_by_nearest_point(self, point, max_dist=0.08):
        point = np.asarray(point, dtype=float)
        d = np.linalg.norm(self.x - point[None, :], axis=1)
        d[self.fixed] = np.inf
        if self.num_nodes > 1:
            d[1] = np.inf

        idx = int(np.argmin(d))
        if d[idx] > max_dist:
            raise ValueError(f"No graspable node within max_dist={max_dist}")

        self.set_grip_by_index_and_center(idx, point)
        return idx

    def move_grip_center(self, center, project=True):
        if self.dragged_idx is None:
            raise RuntimeError("No active grip. Call set_grip_by_index_and_center first.")

        center = np.asarray(center, dtype=float)
        if project:
            self.mouse_world = self.project_center_to_feasible(center)
        else:
            self.mouse_world = self.clamp_grip_center(center)

        self.last_action_status = "center updated by coordinate"
        self.last_blocked_reason = ""

    def release_grip(self):
        self.dragged_idx = None
        self.mouse_world = None
        self.grip_direction = None
        self.last_action_status = "released"
        self.last_blocked_reason = ""


# ----------------------------------------------------------------------
# deterministic coordinate-driven simulation helpers
# ----------------------------------------------------------------------
def linear_centers(start_center, target_center, drag_steps):
    start_center = np.asarray(start_center, dtype=float)
    target_center = np.asarray(target_center, dtype=float)

    centers = []
    for t in range(1, drag_steps + 1):
        alpha = t / drag_steps
        c = (1.0 - alpha) * start_center + alpha * target_center
        centers.append(c)
    return np.asarray(centers)


def run_linear_drag_simulation(
    rope: TabletopSoftRope,
    grip_idx: int,
    grasp_center,
    target_center,
    drag_steps=80,
    relax_steps=60,
    project=True,
):
    grasp_center = np.asarray(grasp_center, dtype=float)
    target_center = np.asarray(target_center, dtype=float)

    rope.set_grip_by_index_and_center(grip_idx, grasp_center)

    centers = linear_centers(grasp_center, target_center, drag_steps)

    pos_hist = []
    center_hist = []
    energy_hist = []
    length_err_hist = []
    floor_margin_hist = []

    for c in centers:
        rope.move_grip_center(c, project=project)
        rope.step()

        pos_hist.append(rope.x.copy())
        center_hist.append(None if rope.mouse_world is None else rope.mouse_world.copy())
        energy_hist.append(rope.bending_energy())

        edge_lengths = np.linalg.norm(rope.x[1:] - rope.x[:-1], axis=1)
        length_err_hist.append(np.max(np.abs(edge_lengths - rope.rest_lengths)))
        floor_margin_hist.append(np.min(rope.x[:, 1] - rope.hole_floor_y))

    rope.release_grip()

    for _ in range(relax_steps):
        rope.step()

        pos_hist.append(rope.x.copy())
        center_hist.append(np.array([np.nan, np.nan], dtype=float))
        energy_hist.append(rope.bending_energy())

        edge_lengths = np.linalg.norm(rope.x[1:] - rope.x[:-1], axis=1)
        length_err_hist.append(np.max(np.abs(edge_lengths - rope.rest_lengths)))
        floor_margin_hist.append(np.min(rope.x[:, 1] - rope.hole_floor_y))

    result = {
        "final_pos": rope.x.copy(),
        "pos_hist": np.asarray(pos_hist),
        "center_hist": np.asarray(center_hist, dtype=float),
        "energy_hist": np.asarray(energy_hist),
        "length_err_hist": np.asarray(length_err_hist),
        "floor_margin_hist": np.asarray(floor_margin_hist),
        "input_centers": centers.copy(),
    }
    return result


def compare_two_runs_same_input():
    """
    완전히 같은 좌표 입력 시퀀스를 두 번 넣어서
    재현성이 있는지 확인하는 테스트.
    """
    kwargs = dict(
        num_nodes=20,
        total_length=0.8333,
        total_mass=0.006,
        dt=0.01,
        damping=0.95,
        gravity=(0.0, 0.0),
        bending_stiffness=1.8e-5,
        constraint_iterations=1,
        stick_speed=0.06,
        grip_node_ratio=0.0,
        hole_anchor=(0.5, -0.131),
        grip_x_min=0.0,
        grip_x_max=1.1,
    )

    grip_idx = 10

    rope1 = TabletopSoftRope(**kwargs)
    rope2 = TabletopSoftRope(**kwargs)

    grasp_center = rope1.x[grip_idx].copy()
    target_center = grasp_center + np.array([0.18, 0.08], dtype=float)

    out1 = run_linear_drag_simulation(
        rope1, grip_idx, grasp_center, target_center,
        drag_steps=80, relax_steps=60, project=True
    )
    out2 = run_linear_drag_simulation(
        rope2, grip_idx, grasp_center, target_center,
        drag_steps=80, relax_steps=60, project=True
    )

    abs_diff = np.abs(out1["final_pos"] - out2["final_pos"])
    print("=== Repeatability test with same coordinate input ===")
    print("max abs diff :", np.max(abs_diff))
    print("rmse         :", np.sqrt(np.mean((out1["final_pos"] - out2["final_pos"]) ** 2)))

    return out1, out2


def demo_linear_drag():
    kwargs = dict(
        num_nodes=20,
        total_length=0.8333,
        total_mass=0.006,
        dt=0.01,
        damping=0.7077,
        gravity=(0.0, 0.0),
        bending_stiffness=3.225057e-05 ,
        constraint_iterations=15,
        stick_speed=0.13333334,
        grip_node_ratio=0.0,
        hole_anchor=(0.5, -0.131),
        grip_x_min=0.0,
        grip_x_max=1.1,
    )

    rope = TabletopSoftRope(**kwargs)
    x_init = rope.x.copy()

    grip_idx = 10
    grasp_center = rope.x[grip_idx].copy()
    target_center = grasp_center + np.array([0.18, 0.08], dtype=float)

    out = run_linear_drag_simulation(
        rope,
        grip_idx=grip_idx,
        grasp_center=grasp_center,
        target_center=target_center,
        drag_steps=80,
        relax_steps=60,
        project=True,
    )

    x_final = out["final_pos"]
    center_hist = out["center_hist"]
    input_centers = out["input_centers"]

    print("=== Diagnostics ===")
    print("Final bending energy      :", out["energy_hist"][-1])
    print("Max length error          :", np.max(out["length_err_hist"]))
    print("Min floor margin          :", np.min(out["floor_margin_hist"]))
    print("Guide x error (node 1)    :", abs(rope.x[1, 0] - rope.hole_anchor[0]))
    print("Guide y error (node 1)    :", abs(rope.x[1, 1] - (rope.hole_anchor[1] + rope.rest_lengths[0])))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(x_init[:, 0], x_init[:, 1], "-o", label="initial")
    ax.plot(x_final[:, 0], x_final[:, 1], "-o", label="final")
    ax.axhline(rope.hole_floor_y, linestyle="--")
    ax.plot(rope.hole_anchor[0], rope.hole_anchor[1], "ks", label="hole")
    ax.set_title("Initial vs final rope")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(out["energy_hist"])
    ax.set_title("Bending energy")
    ax.set_xlabel("step")
    ax.set_ylabel("energy")
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(out["length_err_hist"])
    ax.set_title("Max inextensibility error")
    ax.set_xlabel("step")
    ax.set_ylabel("max |edge-rest|")
    ax.grid(True)

    ax = axes[1, 1]
    ax.plot(input_centers[:, 0], input_centers[:, 1], "--", label="input linear centers")
    valid = ~np.isnan(center_hist[:, 0])
    ax.plot(center_hist[valid, 0], center_hist[valid, 1], label="applied centers")
    ax.set_title("Grip center path")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    out1, out2 = compare_two_runs_same_input()
    demo_linear_drag()