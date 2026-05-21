import heapq  # A* 알고리즘에서 가장 비용이 적은 경로를 빠르게 뽑아내기 위한 우선순위 큐(Heap) 라이브러리
from typing import Dict, List, Optional, Sequence, Tuple  # 타입 힌트용

import numpy as np  # 벡터 및 행렬 연산 라이브러리

from task1 import config  # 설정값 임포트
from geometry_utils import (
    clip_polygon_to_aabb,
    distance_point_to_polygon,
    path_length,
    segment_hits_polygon,
    point_in_polygon,
)
from perf_utils import perf_timer  # 성능 측정용 타이머


class XYMotionPlanner:
    def __init__(
        self,
        workspace_min=(-0.07, -0.075),  # 작업 공간의 최소 X, Y 좌표
        workspace_max=(0.07, 0.075),    # 작업 공간의 최대 X, Y 좌표
        grid_resolution=0.0025,         # 길찾기를 위해 바둑판(Grid)으로 쪼갤 때 1칸의 크기
        obstacle_margin=0.003,          # 물체(장애물)에서 로봇이 떨어져야 하는 최소 안전거리
    ):
        self.workspace_min = np.asarray(workspace_min, dtype=float)
        self.workspace_max = np.asarray(workspace_max, dtype=float)
        self.grid_resolution = float(grid_resolution)
        self.obstacle_margin = float(obstacle_margin)

        # 반복적인 연산을 줄이기 위한 캐싱 변수들
        self.cached_polygon = None
        self.blocked_cells = set()
        self.last_debug_info = {}

    def _set_last_debug_info(self, **kwargs):
        if not isinstance(self.last_debug_info, dict):
            self.last_debug_info = {}
        self.last_debug_info.update(kwargs)

    def _print_last_debug_info(self):
        if not getattr(config, "VERBOSE_DEBUG", False):
            return

        direct_ok = self.last_debug_info.get("direct_ok")
        lshape_ok = self.last_debug_info.get("lshape_ok")
        ushape_ok = self.last_debug_info.get("ushape_ok")
        astar_attempted = self.last_debug_info.get("astar_attempted")
        astar_ok = self.last_debug_info.get("astar_ok")
        failure_reason = self.last_debug_info.get("failure_reason")
        selected_path_type = self.last_debug_info.get("selected_path_type")
        approach_waypoints = self.last_debug_info.get("approach_waypoints")
        max_approach_waypoints = self.last_debug_info.get("max_approach_waypoints")
        raw_approach_waypoints = self.last_debug_info.get("raw_approach_waypoints")
        smoothed_approach_waypoints = self.last_debug_info.get("smoothed_approach_waypoints")

        print("[MotionPlanner DEBUG] direct_ok:", direct_ok)
        print("[MotionPlanner DEBUG] lshape_ok:", lshape_ok)
        print("[MotionPlanner DEBUG] ushape_ok:", ushape_ok)
        print("[MotionPlanner DEBUG] astar_attempted:", astar_attempted)
        print("[MotionPlanner DEBUG] astar_ok:", astar_ok)
        print("[MotionPlanner DEBUG] selected_path_type:", selected_path_type)
        print("[MotionPlanner DEBUG] failure_reason:", failure_reason)

        if approach_waypoints is not None:
            print("[MotionPlanner DEBUG] approach_waypoints:", approach_waypoints)
        if max_approach_waypoints is not None:
            print("[MotionPlanner DEBUG] max_approach_waypoints:", max_approach_waypoints)
        if raw_approach_waypoints is not None:
            print("[MotionPlanner DEBUG] raw_approach_waypoints:", raw_approach_waypoints)
        if smoothed_approach_waypoints is not None:
            print("[MotionPlanner DEBUG] smoothed_approach_waypoints:", smoothed_approach_waypoints)

    def _point_to_grid(self, point: Sequence[float]) -> Tuple[int, int]:
        # 실제 좌표(x, y)를 바둑판 칸 번호(i, j)로 변환
        p = np.asarray(point, dtype=float)
        ij = np.round((p - self.workspace_min) / self.grid_resolution).astype(int)
        return int(ij[0]), int(ij[1])

    def _grid_to_point(self, ij: Tuple[int, int]) -> np.ndarray:
        # 바둑판 칸 번호(i, j)를 실제 좌표(x, y)로 역변환
        return self.workspace_min + self.grid_resolution * np.array(ij, dtype=float)

    def _inside_bounds(self, point: Sequence[float]) -> bool:
        # 특정 좌표가 작업 공간 범위 안에 있는지 검사
        p = np.asarray(point, dtype=float)
        return bool(np.all(p >= self.workspace_min) and np.all(p <= self.workspace_max))

    def _sanitize_obstacle_polygon(self, obstacle_polygon):
        """
        planner에서 사용할 장애물 polygon을 workspace 내부 부분만 남기도록 정리.
        - 물체가 workspace 경계에 걸친 경우: 안쪽 부분만 사용
        - workspace 안에 사실상 남는 부분이 없으면: 빈 리스트 반환
        """
        if obstacle_polygon is None or len(obstacle_polygon) == 0:
            return []

        clipped = clip_polygon_to_aabb(
            obstacle_polygon,
            self.workspace_min,
            self.workspace_max,
        )

        if clipped is None or len(clipped) < 3:
            return []

        return [tuple(map(float, p)) for p in clipped]

    def _is_free(self, point: Sequence[float], obstacle_polygon) -> bool:
        # 특정 좌표로 로봇이 가도 물체에 안 부딪히는지 검사
        if obstacle_polygon is None or len(obstacle_polygon) < 3:
            return self._inside_bounds(point)

        return (
            self._inside_bounds(point)
            and distance_point_to_polygon(point, obstacle_polygon) > self.obstacle_margin
        )

    # [수정 추가] start/goal endpoint margin relaxation helper
    def _relax_endpoint_cell_if_margin_only(
        self,
        point: Sequence[float],
        obstacle_polygon,
        cell_ij: Tuple[int, int],
    ):
        """
        start/goal endpoint가 inflated obstacle margin 때문에 blocked 된 경우만 완화한다.

        핵심 원칙:
        - 실제 물체 polygon 내부면 절대 허용하지 않음.
        - 물체 밖인데 obstacle_margin 때문에 blocked 된 endpoint는 허용.
        - 중간 waypoint/경로 충돌 검사는 기존 obstacle_margin 기준을 유지함.
        """
        if obstacle_polygon is None or len(obstacle_polygon) < 3:
            return False, None

        point = np.asarray(point, dtype=float)
        if not self._inside_bounds(point):
            return False, None

        # 진짜 물체 내부면 허용하면 안 된다.
        if point_in_polygon(point, obstacle_polygon):
            return False, 0.0

        dist = float(distance_point_to_polygon(point, obstacle_polygon))
        eps = float(getattr(config, "ENDPOINT_MARGIN_RELAX_EPS", 1e-9))

        # 실제 물체 밖이고 경계에서 아주 조금이라도 떨어져 있으면
        # margin band 때문에만 막힌 endpoint로 보고 허용한다.
        if dist > eps:
            self.blocked_cells.discard(cell_ij)
            return True, dist

        return False, dist

    def _update_costmap(self, obstacle_polygon):
        # A* 길찾기를 하기 전에, 물체가 있는 부분을 바둑판에 '접근 금지(Blocked)'로 칠하는 함수
        with perf_timer("motion_planner._update_costmap", every=config.PERF_LOG_COSTMAP_EVERY):
            obstacle_polygon = self._sanitize_obstacle_polygon(obstacle_polygon)

            # workspace 안에 남은 장애물이 사실상 없으면 costmap 비움
            if len(obstacle_polygon) < 3:
                self.cached_polygon = []
                self.blocked_cells = set()
                return

            # 이전 턴과 비교해서 물체의 모양이 거의 같다면 costmap 재계산 생략
            if self.cached_polygon is not None and len(self.cached_polygon) == len(obstacle_polygon):
                same = True
                for p1, p2 in zip(self.cached_polygon, obstacle_polygon):
                    if abs(p1[0] - p2[0]) > 1e-5 or abs(p1[1] - p2[1]) > 1e-5:
                        same = False
                        break
                if same:
                    return

            # 물체가 움직였으면 캐시 새로 갱신
            self.cached_polygon = [tuple(p) for p in obstacle_polygon]
            self.blocked_cells = set()

            # 장애물 bounding box 계산
            xs = [p[0] for p in obstacle_polygon]
            ys = [p[1] for p in obstacle_polygon]

            min_x = min(xs) - self.obstacle_margin - self.grid_resolution
            max_x = max(xs) + self.obstacle_margin + self.grid_resolution
            min_y = min(ys) - self.obstacle_margin - self.grid_resolution
            max_y = max(ys) + self.obstacle_margin + self.grid_resolution

            min_ij = self._point_to_grid((min_x, min_y))
            max_ij = self._point_to_grid((max_x, max_y))

            # bounding box 안의 grid cell에 대해 정밀 검사
            for i in range(min_ij[0], max_ij[0] + 1):
                for j in range(min_ij[1], max_ij[1] + 1):
                    pt = self._grid_to_point((i, j))

                    if not self._inside_bounds(pt):
                        self.blocked_cells.add((i, j))
                        continue

                    if distance_point_to_polygon(pt, obstacle_polygon) <= self.obstacle_margin:
                        self.blocked_cells.add((i, j))

    def _polyline_is_valid(
        self,
        points: Sequence[Sequence[float]],
        obstacle_polygon,
        check_endpoints_free: bool = False,
    ) -> bool:
        """
        여러 waypoint를 잇는 경로가 모두 workspace 안이고,
        각 선분이 물체 + margin과 충돌하지 않는지 검사한다.

        check_endpoints_free=False인 이유:
        - start/goal은 attach_approach에서 이미 workspace 검사를 한다.
        - approach point가 obstacle_margin 경계에 아주 가까우면 수치 오차 때문에
          _is_free에서 실패할 수 있다.
        - 그래서 중간 waypoint만 _is_free로 강하게 검사하고,
          전체 선분 충돌은 segment_hits_polygon으로 검사한다.
        """
        if points is None or len(points) < 2:
            return False

        pts = [np.asarray(p, dtype=float) for p in points]

        for idx, p in enumerate(pts):
            if not self._inside_bounds(p):
                return False

            is_endpoint = idx == 0 or idx == len(pts) - 1
            if check_endpoints_free or not is_endpoint:
                if not self._is_free(p, obstacle_polygon):
                    return False

        for a, b in zip(pts[:-1], pts[1:]):
            if segment_hits_polygon(a, b, obstacle_polygon, margin=self.obstacle_margin):
                return False

        return True

    def _try_l_shape_paths(self, start: np.ndarray, goal: np.ndarray, obstacle_polygon) -> Optional[List[np.ndarray]]:
        """
        직선 경로가 막혔을 때 L자 경로 2개를 검사한다.
        1) start -> (start.x, goal.y) -> goal
        2) start -> (goal.x, start.y) -> goal
        """
        via1 = np.array([start[0], goal[1]], dtype=float)
        via2 = np.array([goal[0], start[1]], dtype=float)

        candidates = [
            [start.copy(), via1, goal.copy()],
            [start.copy(), via2, goal.copy()],
        ]

        valid_paths = []
        for path in candidates:
            if self._polyline_is_valid(path, obstacle_polygon, check_endpoints_free=False):
                valid_paths.append(path)

        if not valid_paths:
            return None

        valid_paths.sort(key=path_length)
        return valid_paths[0]

    def _try_u_shape_paths(self, start: np.ndarray, goal: np.ndarray, obstacle_polygon) -> Optional[List[np.ndarray]]:
        """
        직선/L자 경로가 막혔을 때, 물체 bounding box 바깥을 ㄷ자 형태로 우회한다.

        후보 4개:
        - 위쪽 우회:    start -> (start.x, top)    -> (goal.x, top)    -> goal
        - 아래쪽 우회:  start -> (start.x, bottom) -> (goal.x, bottom) -> goal
        - 왼쪽 우회:    start -> (left, start.y)   -> (left, goal.y)   -> goal
        - 오른쪽 우회:  start -> (right, start.y)  -> (right, goal.y)  -> goal

        A*보다 훨씬 싸고 waypoint도 4개로 고정이라,
        L자로 안 되는 케이스를 일부 살리면서 속도를 덜 잡아먹는다.
        """
        if obstacle_polygon is None or len(obstacle_polygon) < 3:
            return None

        obs = np.asarray(obstacle_polygon, dtype=float)

        min_x = float(np.min(obs[:, 0]))
        max_x = float(np.max(obs[:, 0]))
        min_y = float(np.min(obs[:, 1]))
        max_y = float(np.max(obs[:, 1]))

        clearance_scale = getattr(config, "U_SHAPE_CLEARANCE_SCALE", 2.0)
        clearance_extra = getattr(config, "U_SHAPE_CLEARANCE_EXTRA", self.grid_resolution)
        clearance = self.obstacle_margin * float(clearance_scale) + float(clearance_extra)

        x_left = min_x - clearance
        x_right = max_x + clearance
        y_bottom = min_y - clearance
        y_top = max_y + clearance

        candidates = [
            # 위쪽 ㄷ자
            [
                start.copy(),
                np.array([start[0], y_top], dtype=float),
                np.array([goal[0], y_top], dtype=float),
                goal.copy(),
            ],
            # 아래쪽 ㄷ자
            [
                start.copy(),
                np.array([start[0], y_bottom], dtype=float),
                np.array([goal[0], y_bottom], dtype=float),
                goal.copy(),
            ],
            # 왼쪽 ㄷ자
            [
                start.copy(),
                np.array([x_left, start[1]], dtype=float),
                np.array([x_left, goal[1]], dtype=float),
                goal.copy(),
            ],
            # 오른쪽 ㄷ자
            [
                start.copy(),
                np.array([x_right, start[1]], dtype=float),
                np.array([x_right, goal[1]], dtype=float),
                goal.copy(),
            ],
        ]

        valid_paths = []
        for path in candidates:
            if self._polyline_is_valid(path, obstacle_polygon, check_endpoints_free=False):
                valid_paths.append(path)

        if not valid_paths:
            return None

        valid_paths.sort(key=path_length)
        return valid_paths[0]

    def _make_base_debug_info(
        self,
        start,
        goal,
        direct_attempted=False,
        direct_ok=False,
        lshape_attempted=False,
        lshape_ok=False,
        ushape_attempted=False,
        ushape_ok=False,
        astar_attempted=False,
        astar_ok=False,
        failure_reason=None,
        start_grid=None,
        goal_grid=None,
        start_blocked=None,
        goal_blocked=None,
        selected_path_type=None,
    ):
        return {
            "direct_attempted": direct_attempted,
            "direct_ok": direct_ok,
            "lshape_attempted": lshape_attempted,
            "lshape_ok": lshape_ok,
            "ushape_attempted": ushape_attempted,
            "ushape_ok": ushape_ok,
            "astar_attempted": astar_attempted,
            "astar_ok": astar_ok,
            "failure_reason": failure_reason,
            "selected_path_type": selected_path_type,
            "start_xy": tuple(float(v) for v in start),
            "goal_xy": tuple(float(v) for v in goal),
            "start_grid": start_grid,
            "goal_grid": goal_grid,
            "start_blocked": start_blocked,
            "goal_blocked": goal_blocked,
            "visited_count": 0,
            "expanded_count": 0,
            "approach_xy": None,
            "push_start": None,
            "push_end": None,
            "retreat_xy": None,
        }

    def _with_original_start_if_needed(
        self,
        path: Optional[List[np.ndarray]],
        original_start: np.ndarray,
        escape_start_used: bool,
    ) -> Optional[List[np.ndarray]]:
        """
        start margin escape를 쓴 경우 실제 실행 경로 앞에 원래 로봇 위치를 붙인다.
        이렇게 해야 executor가 현재 위치 -> safe_start -> goal 순서로 이동한다.
        """
        if path is None:
            return None
        if not escape_start_used:
            return path

        original_start = np.asarray(original_start, dtype=float)
        first = np.asarray(path[0], dtype=float)
        if np.linalg.norm(first - original_start) < 1e-9:
            return path
        return [original_start.copy()] + path

    def _find_start_escape_point(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacle_polygon,
    ) -> Optional[np.ndarray]:
        """
        로봇 시작점이 실제 물체 내부는 아니지만 obstacle_margin 안쪽/경계에 있을 때,
        planning을 바로 실패시키지 않고 물체 바깥쪽의 가장 가까운 free point를 찾는다.

        핵심:
        - 실제 polygon 내부면 위험하므로 escape를 허용하지 않는다.
        - margin band 안에 있는 경우는 실제 물체를 관통하지 않는 선에서 바깥으로 빠진다.
        - 이 함수는 경로 전체를 만드는 게 아니라 첫 탈출 지점 safe_start만 찾는다.
        """
        obstacle_polygon = self._sanitize_obstacle_polygon(obstacle_polygon)
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)

        if not self._inside_bounds(start):
            return None
        if len(obstacle_polygon) < 3:
            return start.copy()

        # 진짜 물체 내부에 있으면 안전하게 탈출 방향을 가정하기 어렵다.
        if point_in_polygon(start, obstacle_polygon):
            return None

        safe_extra = float(getattr(config, "START_MARGIN_ESCAPE_SAFE_EXTRA", self.grid_resolution))
        safe_clearance = float(self.obstacle_margin + safe_extra)
        start_dist = float(distance_point_to_polygon(start, obstacle_polygon))

        # 이미 충분히 안전하면 그대로 사용한다.
        if start_dist >= safe_clearance and self._is_free(start, obstacle_polygon):
            return start.copy()

        obs = np.asarray(obstacle_polygon, dtype=float)
        obs_center = np.mean(obs, axis=0)

        dirs = []

        # 1순위: 물체 중심에서 로봇 위치로 향하는 방향 = 물체에서 멀어지는 방향
        away = start - obs_center
        away_norm = np.linalg.norm(away)
        if away_norm > 1e-9:
            dirs.append(away / away_norm)

        # 2순위: 목표 방향. 단, 물체에서 멀어지는 방향이 아닐 수 있어서 우선순위는 낮다.
        to_goal = goal - start
        to_goal_norm = np.linalg.norm(to_goal)
        if to_goal_norm > 1e-9:
            dirs.append(to_goal / to_goal_norm)

        # 3순위: 전방향 샘플링
        n_dirs = int(getattr(config, "START_MARGIN_ESCAPE_DIRS", 32))
        for k in range(max(4, n_dirs)):
            th = 2.0 * np.pi * k / max(4, n_dirs)
            dirs.append(np.array([np.cos(th), np.sin(th)], dtype=float))

        max_radius = float(getattr(
            config,
            "START_MARGIN_ESCAPE_MAX_DIST",
            self.obstacle_margin + 4.0 * self.grid_resolution,
        ))
        max_radius = max(max_radius, safe_clearance + 2.0 * self.grid_resolution)

        step = max(float(self.grid_resolution), 1e-6)
        radii = np.arange(step, max_radius + 0.5 * step, step)

        best = None
        best_score = float("inf")

        for r in radii:
            for d in dirs:
                q = start + r * d

                if not self._inside_bounds(q):
                    continue
                if point_in_polygon(q, obstacle_polygon):
                    continue

                q_dist = float(distance_point_to_polygon(q, obstacle_polygon))
                if q_dist < safe_clearance:
                    continue

                # start -> q는 margin band 안에서 빠져나가는 특수 구간이라 inflated margin은 무시한다.
                # 하지만 실제 물체 polygon 자체를 관통하면 안 된다.
                if segment_hits_polygon(start, q, obstacle_polygon, margin=0.0):
                    continue

                # 가까운 탈출점을 우선. 목표 방향과 비슷하면 약간 우대.
                score = float(r)
                if to_goal_norm > 1e-9:
                    score -= 0.05 * float(np.dot(d, to_goal / to_goal_norm))

                if score < best_score:
                    best_score = score
                    best = q.copy()

            if best is not None:
                return best

        return None

    def plan_path(self, start: Sequence[float], goal: Sequence[float], obstacle_polygon) -> Optional[List[np.ndarray]]:
        # 시작점에서 목표점까지 물체를 피해 가는 경로 생성
        obstacle_polygon = self._sanitize_obstacle_polygon(obstacle_polygon)

        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        original_start = start.copy()
        escape_start_used = False
        escape_start_xy = None

        direct_attempted = False
        direct_ok = False
        lshape_attempted = False
        lshape_ok = False
        ushape_attempted = False
        ushape_ok = False
        astar_attempted = False
        astar_ok = False
        failure_reason = None

        # start / goal bounds 먼저 방어
        if not self._inside_bounds(start):
            self.last_debug_info = self._make_base_debug_info(
                start=start,
                goal=goal,
                direct_attempted=direct_attempted,
                direct_ok=direct_ok,
                lshape_attempted=lshape_attempted,
                lshape_ok=lshape_ok,
                ushape_attempted=ushape_attempted,
                ushape_ok=ushape_ok,
                astar_attempted=astar_attempted,
                astar_ok=astar_ok,
                failure_reason="start_out_of_bounds",
            )
            self._print_last_debug_info()
            return None

        if not self._inside_bounds(goal):
            self.last_debug_info = self._make_base_debug_info(
                start=start,
                goal=goal,
                direct_attempted=direct_attempted,
                direct_ok=direct_ok,
                lshape_attempted=lshape_attempted,
                lshape_ok=lshape_ok,
                ushape_attempted=ushape_attempted,
                ushape_ok=ushape_ok,
                astar_attempted=astar_attempted,
                astar_ok=astar_ok,
                failure_reason="goal_out_of_bounds",
            )
            self._print_last_debug_info()
            return None

        # 장애물이 없으면 직선 경로로 바로 반환
        if len(obstacle_polygon) < 3:
            direct_attempted = True
            direct_ok = True
            self.last_debug_info = self._make_base_debug_info(
                start=start,
                goal=goal,
                direct_attempted=direct_attempted,
                direct_ok=direct_ok,
                lshape_attempted=False,
                lshape_ok=False,
                ushape_attempted=False,
                ushape_ok=False,
                astar_attempted=False,
                astar_ok=False,
                failure_reason=None,
                start_grid=self._point_to_grid(start),
                goal_grid=self._point_to_grid(goal),
                start_blocked=False,
                goal_blocked=False,
                selected_path_type="direct_no_obstacle",
            )
            self._print_last_debug_info()
            return self._with_original_start_if_needed([start.copy(), goal.copy()], original_start, escape_start_used)

        # -------------------------------------------------
        # initial robot margin escape
        # -------------------------------------------------
        # 로봇 연결 직후 푸셔가 물체와 너무 가까운 경우가 있다.
        # 이때 start가 inflated obstacle margin 안쪽이면 일반 경로계획이 바로 실패하므로,
        # 실제 물체 내부가 아닌 경우에 한해 먼저 safe_start로 빠져나간 뒤 경로계획한다.
        if getattr(config, "START_MARGIN_ESCAPE_ENABLE", True) and len(obstacle_polygon) >= 3:
            safe_start = self._find_start_escape_point(start, goal, obstacle_polygon)
            if safe_start is not None and np.linalg.norm(safe_start - start) > 1e-9:
                escape_start_used = True
                escape_start_xy = safe_start.copy()
                start = safe_start.copy()

        # costmap 갱신
        self._update_costmap(obstacle_polygon)

        start_ij = self._point_to_grid(start)
        goal_ij = self._point_to_grid(goal)
        start_blocked = start_ij in self.blocked_cells
        goal_blocked = goal_ij in self.blocked_cells

        # -------------------------------------------------
        # [수정] endpoint cell relaxation:
        # start/goal endpoint가 obstacle_margin 때문에만 막힌 경우는 허용한다.
        # 단, 실제 polygon 내부에 있는 endpoint는 절대 허용하지 않는다.
        #
        # 이유:
        # - pushing task에서는 push 시작점 근처 approach goal이 물체 margin 안에
        #   들어가는 것이 자연스러운 경우가 많다.
        # - 다음 step 시작 시 robot current 위치도 물체 margin 안쪽일 수 있다.
        # - 하지만 중간 waypoint/경로는 기존 margin 충돌 검사를 그대로 유지한다.
        # -------------------------------------------------
        start_relaxed = False
        start_relax_dist = None
        goal_relaxed = False
        goal_relax_dist = None

        if getattr(config, "ENDPOINT_MARGIN_RELAX_ENABLE", True):
            if start_blocked:
                start_relaxed, start_relax_dist = self._relax_endpoint_cell_if_margin_only(
                    start,
                    obstacle_polygon,
                    start_ij,
                )
                if start_relaxed:
                    start_blocked = False

            if goal_blocked:
                goal_relaxed, goal_relax_dist = self._relax_endpoint_cell_if_margin_only(
                    goal,
                    obstacle_polygon,
                    goal_ij,
                )
                if goal_relaxed:
                    goal_blocked = False

        self.last_debug_info = self._make_base_debug_info(
            start=start,
            goal=goal,
            direct_attempted=direct_attempted,
            direct_ok=direct_ok,
            lshape_attempted=lshape_attempted,
            lshape_ok=lshape_ok,
            ushape_attempted=ushape_attempted,
            ushape_ok=ushape_ok,
            astar_attempted=astar_attempted,
            astar_ok=astar_ok,
            failure_reason=failure_reason,
            start_grid=start_ij,
            goal_grid=goal_ij,
            start_blocked=start_blocked,
            goal_blocked=goal_blocked,
        )
        self._set_last_debug_info(
            original_start_xy=tuple(float(v) for v in original_start),
            start_escape_used=bool(escape_start_used),
            escape_start_xy=None if escape_start_xy is None else tuple(float(v) for v in escape_start_xy),

            # [수정 디버그] endpoint margin relaxation 확인용
            endpoint_margin_relax_enable=bool(getattr(config, "ENDPOINT_MARGIN_RELAX_ENABLE", True)),
            start_relaxed=bool(start_relaxed),
            start_relax_dist=None if start_relax_dist is None else float(start_relax_dist),
            goal_relaxed=bool(goal_relaxed),
            goal_relax_dist=None if goal_relax_dist is None else float(goal_relax_dist),
        )

        # -------------------------------------------------
        # 1) 직선 경로
        # -------------------------------------------------
        direct_attempted = True
        if not segment_hits_polygon(start, goal, obstacle_polygon, margin=self.obstacle_margin):
            direct_ok = True
            self._set_last_debug_info(
                direct_attempted=direct_attempted,
                direct_ok=direct_ok,
                lshape_attempted=lshape_attempted,
                lshape_ok=lshape_ok,
                ushape_attempted=ushape_attempted,
                ushape_ok=ushape_ok,
                astar_attempted=astar_attempted,
                astar_ok=astar_ok,
                failure_reason=failure_reason,
                selected_path_type="direct",
            )
            self._print_last_debug_info()
            return self._with_original_start_if_needed([start.copy(), goal.copy()], original_start, escape_start_used)

        # -------------------------------------------------
        # 2) L자 경로
        # -------------------------------------------------
        lshape_attempted = True
        l_path = self._try_l_shape_paths(start, goal, obstacle_polygon)
        if l_path is not None:
            lshape_ok = True
            self._set_last_debug_info(
                direct_attempted=direct_attempted,
                direct_ok=direct_ok,
                lshape_attempted=lshape_attempted,
                lshape_ok=lshape_ok,
                ushape_attempted=ushape_attempted,
                ushape_ok=ushape_ok,
                astar_attempted=astar_attempted,
                astar_ok=astar_ok,
                failure_reason=failure_reason,
                selected_path_type="l_shape",
            )
            self._print_last_debug_info()
            return self._with_original_start_if_needed(l_path, original_start, escape_start_used)

        # -------------------------------------------------
        # 3) ㄷ자 경로
        # -------------------------------------------------
        ushape_attempted = True
        u_path = self._try_u_shape_paths(start, goal, obstacle_polygon)
        if u_path is not None:
            ushape_ok = True
            self._set_last_debug_info(
                direct_attempted=direct_attempted,
                direct_ok=direct_ok,
                lshape_attempted=lshape_attempted,
                lshape_ok=lshape_ok,
                ushape_attempted=ushape_attempted,
                ushape_ok=ushape_ok,
                astar_attempted=astar_attempted,
                astar_ok=astar_ok,
                failure_reason=failure_reason,
                selected_path_type="u_shape",
            )
            self._print_last_debug_info()
            return self._with_original_start_if_needed(u_path, original_start, escape_start_used)

        # -------------------------------------------------
        # 4) 마지막으로 A*
        # -------------------------------------------------
        astar_attempted = True
        self._set_last_debug_info(
            direct_attempted=direct_attempted,
            direct_ok=direct_ok,
            lshape_attempted=lshape_attempted,
            lshape_ok=lshape_ok,
            ushape_attempted=ushape_attempted,
            ushape_ok=ushape_ok,
            astar_attempted=astar_attempted,
            astar_ok=astar_ok,
            failure_reason=failure_reason,
            selected_path_type=None,
        )

        path = self._astar_path(start, goal)
        astar_ok = path is not None

        if path is None:
            if start_blocked:
                failure_reason = "start_cell_blocked"
            elif goal_blocked:
                failure_reason = "goal_cell_blocked"
            else:
                failure_reason = "no_path_found"
        else:
            failure_reason = None

        self._set_last_debug_info(
            direct_attempted=direct_attempted,
            direct_ok=direct_ok,
            lshape_attempted=lshape_attempted,
            lshape_ok=lshape_ok,
            ushape_attempted=ushape_attempted,
            ushape_ok=ushape_ok,
            astar_attempted=astar_attempted,
            astar_ok=astar_ok,
            failure_reason=failure_reason,
            selected_path_type="astar" if path is not None else None,
        )
        self._print_last_debug_info()
        return self._with_original_start_if_needed(path, original_start, escape_start_used)

    def _astar_path(self, start: np.ndarray, goal: np.ndarray) -> Optional[List[np.ndarray]]:
        # 클래식한 A* 길찾기 알고리즘
        with perf_timer("motion_planner._astar_path", every=config.PERF_LOG_ASTAR_EVERY):
            start_ij = self._point_to_grid(start)
            goal_ij = self._point_to_grid(goal)
            expanded_count = 0

            if goal_ij in self.blocked_cells:
                self._set_last_debug_info(visited_count=0, expanded_count=0)
                return None

            open_heap = [(0.0, start_ij)]
            came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
            g_cost = {start_ij: 0.0}

            neigh = [
                (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
            ]

            def heuristic(a, b):
                return float(np.linalg.norm(np.array(a) - np.array(b)))

            visited = set()

            while open_heap:
                _, current = heapq.heappop(open_heap)
                if current in visited:
                    continue
                visited.add(current)
                expanded_count += 1

                if heuristic(current, goal_ij) <= 1.5:
                    self._set_last_debug_info(visited_count=len(visited), expanded_count=expanded_count)
                    return self._reconstruct(came_from, current, start, goal)

                for di, dj, cost_mult in neigh:
                    nxt = (current[0] + di, current[1] + dj)
                    nxt_pt = self._grid_to_point(nxt)

                    if not self._inside_bounds(nxt_pt):
                        continue
                    if nxt in self.blocked_cells:
                        continue

                    cand = g_cost[current] + cost_mult

                    if cand < g_cost.get(nxt, float("inf")):
                        g_cost[nxt] = cand
                        came_from[nxt] = current
                        f = cand + heuristic(nxt, goal_ij)
                        heapq.heappush(open_heap, (f, nxt))

            self._set_last_debug_info(visited_count=len(visited), expanded_count=expanded_count)
            return None

    def _reconstruct(self, came_from, current, start, goal):
        # A* 탐색 종료 후 실제 궤적 복원
        pts = [goal.copy()]
        c = current
        while c in came_from:
            pts.append(self._grid_to_point(c))
            c = came_from[c]
        pts.append(start.copy())
        pts.reverse()

        return self._compress_path(pts)

    @staticmethod
    def _compress_path(path: List[np.ndarray]) -> List[np.ndarray]:
        # 같은 직선상에 있는 점들을 쳐내서 경로 단순화
        if len(path) <= 2:
            return path

        out = [path[0]]
        for i in range(1, len(path) - 1):
            a = np.asarray(out[-1])
            b = np.asarray(path[i])
            c = np.asarray(path[i + 1])
            v1 = b - a
            v2 = c - b

            if np.linalg.norm(v1) < 1e-9 or np.linalg.norm(v2) < 1e-9:
                continue

            cross = abs(v1[0] * v2[1] - v1[1] * v2[0])

            if cross > 1e-6:
                out.append(b)

        out.append(path[-1])
        return out

    def _line_of_sight_clear(self, a, b, obstacle_polygon) -> bool:
        """
        a -> b를 한 번에 직선 이동해도 되는지 검사.
        A* path smoothing에서 중간 waypoint를 건너뛰기 위한 충돌 검사.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        if not self._inside_bounds(a):
            return False
        if not self._inside_bounds(b):
            return False

        obstacle_polygon = self._sanitize_obstacle_polygon(obstacle_polygon)
        if len(obstacle_polygon) < 3:
            return True

        return not segment_hits_polygon(
            a,
            b,
            obstacle_polygon,
            margin=self.obstacle_margin,
        )

    def _smooth_path_line_of_sight(self, path: List[np.ndarray], obstacle_polygon) -> List[np.ndarray]:
        """
        A*나 ㄷ자/L자 path에서, 중간 waypoint를 건너뛰어도
        obstacle+margin과 충돌하지 않으면 waypoint를 줄인다.

        직선 path는 보통 2개, L자 path는 보통 3개, ㄷ자는 보통 4개라
        너무 긴 A* path만 주로 줄어든다.
        """
        if path is None:
            return None
        if len(path) <= 2:
            return path

        path = [np.asarray(p, dtype=float).copy() for p in path]

        smoothed = [path[0]]
        i = 0
        n = len(path)

        while i < n - 1:
            best_j = i + 1

            # 현재 점에서 가능한 한 먼 waypoint까지 직선으로 갈 수 있는지 뒤에서부터 검사
            for j in range(n - 1, i, -1):
                if self._line_of_sight_clear(path[i], path[j], obstacle_polygon):
                    best_j = j
                    break

            smoothed.append(path[best_j])
            i = best_j

        return smoothed

    def attach_approach(self, robot_xy, start_xy, end_xy, n_hat, obstacle_polygon, retreat_distance=0.03):
        # 찌르기 액션에 우회 경로를 포장해서 반환
        with perf_timer("motion_planner.attach_approach", every=config.PERF_LOG_PATH_EVERY):
            robot_xy = np.asarray(robot_xy, dtype=float)
            start_xy = np.asarray(start_xy, dtype=float)
            end_xy = np.asarray(end_xy, dtype=float)
            n_hat = np.asarray(n_hat, dtype=float)

            approach_xy = start_xy + self.obstacle_margin * 1.0 * n_hat
            retreat_xy = end_xy + retreat_distance * n_hat

            # start/end/approach/retreat 중 하나라도 workspace 밖이면 이 액션은 탈락
            if not self._inside_bounds(start_xy):
                self._set_last_debug_info(failure_reason="push_start_out_of_bounds")
                self._print_last_debug_info()
                return None

            if not self._inside_bounds(end_xy):
                self._set_last_debug_info(failure_reason="push_end_out_of_bounds")
                self._print_last_debug_info()
                return None

            if not self._inside_bounds(approach_xy):
                self._set_last_debug_info(failure_reason="approach_out_of_bounds")
                self._print_last_debug_info()
                return None

            if not self._inside_bounds(retreat_xy):
                self._set_last_debug_info(failure_reason="retreat_out_of_bounds")
                self._print_last_debug_info()
                return None

            # 현재 위치에서 접근점까지의 경로 탐색
            # 순서: 직선 -> L자 -> ㄷ자 -> A*
            path = self.plan_path(robot_xy, approach_xy, obstacle_polygon)

            self._set_last_debug_info(
                approach_xy=tuple(float(v) for v in approach_xy),
                push_start=tuple(float(v) for v in start_xy),
                push_end=tuple(float(v) for v in end_xy),
                retreat_xy=tuple(float(v) for v in retreat_xy),
            )

            if path is None:
                return None

            # -------------------------------------------------
            # Path smoothing
            # -------------------------------------------------
            raw_waypoints = int(len(path))
            path = self._smooth_path_line_of_sight(path, obstacle_polygon)
            smoothed_waypoints = int(len(path))

            self._set_last_debug_info(
                raw_approach_waypoints=raw_waypoints,
                smoothed_approach_waypoints=smoothed_waypoints,
            )

            max_wps = getattr(config, "MAX_APPROACH_WAYPOINTS", None)
            if max_wps is not None and len(path) > int(max_wps):
                self._set_last_debug_info(
                    failure_reason="too_many_approach_waypoints",
                    approach_waypoints=int(len(path)),
                    max_approach_waypoints=int(max_wps),
                    raw_approach_waypoints=raw_waypoints,
                    smoothed_approach_waypoints=smoothed_waypoints,
                )
                self._print_last_debug_info()
                return None

            return {
                "approach_path": path,
                "approach_length": path_length(path),
                "approach_xy": approach_xy,
                "push_start": start_xy,
                "push_end": end_xy,
                "retreat_xy": retreat_xy,
            }
