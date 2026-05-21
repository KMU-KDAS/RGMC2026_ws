"""
Simple pygame-based simulator for the heuristic planner.
(휴리스틱 플래너를 시각적으로 테스트하기 위한 2D 파이게임 시뮬레이터입니다.)
"""

import sys # 프로그램 강제 종료(sys.exit)를 위해 사용
import time # 성능 측정(시간 계산)을 위해 사용
from pathlib import Path
from typing import List # 타입 힌트

# =========================================================
# Path setup
# =========================================================
# 이 파일을 PowerShell에서 직접 실행해도
# brain.py 내부의 `from task1 import config`가 정상 동작하도록
# 프로젝트의 src 폴더와 src/task1 폴더를 sys.path에 추가한다.
_THIS_FILE = Path(__file__).resolve()
TASK1_DIR = _THIS_FILE.parent
SRC_ROOT = TASK1_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent

for _path in (str(SRC_ROOT), str(TASK1_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("SRC_ROOT    :", SRC_ROOT)
print("TASK1_DIR   :", TASK1_DIR)

import numpy as np # 행렬/벡터 연산

import config # 각종 환경설정 값
from brain import PushingBrain # 뇌(플래너) 객체
# 기하학 계산 및 다각형 생성 헬퍼 함수들
from geometry_utils import get_global_corners, make_circle_polygon, shape_alignment_error, wrap_angle
from planner import Planner # IoU(성공률) 심판 객체
from total_ import simulate_push_stroke_normalized, warmup_numba_simulation # 물리 엔진 시뮬레이터

try:
    import pygame # 2D 게임/그래픽 라이브러리 임포트 시도
except ImportError:
    pygame = None # 안 깔려있으면 예외 처리 (아래에서 안내 메시지 출력용)

# 시뮬레이터 창의 가로, 세로 픽셀 해상도입니다. (900x900)
WIDTH, HEIGHT = 900, 900


def to_screen(x_norm, y_norm):
    # 0.0 ~ 1.0 사이로 된 정규화 좌표를 실제 900x900 화면의 픽셀 좌표로 변환합니다.
    screen_x = int(x_norm * WIDTH)
    # Pygame은 모니터 맨 위가 y=0 이므로, 직관적인 물리 좌표계(맨 아래가 0)를 위해 y축을 뒤집어줍니다.
    screen_y = int(HEIGHT - (y_norm * HEIGHT))
    return screen_x, screen_y


def render_path(screen, path: List[np.ndarray], color=(255, 50, 255)):
    # A* 플래너가 만든 우회 경로(path)를 화면에 예쁜 핑크색 선과 점으로 그리는 함수입니다.
    if len(path) < 2:
        return # 점이 2개 미만이면 선을 그릴 수 없으니 패스
    # 경로상의 모든 점을 픽셀 좌표로 변환
    pts = [to_screen(p[0], p[1]) for p in path]
    # 점들을 선(lines)으로 이어서 그립니다. (False: 시작점과 끝점을 닫지 않음)
    pygame.draw.lines(screen, color, False, pts, 2)
    # 꺾이는 지점마다 작은 동그라미(circle)를 그려서 잘 보이게 합니다.
    for pt in pts:
        pygame.draw.circle(screen, color, pt, 4)


def get_shape_polygons(shape_type, current_pose, target_pose, local_corners, radius):
    # 화면에 물체를 그리기 위해, 현재 위치와 목표 위치를 바탕으로 다각형 꼭짓점 리스트를 뽑아옵니다.
    cur_poly_m = (
        make_circle_polygon(current_pose[:2], radius, config.CIRCLE_POLY_POINTS)
        if shape_type == "WEIGHTED_CIRCLE"
        else get_global_corners(current_pose, local_corners)
    )
    tgt_poly_m = (
        make_circle_polygon(target_pose[:2], radius, config.CIRCLE_POLY_POINTS)
        if shape_type == "WEIGHTED_CIRCLE"
        else get_global_corners(target_pose, local_corners)
    )
    return cur_poly_m, tgt_poly_m


def run_auto_pygame():
    # 파이게임 시뮬레이터를 켜는 메인 함수
    if pygame is None:
        raise RuntimeError("pygame is not installed. Run `pip install pygame` first.")

    # 첫 실행 시 버벅거리지 않도록 물리 엔진을 강제로 한 번 공회전시킵니다.
    warmup_numba_simulation()

    pygame.init() # 파이게임 엔진 시동
    screen = pygame.display.set_mode((WIDTH, HEIGHT)) # 900x900 창 생성
    pygame.display.set_caption("CloudGripper Auto-Pilot with Path Planning") # 창 이름 설정
    
    # 화면에 글씨를 쓰기 위한 폰트 설정 (Consolas 굵은 글씨)
    font = pygame.font.SysFont("consolas", 18, bold=True)
    large_font = pygame.font.SysFont("consolas", 28, bold=True)
    
    # 1초에 프레임을 몇 번 그릴지 제어하는 시계(Clock) 객체
    clock = pygame.time.Clock()

    # 인공지능 참모진들 생성
    planner = Planner(config) # IoU 심판
    shape_type = "WEIGHTED_SQUARE" # 시작 기본 물체는 정사각형
    brain = PushingBrain() # 두뇌(플래너) 생성

    # 물체의 시작 위치, 목표 위치 등을 랜덤으로 세팅
    current_pose, target_pose, local_corners, radius = config.reset_scenario(shape_type, np.random.default_rng(0))
    
    # 시뮬레이터 상에서 이 물체가 '진짜로(True)' 어떤 무게중심 가설을 따르는지 첫 번째 가설로 세팅
    true_case = list(config.SHAPE_DB[shape_type]["cases"].keys())[0]
    # 로봇의 시작 위치 세팅
    robot_pose = [float(config.ROBOT_START_XY[0]), float(config.ROBOT_START_XY[1])]
    
    # 시뮬레이터 상태 제어 변수들
    auto_mode = False # 스페이스바를 누르면 켜짐 (스스로 플래닝하고 움직임)
    is_success = False # 목표에 도달했는지 여부
    last_plan = None # 화면에 그리기 위해 직전에 짠 계획을 저장
    pending_plan = None # 아직 실행하지 않고 대기 중인 계획
    is_executing_plan = False # 지금 로봇이 움직이는 중인지 여부
    total_steps = 0 # 로봇이 물체를 민 횟수
    total_rotation_deg = 0.0 # 물체가 뱅글뱅글 돈 총 각도 (SysID 관찰용)
    planner_status = "IDLE" # 뇌가 지금 대기 중인지, 계산 중인지 상태 텍스트
    was_success_last_frame = False 
    last_tick_log_state = None # 이전 프레임의 로그 상태 기억

    # ========================================================
    # ★ 요청하신 대로 로그를 전부 꺼버렸습니다! (입막음 완료) ★
    # ========================================================
    def user_log(message):
        pass # print(f"[AUTO] {message}", flush=True) 

    def verbose_log(message):
        pass # 원래는 세세한 디버그를 띄우는 용도지만 막았습니다.

    def set_pending_plan(value, reason):
        nonlocal pending_plan
        if pending_plan is value or ((pending_plan is None) == (value is None)):
            pending_plan = value
            return
        pending_plan = value
        verbose_log(f"pending_plan -> {'SET' if pending_plan is not None else 'NONE'} ({reason})")

    def set_is_executing(value, reason):
        nonlocal is_executing_plan
        value = bool(value)
        if is_executing_plan == value:
            is_executing_plan = value
            return
        is_executing_plan = value
        verbose_log(f"is_executing_plan -> {is_executing_plan} ({reason})")

    def set_planner_status(value, reason):
        nonlocal planner_status
        if planner_status == value:
            planner_status = value
            return
        planner_status = value
        verbose_log(f"planner_status -> {planner_status} ({reason})")

    def reset_runtime_state(reason, status="IDLE"):
        nonlocal last_tick_log_state
        set_pending_plan(None, reason)
        set_is_executing(False, reason)
        set_planner_status(status, reason)
        last_tick_log_state = None

    def log_tick_state(force=False):
        pass # 프레임마다 현재 상태를 도배하는 로그도 꺼버렸습니다.

    # ========================================================
    # 시뮬레이터 무한 루프(프레임 렌더링) 시작
    # ========================================================
    while True:
        # 사용자의 키보드 입력 및 종료 이벤트 감지
        for event in pygame.event.get():
            # X 버튼 누르면 프로그램 깔끔하게 종료
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # 키보드를 누르는 이벤트가 아니면 무시
            if event.type != pygame.KEYDOWN:
                continue

            # [SPACE BAR]: 오토파일럿(자동 주행) 모드 껐다 켜기
            if event.key == pygame.K_SPACE:
                user_log(f"SPACE keydown detected (is_success={is_success}, auto_mode={auto_mode})")
                if is_success:
                    user_log("SPACE ignored because success condition is already met")
                else:
                    auto_mode = not auto_mode # 모드 토글(Toggle)
                    user_log(f"auto_mode toggled -> {auto_mode}")
                    if auto_mode:
                        reset_runtime_state("auto_mode turned on")
                    else:
                        reset_runtime_state("auto_mode turned off")

            # [1, 2, 3 숫자키]: 1=사각형, 2=원형, 3=T자형 물체로 변경
            if event.key in [pygame.K_1, pygame.K_2, pygame.K_3]:
                shape_type = {
                    pygame.K_1: "WEIGHTED_SQUARE",
                    pygame.K_2: "WEIGHTED_CIRCLE",
                    pygame.K_3: "T_BASE",
                }[event.key]
                # 물체 모양이 바뀌었으니 시나리오를 다시 랜덤으로 세팅
                current_pose, target_pose, local_corners, radius = config.reset_scenario(shape_type, np.random.default_rng())
                brain = PushingBrain() # 뇌도 포맷
                true_case = list(config.SHAPE_DB[shape_type]["cases"].keys())[0] # 가설 초기화
                robot_pose = [float(config.ROBOT_START_XY[0]), float(config.ROBOT_START_XY[1])] # 로봇 원위치
                auto_mode = False # 자동모드 끔
                is_success = False
                last_plan = None
                reset_runtime_state("shape changed")
                total_steps = 0
                total_rotation_deg = 0.0

            # [R 키]: 현재 모양 그대로 위치만 완전히 새롭게 리셋(랜덤 배치)
            if event.key == pygame.K_r:
                current_pose, target_pose, local_corners, radius = config.reset_scenario(shape_type, np.random.default_rng())
                brain = PushingBrain()
                robot_pose = [float(config.ROBOT_START_XY[0]), float(config.ROBOT_START_XY[1])]
                auto_mode = False
                is_success = False
                last_plan = None
                reset_runtime_state("scenario reset")
                total_steps = 0
                total_rotation_deg = 0.0

            # [C 키]: 현재 물체의 '진짜 무게중심(True Case)'을 다른 것으로 슬쩍 바꿈
            # 로봇이 밀었을 때 예상과 다르게 움직이게 만들어서 뇌(SysID)가 잘 적응하는지 테스트하는 용도
            if event.key == pygame.K_c:
                cases = list(config.SHAPE_DB[shape_type]["cases"].keys())
                idx = cases.index(true_case)
                true_case = cases[(idx + 1) % len(cases)] # 다음 케이스로 변경

        # 현재 턴의 물체와 목표 위치의 픽셀 좌표 다각형 획득
        cur_poly_m, tgt_poly_m = get_shape_polygons(shape_type, current_pose, target_pose, local_corners, radius)
        
        # 현재 얼마나 겹치는지(IoU) 검사
        current_iou = planner.calculate_iou(
            planner.world_to_mask(cur_poly_m),
            planner.world_to_mask(tgt_poly_m),
            planner.mask_shape,
        )
        
        # 렌더링에 띄워주기 위해 모양 오차 계산
        current_shape_err = shape_alignment_error(shape_type, current_pose, target_pose, local_corners, radius)
        
        # IoU가 커트라인을 넘으면 성공!
        is_success = current_iou >= config.IOU_SUCCESS_THRES

        # 성공했으면 모든 계획/실행을 초기화하고 대기
        if is_success:
            reset_runtime_state("success reached", status="SUCCESS")
            
        # 성공 전이고, 자동 주행 모드(SPACE ON)라면?
        elif auto_mode:
            # 꼬여있는 플래그 상태를 안전하게 복구하는 로직들
            if is_executing_plan and pending_plan is None:
                reset_runtime_state("stale executing flag recovered")
            elif pending_plan is not None and not isinstance(pending_plan, dict):
                reset_runtime_state("invalid pending plan recovered")

            log_tick_state()

            # 1. 만들어진 계획도 없고 로봇도 안 움직이고 있다면 -> 뇌를 가동해서 플랜을 짜라!
            if pending_plan is None and not is_executing_plan:
                set_planner_status("PLANNING", "planning cycle start")
                user_log("planning triggered")
                plan_start = time.perf_counter()
                
                # 뇌가 수만 가지 경우의 수를 계산해서 최고의 플랜 하나를 찾아옴
                plan = brain.get_best_plan(
                    shape_type,
                    current_pose,
                    target_pose,
                    local_corners,
                    planner=planner,
                    radius=radius,
                    robot_pose=robot_pose,
                )
                
                plan_elapsed_ms = (time.perf_counter() - plan_start) * 1000.0
                user_log(f"planning done, plan_none={plan is None}")
                verbose_log(f"planning end ({plan_elapsed_ms:.1f} ms)")
                
                # 찾아온 플랜을 '대기 중인 계획(pending_plan)'으로 세팅
                set_pending_plan(plan, "planning finished")
                
                if plan is None:
                    # 각이 안 나와서 길을 못 찾았을 경우
                    last_plan = None
                    verbose_log("planning returned None; auto_mode stays ON for the next planning cycle")
                    set_planner_status("NO PLAN", "planning returned None")
                else:
                    last_plan = plan
                    set_planner_status("READY", "plan queued for execution")
                    
            # 2. 짜놓은 계획이 대기 중이고, 로봇은 쉬고 있다면 -> 물리 엔진을 돌려서 계획을 진짜로 실행해라!
            elif pending_plan is not None and not is_executing_plan:
                user_log("executing plan")
                set_planner_status("EXECUTING", "plan execution start")
                set_is_executing(True, "plan execution start")
                
                plan = pending_plan
                cand = plan["candidate"] # 찌르기 액션 정보
                motion = plan["motion"] # A* 길찾기 경로 정보
                
                # 시뮬레이터 상의 '진짜(True)' 물리 속성을 가져옴
                true_case_info = config.SHAPE_DB[shape_type]["cases"][true_case]
                pre_push_pose = list(current_pose) # 밀기 전 위치 백업
                
                try:
                    # 진짜 물리 엔진에 넣어서 로봇이 밀었을 때 물체가 어떻게 이동할지(텔레포트) 시뮬레이션함!
                    _, _, theta_next, next_body_center_norm = simulate_push_stroke_normalized(
                        m=true_case_info["m"],
                        i_body=true_case_info["I"],
                        com_x=true_case_info["com"][0],
                        com_y=true_case_info["com"][1],
                        body_center_norm=np.array(current_pose[:2], dtype=float),
                        theta=current_pose[2],
                        v=np.array([0.0, 0.0]),
                        w=0.0,
                        start_x_norm=cand["start"][0],
                        start_y_norm=cand["start"][1],
                        end_x_norm=cand["end"][0],
                        end_y_norm=cand["end"][1],
                        n_hat=cand["n_hat"],
                        t_hat=cand["t_hat"],
                        config_dict=config.__dict__,
                        eff_r=config.SHAPE_DB[shape_type]["eff_r"],
                    )
                    total_steps += 1 # 한 번 밀었으니 카운트 상승
                    
                    # 물체가 회전한 각도를 계산해 통계에 더함
                    rot_diff_rad = abs(wrap_angle(theta_next - pre_push_pose[2]))
                    total_rotation_deg += float(np.degrees(rot_diff_rad))
                    
                    # 시뮬레이터 상의 현재 물체 위치를, 밀리고 난 뒤의 새 위치로 덮어씌움(텔레포트)
                    current_pose = [float(next_body_center_norm[0]), float(next_body_center_norm[1]), float(theta_next)]
                    # 로봇 위치도 A* 우회를 끝내고 접근점에 도착한 위치로 텔레포트
                    robot_pose = [float(motion["approach_xy"][0]), float(motion["approach_xy"][1])]
                    
                    # ★ SysID 학습! 방금 물체가 밀린 결과를 뇌에 보고해서 무게중심 가설을 교정하게 함
                    brain.update_com_belief(theta_next - pre_push_pose[2], shape_type, pre_push_pose, cand)
                    
                    # (팁: 로봇이 순식간에 골인하는 걸 방지하고 과정을 지켜보고 싶다면 여기에 time.sleep(0.3)을 추가하면 됩니다)
                finally:
                    # 실행이 끝났으니 다시 플랜 짤 수 있도록 상태 초기화
                    reset_runtime_state("execution finished")
                    verbose_log(
                        "execution reset complete "
                        f"(pending_plan={pending_plan is not None}, is_executing_plan={is_executing_plan})"
                    )
        else:
            # 자동모드가 꺼져있으면 IDLE(대기) 상태 표출
            set_planner_status("IDLE", "manual idle")
            last_tick_log_state = None

        was_success_last_frame = is_success

        # ========================================================
        # 화면 그리기 (렌더링) 파트
        # ========================================================
        screen.fill((30, 35, 40)) # 배경을 진한 남색/회색으로 칠함
        
        # 1. 다각형 그리기 (픽셀 변환)
        tgt_pts = [to_screen(px, py) for px, py in tgt_poly_m] # 목표 다각형 픽셀
        cur_pts = [to_screen(px, py) for px, py in cur_poly_m] # 현재 다각형 픽셀
        
        # 목표 지점은 흐릿한 초록색으로 칠함
        pygame.draw.polygon(screen, (30, 80, 30), tgt_pts, 0) # 속 채우기
        pygame.draw.polygon(screen, (60, 180, 60), tgt_pts, 2) # 테두리 두께 2
        
        # 현재 물체는 기본 보라색, 만약 성공(is_success)하면 황금색으로 빛나게 그림!
        pygame.draw.polygon(screen, (70, 130, 200) if not is_success else (255, 215, 0), cur_pts, 0)
        pygame.draw.polygon(screen, (255, 255, 255), cur_pts, 2) # 흰색 테두리
        
        # 2. 로봇 그리기 (주황색 동그라미)
        radius_px = max(4, int(config.PUSHER_RADIUS * WIDTH))
        pygame.draw.circle(screen, (240, 120, 60), to_screen(robot_pose[0], robot_pose[1]), radius_px)
        
        # 3. 방금 세운 플랜(A* 길과 찌르기 선) 핑크색으로 그리기
        if last_plan is not None:
            render_path(screen, last_plan["motion"]["approach_path"])
            pygame.draw.line(
                screen,
                (255, 50, 255),
                to_screen(*last_plan["candidate"]["start"]),
                to_screen(*last_plan["candidate"]["end"]),
                4, # 찌르기 선은 테두리 4로 두껍게
            )

        # ========================================================
        # UI 텍스트(글자) 정보 그리기
        # ========================================================
        belief = brain.com_belief[shape_type] # 현재 로봇이 믿는 무게중심
        mode_text = "HEURISTIC PLANNER"
        mode_color = (200, 200, 200)

        # 오토파일럿 상태 텍스트
        screen.blit(
            font.render(
                f"AUTOPILOT: {'ON' if auto_mode else 'OFF'}  [SPACE]",
                True,
                (255, 50, 255) if auto_mode else (200, 200, 200),
            ),
            (20, 20),
        )
        
        # 물체 정보 텍스트 (모양, 진짜 무게중심, 믿고있는 무게중심)
        screen.blit(font.render(f"shape={shape_type}  true_case={true_case}  belief={belief}", True, (255, 255, 255)), (20, 50))
        # 뇌 종류 표기
        screen.blit(font.render(f"Brain: {mode_text}", True, mode_color), (20, 80))
        # 플래너 상태 표기 (PLANNING, EXECUTING 등)
        screen.blit(font.render(f"Planner State: {planner_status}", True, (180, 220, 255)), (20, 110))
        # 조작키 설명
        screen.blit(font.render("[1]Sq [2]Cir [3]T | [R]eset [C]ase", True, (180, 255, 180)), (20, 140))
        
        # ★ 큰 글씨로 현재 IoU 퍼센트 표기 (성공 시 황금색)
        screen.blit(
            large_font.render(
                f"IoU: {current_iou * 100:5.1f}%",
                True,
                (255, 215, 0) if is_success else (255, 255, 255),
            ),
            (20, 180),
        )
        
        # 모양 오차 디버그용 숫자
        screen.blit(
            font.render(
                f"Shape Err(debug): {current_shape_err:6.4f}",
                True,
                (190, 190, 190),
            ),
            (20, 220),
        )
        
        # 총 밀기 횟수 및 회전 각도 표기
        screen.blit(font.render(f"Total Steps: {total_steps}", True, (255, 255, 255)), (20, 250))
        screen.blit(font.render(f"Total Rot : {total_rotation_deg:6.1f} deg", True, (255, 255, 255)), (20, 280))
        
        pygame.display.flip() # 지금까지 도화지에 그린 것들을 모니터 화면에 확 뿌림! (업데이트)
        clock.tick(60) # 1초에 60번(60FPS) 이상 돌지 않도록 속도를 제어함


if __name__ == "__main__":
    run_auto_pygame() # 스크립트 실행