import os # 운영체제 환경변수(토큰 등)를 읽어오기 위한 라이브러리
import time # 대기 시간(sleep)을 주거나 시스템 시간을 읽어올 때 사용

import config # 전체 설정값
from brain import PushingBrain # 우리가 분석했던 뇌(플래너 로직) 클래스
from geometry_utils import get_global_corners, make_circle_polygon # 꼭짓점 변환 및 원 생성 헬퍼 함수
from planner import Planner # IoU(겹치는 면적)를 계산해주는 심판 클래스
from robot_executor import CloudGripperExecutor # 뇌의 계획을 실제 로봇 하드웨어 명령으로 바꿔주는 실행기
from total_ import warmup_numba_simulation # 첫 실행 시 물리 엔진 컴파일 렉을 없애주는 웜업 함수

try:
    # 1. 실제 대회용 로봇 환경(CloudGripper) 클라이언트 라이브러리를 임포트 시도합니다.
    from client.cloudgripper_client_mock import GripperRobotMock
except ImportError:
    # 2. 만약 해당 라이브러리가 없다면? (내 컴퓨터에서 테스트 중일 때)
    # 에러를 내고 뻗지 않도록, 실제 로봇인 척 연기하는 '가짜(Mock) 로봇 클래스'를 즉석에서 정의합니다.
    class GripperRobotMock:
        def __init__(self, robot_name, token):
            self.robot_name = robot_name # 로봇 이름
            self.token = token # 인증 토큰
            # 로봇의 현재 위치(x, y)를 저장할 가짜 내부 상태 변수. 처음엔 (0.1, 0.1)에 있다고 가정합니다.
            self._state = {"x_norm": 0.1, "y_norm": 0.1}

        def eval_start(self):
            # 주최측 서버에 평가 시작을 알리는 가짜 함수 (무조건 성공 True 반환)
            return True

        def get_state(self):
            # 로봇의 현재 좌표 딕셔너리와, 현재 시간(타임스탬프)을 반환합니다.
            return dict(self._state), time.time()

        def move_xy(self, x_norm, y_norm):
            # 로봇아 (x, y)로 이동해! 라는 명령을 받으면, 
            # 실제 모터를 돌리는 대신 가짜 상태 변수의 좌표만 해당 위치로 업데이트합니다.
            self._state["x_norm"] = float(x_norm)
            self._state["y_norm"] = float(y_norm)
            return True


def get_perception_data(robot):
    """
    원래는 카메라로 사진을 찍고 딥러닝 모델(YOLO 등)을 돌려 물체의 위치를 파악해야 하는 함수입니다.
    현재는 비전 코드가 안 붙어있어서, 하드코딩된 '가짜 물체 위치'와 '가짜 목표 위치'를 반환하는 뼈대(Stub) 상태입니다.
    """
    # 물체의 현재 위치 [x, y, 각도]
    current_pose = [0.4, 0.4, 0.0]
    # 물체를 밀어넣어야 할 목표 위치 [x, y, 각도] (1.57라디안 = 약 90도)
    target_pose = [0.7, 0.7, 1.57]
    return current_pose, target_pose


def run_task1():
    # 진짜 메인 실행 함수 시작!
    print("=== CloudGripper Task 1 test start ===")
    
    # Numba JIT 컴파일러의 첫 실행 속도 저하를 막기 위해, 의미 없는 데이터를 넣어 미리 컴파일(웜업) 시킵니다.
    warmup_numba_simulation()

    # 컴퓨터 환경 변수에서 로봇 통신용 토큰을 읽어옵니다. (없으면 "test-token" 사용)
    token = os.environ.get("CLOUDGRIPPER_TOKEN", "test-token")
    # 로봇 통신 객체를 만듭니다. (실제 라이브러리가 없으면 위에서 만든 가짜 Mock 객체가 들어갑니다)
    robot = GripperRobotMock("robot6", token)
    # 로봇 API를 한 겹 감싸서 안전하게 명령을 내려줄 Executor(실행기) 객체 생성
    executor = CloudGripperExecutor(robot)

    # IoU(성공 여부)를 계산해 줄 Planner 심판 객체 생성
    planner = Planner(config)
    # 오늘 우리가 다룰 물체의 종류를 '무게가 한쪽으로 쏠린 사각형'으로 고정 설정
    shape_type = "WEIGHTED_SQUARE"
    # 수많은 경우의 수를 계산해 최적의 찌르기 각도를 찾아낼 '두뇌' 객체 생성
    brain = PushingBrain()

    # 해당 물체의 꼭짓점 정보와 반지름(원일 경우)을 config 파일에서 가져옵니다.
    local_corners = config.get_shape_corners(shape_type)
    radius = config.get_shape_radius(shape_type)

    print("[Main] Requesting eval_start.")
    try:
        # 평가 시작 API 호출
        robot.eval_start()
    except Exception:
        # Mock 객체라 에러가 나면 그냥 무시하고 넘어갑니다.
        print("[Main] Mock environment detected, skipping eval_start.")

    step = 0 # 몇 번째 미는 동작인지 카운트하는 변수
    
    # ================= 무한 루프 시작 (목표 달성할 때까지 안 끝남) =================
    while True:
        step += 1
        print(f"\n--- [Step {step}] Planning next push ---")

        # 1. 로봇의 현재 상태(좌표 등)를 읽어옵니다.
        state, ts = robot.get_state()
        if state is None:
            # 통신 에러 등으로 상태를 못 읽어오면, 0.5초 쉬고 루프를 처음부터 다시 돕니다.
            print("[Main] Failed to read robot state. Retrying.")
            time.sleep(0.5)
            continue
        # 로봇 그리퍼(끝단)의 현재 x, y 좌표 저장
        robot_xy = [state["x_norm"], state["y_norm"]]

        # 2. 물체의 현재 위치와 목표 위치를 카메라(현재는 가짜 데이터)에서 받아옵니다.
        cur_pose, tgt_pose = get_perception_data(robot)

        # 3. 목표 달성 여부(IoU)를 검사하기 위해 픽셀 마스크를 만듭니다.
        if shape_type == "WEIGHTED_CIRCLE":
            # 원형이면 중심점과 반지름으로 원 다각형 생성
            cur_poly = make_circle_polygon(cur_pose[:2], radius, config.CIRCLE_POLY_POINTS)
            tgt_poly = make_circle_polygon(tgt_pose[:2], radius, config.CIRCLE_POLY_POINTS)
        else:
            # 다각형이면 꼭짓점들을 글로벌 좌표로 변환
            cur_poly = get_global_corners(cur_pose, local_corners)
            tgt_poly = get_global_corners(tgt_pose, local_corners)

        # 현재 모양(cur_poly)과 목표 모양(tgt_poly)을 픽셀 이미지로 렌더링해서, 얼마나 겹치는지(IoU) 계산합니다.
        current_iou = planner.calculate_iou(
            planner.world_to_mask(cur_poly),
            planner.world_to_mask(tgt_poly),
            planner.mask_shape,
        )

        # 현재 겹침 정도를 퍼센트로 환산하여 화면에 출력합니다.
        print(f"[Main] Current IoU: {current_iou * 100:.1f}% / Target: {config.IOU_SUCCESS_THRES * 100:.1f}%")

        # 4. 성공 판정!
        # 현재 IoU가 우리가 설정한 성공 커트라인(예: 96%) 이상이라면?
        if current_iou >= config.IOU_SUCCESS_THRES:
            print(f"[Success] Target IoU reached: {current_iou * 100:.1f}%")
            break # 미션 클리어! 무한 루프를 탈출하여 프로그램을 종료합니다.

        # 5. 아직 성공하지 못했다면, 뇌(Brain)를 가동해 다음 미는 동작(Plan)을 가져옵니다.
        plan_start = time.perf_counter() # 플래닝 시작 시간 기록
        
        # 수만 개의 경우의 수를 계산하는 그 엄청난 함수 호출!
        plan = brain.get_best_plan(
            shape_type=shape_type,
            current_pose=cur_pose,
            target_pose=tgt_pose,
            local_corners=local_corners,
            planner=planner,
            radius=radius,
            robot_pose=robot_xy,
        )
        
        # 플래닝이 끝난 후 걸린 시간(ms) 계산
        plan_elapsed_ms = (time.perf_counter() - plan_start) * 1000.0
        
        # 성능 로깅이 켜져 있으면, 뇌가 얼마나 빨리 짱구를 굴렸는지 화면에 출력
        if config.PERF_LOG:
            print(f"[PERF] main_task1.get_best_plan: {plan_elapsed_ms:.1f} ms")

        # 6. 뇌가 성공적으로 계획(Plan)을 만들어냈다면?
        if plan is not None:
            exec_start = time.perf_counter() # 실행 시작 시간 기록
            
            # 실행기(Executor)에게 계획을 넘겨주어 실제 로봇 모터를 움직이게(move_xy) 합니다.
            success = executor.execute_push(plan)
            
            exec_elapsed_ms = (time.perf_counter() - exec_start) * 1000.0
            
            # 물리적 이동 명령을 내리는 데 걸린 시간 출력
            if config.PERF_LOG:
                print(f"[PERF] main_task1.execute_push: {exec_elapsed_ms:.1f} ms")
                
            # 모터 제어 중 오류가 났다면 경고 메시지 출력
            if not success:
                print("[Main] Push execution failed. Waiting and retrying.")
                
        # 뇌가 길이나 찌를 각도를 도저히 못 찾아서 계획 도출에 실패(None)했다면?
        else:
            print("[Main] No valid plan found. Retrying until IoU target is reached.")
            time.sleep(1.0) # 1초 쉬고 나서 다시 사진 찍고 플래닝 재시도
            continue

        # 한 번 찌르는 동작(우회 접근 -> 찌르기 -> 후퇴)을 완료했으면,
        # 물체가 관성에 의해 굴러가는 것이 완전히 멈출 때까지 1.0초간 가만히 대기합니다. (안정화 딜레이)
        time.sleep(1.0)


# 파이썬 스크립트가 직접 실행되었을 때(import 된 게 아닐 때) run_task1() 메인 함수를 구동시키는 엔트리 포인트
if __name__ == "__main__":
    run_task1()