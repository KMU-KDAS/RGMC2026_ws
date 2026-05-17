import numpy as np

IOU_SUCCESS_C= 0.9
IOU_SUCCESS_S = 0.8
IOU_SUCCESS_T = 0.7
# measured robot push speed (m/s)
PUSH_SPEED_MPS = 0.0933   # 일단 측정값 넣기
MIN_PUSH_STEPS = 3


T_BASE_STROKE_SCALE = 1.0

T_ALLOWED_STROKES_FAR = [1.75, 1.9]
T_ALLOWED_STROKES_MID = [0.65, 0.95 ]
T_ALLOWED_STROKES_NEAR = [0.25, 0.55]

# 원/네모도 T자처럼 config에서 stroke 배율을 조절
# 이 값들은 절대 길이가 아니라 base_stroke에 곱해지는 배율임.
# 최종 stroke_len = min(base_stroke * 배율, MAX_STROKE_LEN * 배율)
SQUARE_ALLOWED_STROKES_FAR = [1.65, 1.85]
SQUARE_ALLOWED_STROKES_MID = [0.6, 0.95]
SQUARE_ALLOWED_STROKES_NEAR = [0.25, 0.55]

CIRCLE_ALLOWED_STROKES_FAR = [1.8, 2.0]
CIRCLE_ALLOWED_STROKES_MID = [0.7, 1.0]
CIRCLE_ALLOWED_STROKES_NEAR = [0.25, 0.55]
RETREAT_DISTANCE = 0.03
T_RETREAT_DISTANCE = 0.02
T_BASE_SCALE = 1.06
SQUARE_SCALE = 1.06
CIRCLE_SCALE = 1.06
# =========================================================
# 0. Physical workspace size in meters (물리적 작업 공간 크기)
# =========================================================
# 로봇이 실제로 움직이는 바닥 공간의 가로/세로 길이입니다. (단위: 미터)
# 0.12m = 12cm. 즉, 12cm x 12cm의 아주 작은 미세 조작 환경을 의미합니다.
REAL_WORKSPACE_SIZE_X = 0.15
REAL_WORKSPACE_SIZE_Y = 0.15
# 밀기 힘(allowed_strokes 배율)을 조절하기 위한 남은 거리 기준선
STROKE_DIST_FAR = 0.05 / REAL_WORKSPACE_SIZE_X 
STROKE_DIST_MID = 0.01 / REAL_WORKSPACE_SIZE_X 

# 이를 전체 공간 크기(15cm)로 나누어 0.0~1.0 사이의 비율(정규화)로 변환해 저장합니다.
MAX_STROKE_LEN = 0.03 / REAL_WORKSPACE_SIZE_X

# 목표에 거의 다 왔을 때 힘을 원래의 30%로 줄임
DIRECT_NEAR_GOAL_STROKE_SCALE = 0.5
# 아무리 짧게 밀어도 로봇 푸셔 반지름의 75%보다는 길게 밀어야 함 (찔끔거림 방지)
PUSHER_RADIUS = 0.0055  / REAL_WORKSPACE_SIZE_X
DIRECT_MIN_STROKE_LEN = 0.15 * PUSHER_RADIUS
# =========================================================
# 1. Base settings (기본 설정)
# =========================================================
# 한 미션당 허용되는 최대 시간(초)
TIME_LIMIT_SEC = 120.0 
# 물리 엔진(total_.py)에서 적분할 때 쓰는 미소 시간 간격 (60FPS 기준)
DT = 1.0 / 60.0 

# AI 플래너가 인식하는 가상 작업 공간의 좌표 범위 (0% ~ 100%)
WORKSPACE_MIN = (0.00, 0.00)
WORKSPACE_MAX = (1.00, 1.00)

# 한 번 밀 때(Stroke) 허용되는 최대 길이. (0.023m = 2.3cm)
# 이를 전체 공간 크기(15cm)로 나누어 0.0~1.0 사이의 비율(정규화)로 변환해 저장합니다.

# 목표 달성(Success)으로 인정해주는 기준치들


SHAPE_ALIGN_SUCCESS_THRES = 0.005 / REAL_WORKSPACE_SIZE_X # 모양의 오차가 5mm 이하로 들어오면 성공

# =========================================================
# 2. Physics model (물리 모델 상수)
# =========================================================
# total_.py 물리 엔진에서 씁니다. 로봇이 물체를 밀 때의 가상 스프링 강성(밀어내는 힘의 세기)
K_SPRING = 25
# 바닥과 물체 사이의 동마찰계수 (물체가 바닥에서 얼마나 잘 미끄러지는가)
MU_G = 0.6
# 물체 간, 혹은 로봇과 물체 사이의 마찰계수 (현재 코드에선 안 쓰이나 예비용)

# =========================================================
# 3. Shape DB (물체 정보 데이터베이스)
# =========================================================
SHAPE_DB = {
    "square": { # 한쪽으로 무게가 쏠린 정사각형 블록
        "eff_r": 1.5*0.01148, # 유효 반경(회전 마찰력을 계산할 때 쓰는 바닥 면적의 등가 반지름)
        "cases": { # 가능한 무게중심 가설(Case)들
            # CASE_0: 무게가 정확히 정중앙에 있는 이상적인 상태 (질량 m, 관성모멘트 I, 무게중심 com 좌표)
            "CASE_0": {"m": 0.0092, "I": 0.00000133, "com": np.array([0.0, 0.0])},
            # CASE_1_A: 무게추(너트)가 왼쪽(-x 방향)으로 약간 쏠린 상태
            "CASE_1_A": {"m": 0.0096, "I": 0.00000137, "com": np.array([-0.000375, 0.0])},
            # CASE_1_B: 무게추가 왼쪽 위(대각선)로 쏠린 상태
            "CASE_1_B": {"m": 0.0096, "I": 0.00000137, "com": np.array([-0.000291, 0.000291])},
            # CASE_1_C: 무게추가 왼쪽 아래로 쏠린 상태
            "CASE_1_C": {"m": 0.0096, "I": 0.00000137, "com": np.array([-0.000291, -0.000291])},
            # CASE_2, 3 시리즈: 무게추가 2개, 3개씩 달려서 무게중심이 더 극단적으로 변한 상태들
            "CASE_2_AB": {"m": 0.0100, "I": 0.00000141, "com": np.array([-0.000640, 0.000280])},
            "CASE_2_AC": {"m": 0.0100, "I": 0.00000141, "com": np.array([-0.000640, -0.000280])},
            "CASE_2_BC": {"m": 0.0100, "I": 0.00000141, "com": np.array([-0.000560, 0.0])},
            "CASE_3_ALL": {"m": 0.0104, "I": 0.00000145, "com": np.array([-0.000885, 0.0])},
        },
    },
    "circle": { # 한쪽으로 무게가 쏠린 원형 블록 (구조는 위와 동일)
        "eff_r": 1.5*0.01000,
        "cases": {
            "CASE_0": {"m": 0.00726, "I": 0.00000077, "com": np.array([0.0, 0.0])},
            "CASE_1_A": {"m": 0.00766, "I": 0.00000081, "com": np.array([0.0, 0.000522])},
            "CASE_1_B": {"m": 0.00766, "I": 0.00000081, "com": np.array([0.0, -0.000522])},
            "CASE_1_C": {"m": 0.00766, "I": 0.00000081, "com": np.array([-0.000365, 0.000365])},
            "CASE_2_AB": {"m": 0.00806, "I": 0.00000085, "com": np.array([0.0, 0.0])},
            "CASE_2_AC": {"m": 0.00806, "I": 0.00000085, "com": np.array([-0.000347, 0.000843])},
            "CASE_2_BC": {"m": 0.00806, "I": 0.00000085, "com": np.array([-0.000347, -0.000148])},
            "CASE_3_ALL": {"m": 0.00846, "I": 0.00000089, "com": np.array([-0.000331, 0.000331])},
        },
    },
    "t": { # T자형 블록 (얘는 대회 규정상 무게추가 안 붙어서 CASE_0 하나만 존재)
        "eff_r": 8*0.01100,
        "cases": {
            "CASE_0": {"m": 0.00833, "I": 0.00000233, "com": np.array([0.00265, 0.0])}
        },
    },
}
# 가장 가설(Case)이 많은 물체의 가설 개수를 저장해둠
MAX_CASES = max(len(v["cases"]) for v in SHAPE_DB.values())

# =========================================================
# 4. System ID (로봇 스스로 무게중심 맞추기 설정)
# =========================================================
# brain.py의 update_com_belief 함수에서 사용됩니다.
# 실제 물체 회전량과 가설의 예측 회전량 오차가 5도 이내면 "이 가설이 맞구나!" 하고 득표 인정
SYSID_ANGLE_DEADZONE = np.deg2rad(5.0) 
# 한 가설이 3번(연속) 득표하면 로봇의 뇌가 공식적으로 해당 가설을 믿음(Belief)
SYSID_VOTE_THRESHOLD = 2
# 시스템 식별 로직이 동작할 때 터미널에 프린트할지 말지
SYSID_VERBOSE = False

# =========================================================
# 5. Workspace / Path Planning / Candidate Sampling
# =========================================================
# 목표 도달 여부(IoU)를 계산하기 위해 렌더링하는 흑백 마스크 이미지의 해상도 (256x256 픽셀)
MASK_W = 256
MASK_H = 256
# 원을 다각형으로 그릴 때 몇 각형으로 쪼갤지 (64각형으로 부드럽게 그림)
CIRCLE_POLY_POINTS = 16
# 원형 물체를 찌를 때, 360도를 몇 개의 각도로 쪼개서 테스트할지 (24방향에서 찔러봄)
# 다각형(사각형) 모서리(Edge)의 어느 지점들을 찔러볼지 비율 (10%, 30%, 50%, 70%, 90% 지점)
# T자형 물체의 찌르기 지점 비율
T_EDGE_RATIOS = [ 0.2, 0.5, 0.8]
T_FACE_FILTER_TOPK = 8
# 스핀(회전)을 먹이기 위해 수직이 아닌 접선(옆) 방향으로 찌를 때, 거리를 얼마나 곱해줄지 비율
# 정밀 조작 모드일 때 스트로크(밀기 길이)를 원래의 45%로 줄임

# 로봇 푸셔(찌르는 막대기)의 물리적 두께(반경 5.5mm)를 0~1 비율로 정규화
# A* 길찾기 알고리즘이 바닥을 바둑판으로 쪼갤 때 1칸의 크기 (3mm)
PATH_GRID_RES = 0.002 / REAL_WORKSPACE_SIZE_X

# =========================================================
# Path execution / A* safety limit
# =========================================================
# A*가 긴 우회 경로를 만들면 waypoint마다 move_xy + sleep이 들어가서 매우 느려질 수 있다.
# 직선 path는 보통 2개, L자 path는 보통 3개라서 10개 제한은 직선/L자에는 거의 영향 없음.
# A*를 쓰는 motion_planner.py에서 len(approach_path)가 이 값을 넘으면 해당 후보를 버린다.
MAX_APPROACH_WAYPOINTS = 5
APPROACH_WAYPOINT_LIMIT_SEQUENCE = [5, 7, 9, 12]
# =========================================================
# U-shape path planning
# =========================================================
# 직선/L자 경로가 막혔을 때, A* 전에 ㄷ자 우회 경로를 시도한다.
# ㄷ자는 물체 bounding box 바깥을 위/아래/왼쪽/오른쪽으로 돌아가는 4-waypoint 경로다.
#
# clearance 계산식:
# clearance = obstacle_margin * U_SHAPE_CLEARANCE_SCALE + U_SHAPE_CLEARANCE_EXTRA
#
# 값이 클수록 물체에서 더 멀리 돌아서 안전하지만, workspace 밖으로 나가 실패할 가능성이 커진다.
# 값이 작을수록 더 가까이 돌아서 경로가 잘 잡히지만, 충돌 판정에 걸리거나 실제 로봇이 물체에 닿을 위험이 커진다.
U_SHAPE_CLEARANCE_SCALE = 0.85
U_SHAPE_CLEARANCE_EXTRA = PATH_GRID_RES

# Executor sleep settings
# waypoint마다 기다리는 시간. 기존 0.2초에서 0.05초로 낮춰 긴 A* path 실행 시간을 줄인다.
EXECUTOR_PATH_SLEEP_TIME = 0.05

# start 도착 후 / retreat 후 안정화 대기
EXECUTOR_SLEEP_TIME = 0.15

# push 후 물체 안정화 대기
EXECUTOR_PUSH_SLEEP_TIME = 0.1
# A* 가 물체를 우회할 때 띄우는 여유 간격 (5mm)
PATH_OBSTACLE_MARGIN = 0.005 / REAL_WORKSPACE_SIZE_X

# =========================================================
# Initial / approach safety handling
# =========================================================
# push_start는 이미 물체 표면에서 PUSHER_RADIUS만큼 떨어진 점이다.
# approach_xy는 push_start에서 APPROACH_EXTRA_MARGIN만큼 더 바깥으로 뺀다.
# 기존처럼 PATH_OBSTACLE_MARGIN + 0.001만 쓰면 obstacle_margin 경계에 너무 붙어서
# goal_cell_blocked가 뜰 수 있으므로 grid 한 칸(PATH_GRID_RES) 여유를 둔다.
APPROACH_EXTRA_MARGIN = PATH_OBSTACLE_MARGIN + PATH_GRID_RES

# 로봇을 연결했을 때 푸셔가 물체 inflated margin 안쪽/경계에서 시작하는 경우,
# 일반 path planner를 바로 돌리면 start_cell_blocked 또는 segment collision으로 plan이 터질 수 있다.
# 실제 물체 내부가 아니라 margin band 안에 있는 경우는 먼저 바깥쪽 safe_start로 빠져나가도록 허용한다.
START_MARGIN_ESCAPE_ENABLE = True
START_MARGIN_ESCAPE_SAFE_EXTRA = PATH_GRID_RES
START_MARGIN_ESCAPE_MAX_DIST = (PUSHER_RADIUS + PATH_OBSTACLE_MARGIN) + 4.0 * PATH_GRID_RES
START_MARGIN_ESCAPE_DIRS = 32
# 경로 오차를 계산할 때, 위치 오차와 각도 오차를 합칠 때 각도에 곱해주는 가중치
# 목표에 도달했다고 판정할 위치 오차 관대함 (1mm)
# 오차가 1.5cm 이내로 들어오면 세밀한(Fine) 조작을 시작함

# 매 미션마다 로봇이 최초에 대기하고 있는 시작 위치 좌표 (10%, 10% 지점)
ROBOT_START_XY = (0.1, 0.1)

# brain.py의 build_reference_path에서 가상의 목표 궤적을 만들 때 점을 20개로 쪼갬
REF_PATH_STEPS = 20
# 기본 전방 주시 거리(Lookahead): 궤적 상의 4번째 앞 점을 당장의 타겟으로 삼음
REF_PATH_LOOKAHEAD = 4
REF_PATH_LOOKAHEAD_MIN = 2 # 최소 전방 주시 (2칸 앞)
REF_PATH_LOOKAHEAD_MAX = 6 # 최대 전방 주시 (6칸 앞)

# 목표까지 남은 거리에 따른 전방 주시 거리 세팅
REF_PATH_LOOKAHEAD_FAR = 6   # 목표가 멀면 6칸 앞을 봄
REF_PATH_LOOKAHEAD_MID = 4   # 중간이면 4칸 앞을 봄
REF_PATH_LOOKAHEAD_CLOSE = 3 # 코앞이면 3칸 앞을 봄

# 플랜에 실패해서 타겟을 뒤로 물릴 때(Retry), 한 번에 2칸씩(DELTA) 멀리 내다보며 최대 8칸(MAX)까지 양보함
REF_PATH_LOOKAHEAD_FALLBACK_DELTA = 2
REF_PATH_LOOKAHEAD_FALLBACK_MAX = 8

# 역학 필터링(filter_faces_by_dynamics) 후 살아남을 찌르기 면(Face)의 개수 (상위 4개 면만 집중 공략)
FACE_FILTER_TOPK = 6

# 밀기 힘(allowed_strokes 배율)을 조절하기 위한 남은 거리 기준선

# =========================================================
# Direct heuristic controller (가성비 직행 플래너 모드 설정)
# =========================================================
# 이 방식을 켤지 말지 (True니까 켜짐)

# 휴리스틱 모드에서 원을 찌를 때 생성할 각도 수 (위의 24보다 좀 더 듬성듬성한 16개)
DIRECT_CIRCLE_CANDIDATE_ANGLES = 16
# 휴리스틱 모드 다각형 찌르기 비율
DIRECT_EDGE_RATIOS = [0.1, 0.25, 0.5, 0.75, 0.9]
# 고려할 액션 종류: 정직하게 밀기, 좌스핀, 우스핀
# 휴리스틱 모드에서 스핀 먹일 때 접선 방향 스케일
DIRECT_SPIN_TANGENT_SCALE = 0.3
#DIRECT_SPIN_TANGENT_SCALE = 0.25
# 목표에 거의 다 왔을 때 힘을 원래의 30%로 줄임
# 아무리 짧게 밀어도 로봇 푸셔 반지름의 75%보다는 길게 밀어야 함 (찔끔거림 방지)

# 남은 거리가 0.5mm 이하가 되면 밀기를 멈춤
#DIRECT_GOAL_STOP_THRESH = 0.005 / REAL_WORKSPACE_SIZE_X
# 남은 거리가 1cm 이하가 되면 '목표 코앞(Near Goal)'으로 간주해 살살 밀기 시작
DIRECT_NEAR_GOAL_THRESH = 0.015 / REAL_WORKSPACE_SIZE_X

# 전방 주시(Lookahead) 모드를 결정할 때 참고하는 에러율(오차율) 한계점
REF_PATH_LOOKAHEAD_FAR_ERR_THRESH = 0.12  # 오차 12% 이상이면 FAR
REF_PATH_LOOKAHEAD_MID_ERR_THRESH = 0.06  # 오차 6% 이상이면 MID
REF_PATH_LOOKAHEAD_CLOSE_ERR_THRESH = 0.03 # 오차 3% 이상이면 CLOSE
DIRECT_FAR_REPLAN_ERR = 0.04

# 벡터를 만들 때 이동(Translation)과 회전(Rotation)의 중요도(Gain) 밸런스. 
# 회전에 약간의 가중치를 줌
DIRECT_TRANSLATION_GAIN = 1.0
DIRECT_ROTATION_GAIN = 0.15 / (np.pi / 2.0)

# 최종 점수 계산 시 A* 우회 경로가 너무 길면 주는 감점(Penalty) 가중치
DIRECT_PATH_PENALTY = 0.03
# 최소 이만큼(0.0002)은 오차가 줄어야 '진척(Progress)이 있었다'고 쳐줌
DIRECT_MIN_PROGRESS = 0.0002
# 최종 점수 통과 커트라인
DIRECT_NEGATIVE_SCORE_CUTOFF = 0.05
# 1차 예선에서 살려둘 후보군의 수 (상위 10개)
DIRECT_ACTION_SHORTLIST_TOPK = 16

# ----------------- 성능 및 디버그 로깅 스위치 모음 -----------------
PERF_LOG = True # 타이머 속도 측정 로그 켤까?
PERF_LOG_SIM_EVERY = 25 # 시뮬레이터 함수는 25번 호출될 때마다 1번씩 로그 찍음 (도배 방지)
PERF_LOG_PATH_EVERY = 20 # 길찾기 함수 로그 빈도
PERF_LOG_COSTMAP_EVERY = 20 
PERF_LOG_ASTAR_EVERY = 10 
AUTOPILOT_DEBUG_LOG = False # Pygame 화면 로깅 여부
DEBUG_BRAIN = False # brain.py 기본 로그
DEBUG_BRAIN_PROGRESS = False # 잘린 후보군들 이유(progress) 보여줄까? (True)
DEBUG_BRAIN_TOP_CANDIDATES = False # 최종 후보 Top 3 상세 정보 보여줄까? (True)
DEBUG_RELAX_CUTOFF = False 
VERBOSE_DEBUG = False # 찐막 전체 로깅

# =========================================================
# 헬퍼 함수 1: 물체 모양별 로컬(Local) 꼭짓점 좌표 반환
# =========================================================
def get_shape_corners(shape_type):
    # s = 중심에서 꼭짓점까지의 거리(15mm를 정규화한 값)
    s = 0.015 / REAL_WORKSPACE_SIZE_X
    
    if shape_type == "square":
        k = SQUARE_SCALE
        return [
            ( k * s,  k * s),
            (-k * s,  k * s),
            (-k * s, -k * s),
            ( k * s, -k * s),
        ]
    if shape_type == "circle":
        # 원형은 중심점 하나만 필요
        return [(0.0, 0.0)]
    if shape_type == "t":
        sx, sy = REAL_WORKSPACE_SIZE_X, REAL_WORKSPACE_SIZE_Y
        k = T_BASE_SCALE
        return [
            (k * 0.005 / sx,  k * -0.015 / sy),
            (k * 0.015 / sx,  k * -0.015 / sy),
            (k * 0.015 / sx,  k *  0.015 / sy),
            (k * 0.005 / sx,  k *  0.015 / sy),
            (k * 0.005 / sx,  k *  0.005 / sy),
            (k * -0.015 / sx, k *  0.005 / sy),
            (k * -0.015 / sx, k * -0.005 / sy),
            (k * 0.005 / sx,  k * -0.005 / sy),
        ]
    raise ValueError(f"unknown shape: {shape_type}")

# =========================================================
# 헬퍼 함수 2: 원형 물체의 반지름 반환
# =========================================================
def get_shape_radius(shape_type):
    if shape_type == "circle":
        # 원의 실제 반지름 1.5cm를 정규화해서 반환
        return CIRCLE_SCALE * (0.015 / REAL_WORKSPACE_SIZE_X)
    return None

# =========================================================
# 8. Scenario reset (테스트용 랜덤 시나리오 생성기)
# =========================================================
def reset_scenario(shape_type, rng=np.random.default_rng()):
    # autopilot.py 시뮬레이터에서 'R' 키를 누르면 물체 위치를 리셋할 때 씁니다.
    
    # 1. 물체의 현재 위치(시작점)를 난수로 지정: x,y는 0.25~0.375 사이, 각도는 랜덤
    current_pose = [
        float(rng.uniform(0.25, 0.375)),
        float(rng.uniform(0.25, 0.375)),
        float(rng.uniform(-0.4, 0.4)),
    ]
    # 2. 물체가 도착해야 할 목표 위치를 난수로 지정: x,y는 0.60~0.75 사이, 각도는 랜덤
    target_pose = [
        float(rng.uniform(0.60, 0.75)),
        float(rng.uniform(0.60, 0.75)),
        float(rng.uniform(-0.8, 0.8)),
    ]

    # 해당 물체의 꼭짓점 정보와 반지름 정보를 가져옴
    local_corners = get_shape_corners(shape_type)
    radius = get_shape_radius(shape_type)

    # 생성된 랜덤 시작/목표 좌표와 물체 형상 데이터를 묶어서 반환!
    return current_pose, target_pose, local_corners, radius
