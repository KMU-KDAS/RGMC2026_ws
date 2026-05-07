import numpy as np # 벡터, 행렬 등 수학적 배열 계산을 빠르게 하기 위한 라이브러리
from numba import njit # 파이썬 코드를 C언어 수준으로 번역해 실행 속도를 폭발적으로 높여주는 JIT 컴파일러

from task1 import config # 로봇의 초기 설정값, 물리 상수(마찰계수, 크기 등)가 정의된 외부 파일
from perf_utils import perf_timer # 특정 함수가 실행되는 데 걸린 시간을 측정하기 위한 성능 측정 유틸리티

# =========================================================
# [1] 주최측 공식 규격(30mm) 기반 물리량 계산 함수
# =========================================================
def calc_T_shape_official(m_total=0.015):
    # T자형 물체의 무게중심(COM)과 관성모멘트를 구하는 함수입니다. (총 질량 기본값 0.015kg = 15g)
    w1, h1 = 0.03, 0.01  # T자의 윗부분(머리) 파트 규격. 너비 30mm(0.03m), 높이 10mm(0.01m)
    w2, h2 = 0.01, 0.02  # T자의 아랫부분(몸통) 파트 규격. 너비 10mm(0.01m), 높이 20mm(0.02m)
    m1, m2 = m_total * 0.5, m_total * 0.5 # 위아래 파트의 질량을 정확히 절반(7.5g씩)으로 가정함
    y1 = 0.01    # 기준점(보통 다각형의 형태학적 중심)에서 윗부분 파트의 중심까지의 y축 거리 (10mm 위)
    y2 = -0.005  # 기준점에서 아랫부분 파트의 중심까지의 y축 거리 (5mm 아래)

    # 새로운 무게중심의 y좌표 = (질량1*거리1 + 질량2*거리2) / 총질량 (가중 평균)
    com_y = (m1 * y1 + m2 * y2) / (m1 + m2)

    # 평행축 정리를 이용한 윗부분의 관성모멘트: (기본 직사각형 관성모멘트) + 질량 * (이동한 거리)^2
    i1 = (1/12) * m1 * (w1**2 + h1**2) + m1 * (y1 - com_y)**2
    # 평행축 정리를 이용한 아랫부분의 관성모멘트
    i2 = (1/12) * m2 * (w2**2 + h2**2) + m2 * (y2 - com_y)**2

    # 최종 무게중심 y좌표와 두 파트를 합친 총 관성모멘트 반환
    return com_y, (i1 + i2)

def calc_real_nut_offset(base_m=0.01, nut_m=0.002, pos_x=0.01, pos_y=0.01):
    # 정사각형 베이스 위에 특정 위치(pos_x, pos_y)에 너트(무게추)를 올려놨을 때의 동역학적 변화 계산
    total_m = base_m + nut_m # 총 질량 = 베이스 질량(10g) + 너트 질량(2g)

    # 너트 때문에 x축 방향으로 쏠린 새로운 무게중심 = (너트 질량 * 너트 x위치) / 총 질량
    new_com_x = (nut_m * pos_x) / total_m
    # 너트 때문에 y축 방향으로 쏠린 새로운 무게중심
    new_com_y = (nut_m * pos_y) / total_m

    # 베이스 자체의 순수 관성모멘트 (정사각형 공식: 1/12 * 질량 * (가로^2 + 세로^2))
    i0 = (1/12) * base_m * (0.03**2 + 0.03**2)

    # 전체 관성모멘트 = 베이스의 새로운 관성모멘트(평행축 정리) + 너트의 관성모멘트(점질량 처리: mr^2)
    i_total = i0 + base_m * (new_com_x**2 + new_com_y**2) + nut_m * ((pos_x - new_com_x)**2 + (pos_y - new_com_y)**2)

    # 새로운 무게중심 좌표 배열, 총 질량, 총 관성모멘트 반환
    return np.array([new_com_x, new_com_y]), total_m, i_total

@njit # Numba 데코레이터: 파이썬 인터프리터를 무시하고 기계어로 직행 컴파일해 실행 속도를 폭발적으로 높임
def get_circle_normal(cx, cy, px, py, radius):
    # 원형 물체 표면의 한 점(px, py)에서 물체 중심(cx, cy)을 향하는 방향 벡터(Normal)를 구함
    dx = px - cx # 중심에서 표면 점까지의 x축 거리 차이
    dy = py - cy # 중심에서 표면 점까지의 y축 거리 차이
    dist = np.sqrt(dx**2 + dy**2) # 피타고라스 정리로 두 점 사이의 직선 거리 계산

    if dist < 1e-9: # 두 점이 거의 겹쳐있을 경우 (0으로 나누는 치명적 에러 방지용)
        return np.array([1.0, 0.0]) # 기본값(x축 방향 벡터) 반환

    # 단위 벡터(길이가 1인 방향 벡터)로 만들어서 반환
    return np.array([dx / dist, dy / dist])

# =========================================================
# [2] Numba JIT 적용을 위한 내부 고속 물리 엔진
# =========================================================
@njit # 역시 기계어로 번역하여 속도 극대화 (물리 엔진 코어이므로 필수, 반복 연산이 매우 잦음)
def _simulate_step_numba(m, i_body, com_world_x, com_world_y, contact_x, contact_y, theta, v, w, f_app_x, f_app_y, dt, mu_g, eff_r):
    # 단일 시간 단위(dt) 동안 물체의 위치/회전 변화를 1스텝 적분하는 함수
    g = 9.81 # 중력 가속도 (m/s^2)
    v_norm = np.sqrt(v[0]**2 + v[1]**2) # 현재 물체의 병진 속도(직선으로 이동하는 속력) 크기 계산

    # --- 1. 병진 운동 마찰력 계산 ---
    if v_norm > 1e-6: # 물체가 아주 조금이라도 움직이고 있다면 동마찰력 적용
        max_fric = mu_g * m * g # 동마찰력 공식: 마찰계수(mu) * 수직항력(mg)
        stop_fric = (v_norm * m) / dt  # F=ma에서 유도. 현재 속도를 한 스텝(dt)만에 0으로 멈추게 하는 데 필요한 힘의 크기
        actual_fric = min(max_fric, stop_fric) # 마찰력이 너무 세서 물체가 뒤로 튕기는 수치적 오류를 막기 위해 최솟값 선택
        f_fric_trans = -(v / v_norm) * actual_fric # 움직이는 방향(v / v_norm)의 정확히 반대(-) 방향으로 마찰력 벡터 생성
    else:
        f_fric_trans = np.zeros(2) # 안 움직이면 마찰력도 0 벡터

    # --- 2. 회전 운동 마찰 토크 계산 ---
    if abs(w) > 1e-6: # 물체가 회전(각속도 w)하고 있다면 회전 마찰 적용
        max_tau = mu_g * m * g * eff_r # 최대 회전 마찰 토크 = 마찰계수 * 수직항력 * 유효 반경(바닥에 닿는 면적 비례)
        stop_tau = (abs(w) * i_body) / dt # 현재 각속도를 한 스텝만에 0으로 만드는 데 필요한 토크
        actual_tau = min(max_tau, stop_tau) # 수치 오류 방지를 위한 최솟값 선택
        tau_fric_rot = -1.0 * actual_tau if w > 0 else actual_tau # 회전하는 방향과 반대로 마찰 토크 부호 결정
    else:
        tau_fric_rot = 0.0 # 회전 안 하면 회전 마찰 토크 0

    # --- 3. 로봇이 미는 힘(Pushing Force)에 의한 토크 계산 ---
    r_x = contact_x - com_world_x # 모멘트 암의 x성분 = 접촉점 x - 무게중심 x
    r_y = contact_y - com_world_y # 모멘트 암의 y성분 = 접촉점 y - 무게중심 y
    # 토크(외적) = r x F = r_x * F_y - r_y * F_x (2D 평면에서의 외적 공식)
    tau_app = r_x * f_app_y - r_y * f_app_x

    # --- 4. 알짜힘 및 가속도 계산 ---
    f_net_x = f_app_x + f_fric_trans[0] # x축 알짜힘 = 미는 힘 x + 마찰력 x (마찰력은 이미 음수 부호 가짐)
    f_net_y = f_app_y + f_fric_trans[1] # y축 알짜힘 = 미는 힘 y + 마찰력 y

    a_x = f_net_x / m # 뉴턴 제2법칙 (a = F/m): x축 가속도
    a_y = f_net_y / m # y축 가속도
    alpha = (tau_app + tau_fric_rot) / i_body # 회전 뉴턴 제2법칙 (alpha = Tau/I): 각가속도 = (미는 토크 + 마찰 토크) / 관성모멘트

    # --- 5. 상태 업데이트 (오일러 적분법) ---
    v_next = np.array([v[0] + a_x * dt, v[1] + a_y * dt]) # 다음 속도 = 현재 속도 + (가속도 * 시간)
    w_next = w + alpha * dt # 다음 각속도 = 현재 각속도 + (각가속도 * 시간)
    theta_next = theta + w_next * dt # 다음 각도 = 현재 각도 + (다음 각속도 * 시간)
    delta_pos = v_next * dt # 이번 스텝에서 물체가 이동한 x, y 거리 변화량 벡터

    return v_next, w_next, theta_next, delta_pos # 다음 상태값 반환


@njit
def _simulate_push_stroke_numba(m, i_body, com_x, com_y, body_center, theta, v, w,
                                start_x, start_y, end_x, end_y,
                                dt, k_spring, mu_g, eff_r,
                                push_speed_mps, min_push_steps):
    # 로봇이 시작점(start)에서 끝점(end)까지 한 번에 쭈욱 미는 동작 전체를 계산하는 함수
    c = np.cos(theta) # 현재 각도의 코사인 값 (회전 변환용)
    s = np.sin(theta) # 현재 각도의 사인 값

    # 물체의 로컬 좌표계(중심이 0,0)에 있는 무게중심(com)을 월드 좌표계(실제 지도 좌표)로 변환 (2D 회전 행렬 적용)
    com_world_x = body_center[0] + (c * com_x - s * com_y)
    com_world_y = body_center[1] + (s * com_x + c * com_y)

    dx = end_x - start_x
    dy = end_y - start_y
    total_dist = np.sqrt(dx**2 + dy**2)

    if total_dist < 1e-8:
        return v, w, theta, body_center

    stroke_dir_x = dx / total_dist
    stroke_dir_y = dy / total_dist

    # 실제 로봇 속도 기반 총 접촉 시간
    total_time = total_dist / max(push_speed_mps, 1e-6)

    # dt 기준 적분 step 수
    steps = int(np.ceil(total_time / dt))
    if steps < min_push_steps:
        steps = min_push_steps

    # step당 푸셔 이동 거리
    step_dist = total_dist / steps

    curr_v, curr_w, curr_theta = v, w, theta
    curr_body_center = np.copy(body_center)

    for k in range(steps):
        # 푸셔가 실제로 start -> end로 움직이게 contact 전진
        alpha = (k + 1) / steps
        contact_x = start_x + alpha * dx
        contact_y = start_y + alpha * dy

        # step당 이동거리에 비례한 힘
        f_app_mag = k_spring * step_dist
        f_app_x = stroke_dir_x * f_app_mag
        f_app_y = stroke_dir_y * f_app_mag

        curr_v, curr_w, curr_theta, d_pos = _simulate_step_numba(
            m, i_body, com_world_x, com_world_y, contact_x, contact_y,
            curr_theta, curr_v, curr_w, f_app_x, f_app_y, dt, mu_g, eff_r
        )

        curr_body_center[0] += d_pos[0]
        curr_body_center[1] += d_pos[1]
        com_world_x += d_pos[0]
        com_world_y += d_pos[1]

    # 모든 스텝을 완료한 후의 최종 속도, 각속도, 각도, 형태 중심 반환
    return curr_v, curr_w, curr_theta, curr_body_center

# =========================================================
# [3] 기존 파일과의 호환성을 위한 Wrapper (다른 파일 수정 방지)
# =========================================================
def simulate_push_stroke(m, i_body, com_x, com_y, body_center, theta, v, w, start_x, start_y, end_x, end_y, n_hat, t_hat, config_dict, eff_r):
    # 딕셔너리(config_dict)는 C언어로 변환되는 Numba 엔진에 통째로 전달할 수 없으므로, 
    # 여기서 필요한 값을 하나씩 빼서(언패킹) Numba 함수로 넘겨주는 다리(Wrapper) 역할
    return _simulate_push_stroke_numba(
        m, i_body, com_x, com_y, body_center, theta, v, w,
        start_x, start_y, end_x, end_y,
        config_dict["DT"],
        config_dict["K_SPRING"],
        config_dict["MU_G"],
        eff_r,
        config_dict["PUSH_SPEED_MPS"],
        config_dict.get("MIN_PUSH_STEPS", 3),
    )

def simulate_push_stroke_normalized(m, i_body, com_x, com_y, body_center_norm, theta, v, w, start_x_norm, start_y_norm, end_x_norm, end_y_norm, n_hat, t_hat, config_dict, eff_r):
    # AI 플래너가 계산하기 편하게 0.0~1.0 사이로 쓰는 '정규화 좌표'를 
    # 실제 물리 환경의 '미터(m) 단위'로 변환해주는 래퍼 함수
    with perf_timer("total_.simulate_push_stroke_normalized", every=config.PERF_LOG_SIM_EVERY): # 이 블록이 몇 초 걸리는지 성능 측정
        
        # 1. 실제 작업 공간의 가로/세로 길이(예: 0.12m = 12cm) 가져오기
        W_SIZE_X = config_dict["REAL_WORKSPACE_SIZE_X"]
        W_SIZE_Y = config_dict["REAL_WORKSPACE_SIZE_Y"]
        
        # 2. 작업 공간의 중앙을 (0,0)으로 잡았을 때의 최소 x, y 좌표값 (예: -0.06m)
        W_MIN_X = -(W_SIZE_X / 2.0)
        W_MIN_Y = -(W_SIZE_Y / 2.0)

        # 3. 플래너가 준 정규화 좌표(0.0~1.0)를 실제 미터(m) 단위의 글로벌 좌표로 변환
        # 공식: 최소값 + (정규화비율 * 전체길이)
        bc_m = np.array([W_MIN_X + body_center_norm[0] * W_SIZE_X, W_MIN_Y + body_center_norm[1] * W_SIZE_Y])
        st_m_x = W_MIN_X + start_x_norm * W_SIZE_X
        st_m_y = W_MIN_Y + start_y_norm * W_SIZE_Y
        ed_m_x = W_MIN_X + end_x_norm * W_SIZE_X
        ed_m_y = W_MIN_Y + end_y_norm * W_SIZE_Y

        # 4. 변환된 미터 단위 좌표를 물리 엔진에 넣어서 결과 받아오기
        next_v, next_w, theta_next, next_bc_m = _simulate_push_stroke_numba(
            m, i_body, com_x, com_y, bc_m, theta, v, w,
            st_m_x, st_m_y, ed_m_x, ed_m_y,
            config_dict["DT"],
            config_dict.get("K_SPRING", 500.0),
            config_dict.get("MU_G", 0.3),
            eff_r,
            config_dict["PUSH_SPEED_MPS"],
            config_dict.get("MIN_PUSH_STEPS", 3),
        )

        # 5. 반환받은 미터 단위의 물체 결과 위치를 다시 플래너가 쓰는 정규화 좌표(0.0~1.0)로 원복
        next_bc_norm = np.array([
            (next_bc_m[0] - W_MIN_X) / W_SIZE_X,
            (next_bc_m[1] - W_MIN_Y) / W_SIZE_Y
        ])
        
        # 6. 최종 결과 반환
        return next_v, next_w, theta_next, next_bc_norm

def warmup_numba_simulation(repeats=2, shape_type="WEIGHTED_SQUARE"):
    # Numba 엔진의 "첫 실행 지연(First-Execution Penalty)"을 없애기 위한 웜업(준비 운동) 함수
    # Numba는 최초 실행 시 파이썬 코드를 C 코드로 컴파일하느라 몇 초 정도 버벅거립니다.
    # 로봇이 실제 미션 중에 버벅거리는 걸 막기 위해, 프로그램 시작 직후 더미 데이터를 넣어 미리 컴파일을 강제로 시켜둡니다.

    # 지정된 형태(shape_type)의 물리 속성 데이터베이스를 불러옴
    shape_data = config.SHAPE_DB[shape_type]
    case_name = next(iter(shape_data["cases"])) # 첫 번째 케이스 이름 추출
    case_info = shape_data["cases"][case_name] # 해당 케이스의 질량, 관성 정보 추출

    # 엔진을 돌리기 위한 임의의 가짜(더미) 데이터 세팅
    body_center_norm = np.array([0.35, 0.35], dtype=float)
    start_xy = np.array([0.35, 0.475], dtype=float)
    end_xy = np.array([0.35, 0.375], dtype=float)
    n_hat = np.array([0.0, 1.0], dtype=float)
    t_hat = np.array([1.0, 0.0], dtype=float)

    # 주어진 반복 횟수(기본 2번)만큼 의미 없는 연산을 강제 실행하여 컴파일을 완료시킴
    for _ in range(max(1, int(repeats))):
        simulate_push_stroke_normalized(
            m=case_info["m"],
            i_body=case_info["I"],
            com_x=case_info["com"][0],
            com_y=case_info["com"][1],
            body_center_norm=body_center_norm,
            theta=0.0,
            v=np.array([0.0, 0.0], dtype=float),
            w=0.0,
            start_x_norm=float(start_xy[0]),
            start_y_norm=float(start_xy[1]),
            end_x_norm=float(end_xy[0]),
            end_y_norm=float(end_xy[1]),
            n_hat=n_hat,
            t_hat=t_hat,
            config_dict=config.__dict__,
            eff_r=shape_data["eff_r"],
        )