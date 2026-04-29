import os # 운영체제와 상호작용하기 위한 라이브러리 (폴더 생성, 파일 경로 설정 등에 사용)
import concurrent.futures # 병렬 처리(멀티프로세싱)를 수행하여 여러 에피소드를 동시에 돌리기 위한 라이브러리
from collections import Counter # 리스트나 튜플 내의 데이터 개수를 쉽게 세어주는 딕셔너리 확장 클래스 (진동(Oscillation) 감지에 사용)

import numpy as np # 행렬, 벡터 계산 및 수학 연산을 위한 필수 라이브러리
import pandas as pd # 시뮬레이션 결과 데이터를 표(엑셀) 형태로 다루고 저장하기 위한 라이브러리
import matplotlib # 그래프를 그리기 위한 라이브러리
# "Agg"는 Anti-Grain Geometry의 약자로, 화면(GUI 창)에 그래프를 띄우지 않고 
# 백그라운드에서 곧바로 이미지 파일로 저장하기 위해 사용하는 렌더링 엔진(백엔드)입니다. (Headless 환경에 필수)
matplotlib.use("Agg")
import matplotlib.pyplot as plt # 그래프 그리기용 주요 모듈

import config # 물체 규격, 마찰계수, 작업 공간 크기 등이 정의된 환경설정 파일
from brain import PushingBrain # 시스템 식별(SysID)과 A* 우회/밀기 플래닝을 담당하는 인공지능 '뇌'
from planner import Planner # 현재 모양과 목표 모양의 IoU(겹침 비율)를 계산하는 심판 역할
from total_ import simulate_push_stroke_normalized # Numba 기반의 초고속 2D 동역학 물리 엔진
from geometry_utils import shape_alignment_error, wrap_angle # 모양 오차 계산 및 각도 정규화(-pi~pi) 수학 함수


# =====================================================================
# 설정 (Configuration)
# =====================================================================
# 한 에피소드(미션) 당 로봇이 물체를 밀 수 있는 최대 횟수입니다. 
# 80번을 밀어도 목표에 도달하지 못하면 '시간 초과(timeout)'로 간주합니다.
MAX_STEPS = 80 
# 각 물체/무게중심 가설(Case) 하나당 테스트할 독립적인 에피소드의 개수입니다.
EPISODES_PER_CASE = 10 
# 난수 생성기(Random Number Generator)의 시작 시드 값입니다. 
# 이를 고정하면 테스트할 때마다 항상 같은 상황이 재현되어 알고리즘 수정 전/후를 공정하게 비교할 수 있습니다.
BASE_SEED = 42 
# 평가 결과(CSV 데이터, 그래프 이미지 등)를 저장할 폴더의 이름입니다.
OUT_DIR = "eval_results" 


# =====================================================================
# 유틸리티 함수 (Utilities)
# =====================================================================
def quantized_state(pose, pos_q=0.01, ang_q=np.deg2rad(5.0)):
    """
    로봇이 물체를 이리저리 밀다가 '계속 같은 자리를 맴도는 현상(Oscillation)'을 감지하기 위해 
    현재 위치(연속적인 실수 값)를 듬성듬성한 격자(이산적인 정수 값)로 변환하는 함수입니다.
    
    - pos_q: 위치를 1cm(0.01) 단위로 뭉뚱그림
    - ang_q: 각도를 5도(np.deg2rad(5.0)) 단위로 뭉뚱그림
    """
    x, y, th = pose
    return (
        int(round(x / pos_q)), # x 좌표를 1cm 단위 칸으로 변환
        int(round(y / pos_q)), # y 좌표를 1cm 단위 칸으로 변환
        int(round(wrap_angle(th) / ang_q)), # 각도를 5도 단위 칸으로 변환
    )


def classify_failure(step_count, no_plan_step, progress_window, repeated_state_count, final_pose):
    """
    목표 도달에 실패했을 때, 그 실패의 '원인'을 정밀하게 분석하여 텍스트로 반환하는 진단 함수입니다.
    """
    x, y, _ = final_pose

    # 1. 물체가 작업 공간(0.0 ~ 1.0) 밖으로 튕겨 나간 경우
    if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
        return "out_of_workspace"

    # 2. 첫 스텝(시작하자마자)부터 찌를 각이나 우회 경로가 없어서 플랜을 아예 못 짠 경우
    if no_plan_step == 0:
        return "no_plan_from_start"

    # 3. 중간까지는 잘 밀다가 구석에 몰리거나 해서 중간에 플랜을 못 짜게 된 경우
    if no_plan_step is not None:
        return "no_plan_midway"

    # 4. 동일한 이산 상태(위치+각도)에 3번 이상 머물렀을 경우 (제자리걸음, 진동)
    if repeated_state_count >= 3:
        return "oscillation"

    # 5. 최근 8번의 밀기(step) 동안 오차가 줄어든 양(progress)의 합이 0.01 미만일 경우
    # 즉, 계속 헛손질을 하거나 거의 움직이지 않고 낑겨있는 상태
    if len(progress_window) >= 8 and sum(progress_window[-8:]) < 0.01:
        return "stuck_low_progress"

    # 6. 제한된 횟수(MAX_STEPS, 80번)를 모두 소모할 때까지 목표에 도달하지 못한 경우
    if step_count >= MAX_STEPS:
        return "timeout_max_steps"

    # 위 조건에 해당하지 않는 알 수 없는 에러
    return "unknown_failure"


# =====================================================================
# 단일 에피소드 실행 (Run 1 Episode)
# =====================================================================
def run_one_episode(shape_type, true_case, seed):
    """
    주어진 모양(shape_type)과 진짜 무게중심 가설(true_case)을 바탕으로
    1회의 완전한 로봇 밀기 시뮬레이션을 처음부터 끝까지 수행합니다.
    """
    # 에피소드별로 독립적인 난수 생성기(RNG)를 만듭니다.
    rng = np.random.default_rng(seed)
    # IoU(성공률) 검사 등을 위한 플래너 객체 생성
    planner = Planner(config)
    # 플랜을 짜고 무게중심을 추정할 인공지능 뇌 객체 생성
    brain = PushingBrain()

    # 물체의 시작 위치, 목표 위치 등을 랜덤으로 세팅합니다. (config.py의 설정 사용)
    current_pose, target_pose, local_corners, radius = config.reset_scenario(shape_type, rng)
    # 로봇의 초기 대기 위치를 설정합니다.
    robot_pose = [float(config.ROBOT_START_XY[0]), float(config.ROBOT_START_XY[1])]

    # 에피소드 동안 기록할 통계 변수들 초기화
    total_steps = 0 # 로봇이 물체를 민 총 횟수
    total_rotation_deg = 0.0 # 물체가 회전한 총 각도 (deg)
    no_plan_step = None # 플랜에 실패한 스텝 번호
    progress_window = [] # 최근 몇 번의 스텝 동안 얼마나 목표에 가까워졌는지(진척도)를 기록하는 윈도우
    visited = Counter() # 방문했던 이산 상태들을 기록하여 진동(Oscillation)을 감지하기 위한 카운터

    # 시작 시점에서의 초기 오차(현재 위치 vs 목표 위치의 모양 오차) 계산
    initial_err = shape_alignment_error(shape_type, current_pose, target_pose, local_corners, radius)
    prev_err = initial_err # 다음 스텝과의 진척도 비교를 위해 이전 오차로 저장

    success = False # 에피소드 성공 여부 플래그

    # 최대 허용 횟수(MAX_STEPS)만큼 밀기 루프를 돕니다.
    for step_idx in range(MAX_STEPS):
        # 1. 진동 감지를 위해 현재 상태를 격자 단위로 묶어서 카운트 증가
        state_key = quantized_state(current_pose)
        visited[state_key] += 1

        # 2. 현재 시점의 오차를 계산하고, 성공 커트라인(SHAPE_ALIGN_SUCCESS_THRES) 이내로 들어왔는지 검사
        current_err = shape_alignment_error(shape_type, current_pose, target_pose, local_corners, radius)
        if current_err <= config.SHAPE_ALIGN_SUCCESS_THRES:
            success = True # 성공했다면 플래그를 True로 바꾸고 루프 탈출
            break

        # 3. 뇌(Brain)에게 현재 위치에서 목표로 가기 위한 최적의 우회 및 찌르기 계획을 물어봅니다.
        plan = brain.get_best_plan(
            shape_type=shape_type,
            current_pose=current_pose,
            target_pose=target_pose,
            local_corners=local_corners,
            planner=planner,
            radius=radius,
            robot_pose=robot_pose,
        )

        # 4. 각이 안 나와서 길을 못 찾았을 경우, 실패한 스텝 번호를 기록하고 루프 탈출
        if plan is None:
            no_plan_step = step_idx
            break

        # 5. 플랜이 성공적으로 나왔다면 액션(찌르기)과 모션(이동 경로) 정보를 분리
        cand = plan["candidate"] # 밀기 동작(시작점, 끝점, 방향)
        motion = plan["motion"]  # A* 우회 경로 정보
        
        # 현재 물체의 '진짜 물리 속성(질량, 진짜 무게중심 좌표, 관성모멘트)'을 config에서 불러옴
        true_case_info = config.SHAPE_DB[shape_type]["cases"][true_case]

        pre_push_pose = list(current_pose) # 밀기 전 위치 백업 (SysID 학습을 위해)

        # 6. 초고속 물리 엔진을 돌려서 로봇이 진짜로 물체를 밀었을 때의 결과를 계산 (가상 시뮬레이션)
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

        # 7. 이번 스텝에서 물체가 회전한 각도(차이)를 누적함
        rot_diff_rad = abs(wrap_angle(theta_next - pre_push_pose[2]))
        total_rotation_deg += float(np.degrees(rot_diff_rad))
        total_steps += 1 # 스텝 카운트 증가

        # 8. 물체와 로봇의 위치를 물리 엔진이 계산한 '다음 도착 위치'로 갱신함
        current_pose = [
            float(next_body_center_norm[0]),
            float(next_body_center_norm[1]),
            float(theta_next),
        ]
        robot_pose = [
            float(motion["approach_xy"][0]),
            float(motion["approach_xy"][1]),
        ]

        # 9. ★ 핵심 로직: 뇌(Brain)가 방금 일어난 실제 움직임(theta_next - pre_push_pose[2])을 보고, 
        # 자신이 믿고 있던 무게중심(COM) 가설이 맞았는지 확인하고 업데이트(투표)함
        brain.update_com_belief(theta_next - pre_push_pose[2], shape_type, pre_push_pose, cand)

        # 10. 밀고 난 뒤의 새 오차를 계산하고, 방금 턴에서 오차가 얼마나 줄었는지(progress) 기록
        next_err = shape_alignment_error(shape_type, current_pose, target_pose, local_corners, radius)
        progress = max(0.0, prev_err - next_err) # 뒷걸음질(음수)은 0으로 처리
        progress_window.append(progress)
        prev_err = next_err

    # 루프가 완전히 끝난 뒤 최종 오차와, 가장 많이 머물렀던 상태의 카운트를 계산
    final_err = shape_alignment_error(shape_type, current_pose, target_pose, local_corners, radius)
    repeated_state_count = max(visited.values()) if visited else 0

    # 성공했다면 사유를 "success"로, 실패했다면 위에서 정의한 진단 함수로 원인을 분석함
    if success:
        failure_reason = "success"
    else:
        failure_reason = classify_failure(
            step_count=total_steps,
            no_plan_step=no_plan_step,
            progress_window=progress_window,
            repeated_state_count=repeated_state_count,
            final_pose=current_pose,
        )

    # 1회의 에피소드 동안 수집된 모든 정보(성적표)를 딕셔너리 형태로 반환
    return {
        "shape_type": shape_type, # 물체 모양 (예: 사각형)
        "true_case": true_case,   # 테스트한 진짜 무게중심 가설
        "seed": seed,             # 사용한 난수 시드
        "success": int(success),  # 성공 1, 실패 0
        "steps": total_steps,     # 밀기 횟수
        "total_rotation_deg": total_rotation_deg, # 총 회전 각도
        "initial_err": initial_err, # 처음 오차
        "final_err": final_err,     # 마지막 오차
        "failure_reason": failure_reason, # 성공 또는 구체적 실패 사유
    }


# =====================================================================
# 전체 실행 및 병렬 처리 (Run All via Multiprocessing)
# =====================================================================
def run_all():
    """
    설정된 모든 모양, 모든 케이스에 대해 지정된 횟수만큼 에피소드를 생성하고,
    CPU의 멀티코어를 활용해 이를 병렬로 빠르게 평가(Evaluation)하는 메인 함수입니다.
    """
    # 결과물을 저장할 폴더가 없으면 새로 만듭니다.
    os.makedirs(OUT_DIR, exist_ok=True)

    tasks = [] # 평가할 에피소드들의 설정값을 담을 리스트
    seed_counter = BASE_SEED # 시드 값을 순차적으로 증가시키기 위한 변수

    # 1. config.py에 정의된 모든 물체(SHAPE_DB)와 모든 무게중심 가설(cases)에 대해 반복
    for shape_type, shape_info in config.SHAPE_DB.items():
        for true_case in shape_info["cases"].keys():
            # 각 케이스당 10번(EPISODES_PER_CASE)씩 독립적인 미션을 생성하여 작업을 예약함
            for _ in range(EPISODES_PER_CASE):
                tasks.append((shape_type, true_case, seed_counter))
                seed_counter += 1

    rows = [] # 각 에피소드가 끝나고 반환한 성적표(딕셔너리)를 담을 리스트
    print(f"총 {len(tasks)}개의 에피소드 평가를 시작합니다...")
    
    # 2. 병렬 처리 진입점: ProcessPoolExecutor를 사용해 CPU 코어를 갈굼
    # Ryzen 7 8845HS는 16스레드이므로, max_workers=12로 설정해 12개의 백그라운드 일꾼을 생성합니다.
    # (나머지 4개는 윈도우 OS 구동, 웹서핑 등 여유 자원으로 남겨둠)
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        # submit 함수를 통해 예약된 작업(tasks)들을 12명의 일꾼에게 골고루 던져줍니다.
        futures = [executor.submit(run_one_episode, *task) for task in tasks]

        # as_completed는 작업이 '끝나는 순서대로' 결과를 뱉어내는 제너레이터입니다.
        for idx, future in enumerate(concurrent.futures.as_completed(futures), 1):
            # 완료된 에피소드의 결과(성적표 딕셔너리)를 받아와서 rows 리스트에 추가합니다.
            rows.append(future.result())
            
            # 터미널 창에 10개 단위로 진행률을 출력하여 사용자가 멈춘 게 아님을 알게 해줍니다.
            if idx % 10 == 0 or idx == len(tasks):
                print(f"진행 상황: {idx}/{len(tasks)} 완료")

    print("평가 완료! 데이터를 저장하고 그래프를 그립니다...")
    
    # 3. 데이터 처리: 리스트에 담긴 수많은 성적표를 Pandas의 DataFrame(엑셀 표 형태)으로 변환
    df = pd.DataFrame(rows)
    # csv 파일로 결과를 저장 (나중에 엑셀 등으로 열어볼 수 있음)
    df.to_csv(os.path.join(OUT_DIR, "episode_results.csv"), index=False)

    # 4. 수집된 데이터를 바탕으로 결과 그래프(PNG)들을 생성합니다.
    make_plots(df)
    print("모든 작업이 끝났습니다. eval_results 폴더를 확인하세요!")


# =====================================================================
# 결과 그래프 생성 (Visualization)
# =====================================================================
def make_plots(df):
    """
    Pandas DataFrame을 입력받아 성공률, 소모된 스텝, 회전량, 오차 감소량 등을
    시각적인 꺾은선 그래프(Plot) 이미지로 만들어 저장하는 함수입니다.
    """
    df = df.copy()
    # groupby와 cumcount를 이용해 같은 모양_케이스 그룹 내에서 1, 2, 3... 번호를 매깁니다. (x축 용도)
    df["episode_idx"] = df.groupby(["shape_type", "true_case"]).cumcount() + 1
    # 그래프 범례(Legend)에 표시할 이름 생성 (예: "WEIGHTED_SQUARE_CASE_1_A")
    df["case_label"] = df["shape_type"].astype(str) + "_" + df["true_case"].astype(str)

    # ----- 1. 성공 여부(Success) 그래프 -----
    plt.figure(figsize=(12, 6)) # 가로 12, 세로 6인치 도화지 생성
    for label, g in df.groupby("case_label"):
        g = g.sort_values("episode_idx") # 에피소드 순서대로 정렬
        # x축은 에피소드 번호, y축은 성공 여부(0 또는 1)
        plt.plot(g["episode_idx"], g["success"], marker="o", label=label)
    plt.yticks([0, 1], ["Fail", "Success"]) # y축 눈금을 Fail, Success로 명시
    plt.title("Success per Episode")
    plt.xlabel("Episode")
    plt.grid(True) # 눈금선(그리드) 표시
    plt.legend() # 우측에 범례 표시
    plt.savefig(os.path.join(OUT_DIR, "line_success.png")) # 파일로 저장
    plt.close() # 도화지 닫기 (메모리 누수 방지)

    # ----- 2. 소모된 스텝 수(Steps) 그래프 -----
    plt.figure(figsize=(12, 6))
    for label, g in df.groupby("case_label"):
        g = g.sort_values("episode_idx")
        plt.plot(g["episode_idx"], g["steps"], marker="o", label=label)
    plt.title("Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "line_steps.png"))
    plt.close()

    # ----- 3. 총 회전 각도(Rotation) 그래프 -----
    # 시스템 식별(SysID)이 잘 작동한다면, 뒤로 갈수록 불필요한 회전이 줄어드는 경향을 볼 수 있습니다.
    plt.figure(figsize=(12, 6))
    for label, g in df.groupby("case_label"):
        g = g.sort_values("episode_idx")
        plt.plot(g["episode_idx"], g["total_rotation_deg"], marker="o", label=label)
    plt.title("Rotation per Episode")
    plt.xlabel("Episode")
    plt.ylabel("deg")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "line_rotation.png"))
    plt.close()

    # ----- 4. 최종 오차(Final Error) 그래프 -----
    # 얼마나 목표에 가까이 완벽하게 주차했는지를 나타냅니다. (0에 가까울수록 좋음)
    plt.figure(figsize=(12, 6))
    for label, g in df.groupby("case_label"):
        g = g.sort_values("episode_idx")
        plt.plot(g["episode_idx"], g["final_err"], marker="o", label=label)
    plt.title("Final Error per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "line_error.png"))
    plt.close()


# 이 스크립트를 파이썬에서 직접 실행했을 때만(import 된 것이 아닐 때) run_all() 메인 함수를 가동합니다.
if __name__ == "__main__":
    run_all()