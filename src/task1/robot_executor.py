import time

try:
    import config
except Exception:
    config = None


class CloudGripperExecutor:
    def __init__(self, robot_client):
        """
        robot_client: CloudGripper API 객체 (실제 GripperRobot 또는 Mock)
        """
        self.robot = robot_client

        # Z축을 쓸 수 없으므로 높이 관련 변수는 전부 삭제했습니다.
        # sleep 값은 config.py에서 조절 가능하게 둡니다.
        self.SLEEP_TIME = float(getattr(config, "EXECUTOR_SLEEP_TIME", 0.5)) if config is not None else 0.5
        self.PATH_SLEEP_TIME = float(getattr(config, "EXECUTOR_PATH_SLEEP_TIME", 0.05)) if config is not None else 0.05
        self.PUSH_SLEEP_TIME = float(getattr(config, "EXECUTOR_PUSH_SLEEP_TIME", 1.0)) if config is not None else 1.0

    def execute_push(self, plan: dict):
        """
        Brain이 만든 plan 딕셔너리를 받아 로봇의 2D 평면 순차 동작으로 풀어냅니다.
        """
        if plan is None:
            return False

        cand = plan["candidate"]
        motion = plan["motion"]

        # A* 플래너가 만든 안전한 우회 경로 또는 직선/L자 접근 경로
        approach_path = motion["approach_path"]
        start_xy = cand["start"]
        end_xy = cand["end"]
        retreat_xy = motion.get("retreat_xy")

        print("[Executor] Z축 고정(Task1) 모드로 Pushing Plan 실행!")

        try:
            # 1. 물체를 피해 시작점까지 접근
            print(f"  1) 장애물 우회하여 접근 (총 {len(approach_path)}개 웨이포인트)")
            for idx, pt in enumerate(approach_path):
                self.robot.move_xy(float(pt[0]), float(pt[1]))
                time.sleep(self.PATH_SLEEP_TIME)

            # 확실하게 시작점에 안착하기 위해 한 번 더 명령 및 딜레이
            self.robot.move_xy(float(start_xy[0]), float(start_xy[1]))
            time.sleep(self.SLEEP_TIME)

            # 2. 물체 밀기
            print(f"  2) 푸시(Push) 진행 중... 목표점: ({end_xy[0]:.2f}, {end_xy[1]:.2f})")
            self.robot.move_xy(float(end_xy[0]), float(end_xy[1]))
            time.sleep(self.PUSH_SLEEP_TIME)

            # 3. 후퇴
            if retreat_xy is not None:
                print("  3) 푸시 완료 후 후퇴 (Retreat)")
                self.robot.move_xy(float(retreat_xy[0]), float(retreat_xy[1]))
                time.sleep(self.SLEEP_TIME)

            return True

        except Exception as e:
            print(f"[Executor] 🚨 로봇 제어 중 에러 발생: {e}")
            return False
