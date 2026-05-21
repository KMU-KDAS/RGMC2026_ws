import cv2 # 컴퓨터 비전(이미지 처리)을 위한 OpenCV 라이브러리. 여기선 다각형을 그리고 색칠하는 데 쓰입니다.
import numpy as np # 행렬 및 벡터 계산을 위한 필수 라이브러리

from geometry_utils import make_circle_polygon, wrap_angle # 수학/기하학 계산 헬퍼 함수들 임포트


class Planner:
    def __init__(self, config):
        # 클래스가 생성될 때 config(설정 파일)를 통째로 받아옵니다.
        self.config = config
        # 물체의 그림자를 그릴 도화지(마스크 이미지)의 해상도(세로, 가로)를 설정합니다. (예: 256x256)
        self.mask_shape = (config.MASK_H, config.MASK_W)

    @staticmethod
    def wrap_angle(angle):
        # 각도가 -180도 ~ 180도(-pi ~ pi) 범위를 벗어나지 않게 예쁘게 말아주는 헬퍼 함수입니다.
        return wrap_angle(angle)

    def get_projected_corners(self, current_pose, current_corners, target_pose):
        # "현재 모양 그대로, 목표 위치(target_pose)로 이동한다면 꼭짓점들이 어디에 찍힐까?"를 계산하는 함수입니다.
        cx, cy, ctheta = current_pose # 현재 중심점 x, y와 각도
        tx, ty, ttheta = target_pose  # 목표 중심점 x, y와 각도
        
        # 목표 각도와 현재 각도의 차이(회전해야 할 량)를 구합니다.
        dtheta = self.wrap_angle(ttheta - ctheta)
        
        # 2D 평면 회전 변환 행렬(Rotation Matrix)을 만듭니다.
        rot = np.array([
            [np.cos(dtheta), -np.sin(dtheta)],
            [np.sin(dtheta),  np.cos(dtheta)]
        ])
        
        projected = [] # 예상되는 목표 꼭짓점 좌표들을 담을 리스트
        for gx, gy in current_corners:
            # 1. 월드 좌표(gx, gy)를 현재 중심점(cx, cy) 기준의 로컬 좌표계로 뺍니다. (물체의 중심을 0,0으로 맞춤)
            local = np.array([gx - cx, gy - cy])
            
            # 2. 로컬 좌표를 회전(rot @ local)시키고, 그 결과를 통째로 목표 중심점(tx, ty)으로 평행 이동(+)시킵니다.
            moved = rot @ local + np.array([tx, ty])
            
            # 계산된 새로운 꼭짓점 좌표를 리스트에 추가합니다.
            projected.append((float(moved[0]), float(moved[1])))
        return projected

    def world_to_mask(self, polygon_world):
        # 물리적 미터(m) 단위나 정규화(0~1) 단위로 된 다각형 꼭짓점 리스트를, 
        # 컴퓨터 화면(이미지)의 픽셀 좌표 리스트로 한꺼번에 변환해 줍니다.
        return [self.to_mask_point(x, y) for x, y in polygon_world]

    def to_mask_point(self, x, y):
        # 실제 공간의 좌표 (x, y)를 이미지 상의 픽셀 좌표 (px, py)로 매핑하는 핵심 함수입니다.
        xmin, ymin = self.config.WORKSPACE_MIN # 작업 공간 최소값 (예: 0.0)
        xmax, ymax = self.config.WORKSPACE_MAX # 작업 공간 최대값 (예: 1.0)
        
        w = self.config.MASK_W - 1 # 이미지의 가로 최대 픽셀 인덱스 (예: 255)
        h = self.config.MASK_H - 1 # 이미지의 세로 최대 픽셀 인덱스 (예: 255)
        
        # X축 변환: (현재값 - 최소값) / (최대값 - 최소값) 으로 비율을 구한 뒤, 이미지 너비(w)를 곱해 픽셀 위치를 찾습니다.
        # np.clip으로 값이 이미지 밖으로 나가지 않게 자르고(0~w), 정수(int)로 만듭니다.
        px = int(np.clip(round((x - xmin) / (xmax - xmin) * w), 0, w))
        
        # Y축 변환: 컴퓨터 그래픽스에서는 보통 맨 위가 y=0이고 아래로 갈수록 y가 커집니다.
        # 반면 우리가 쓰는 물리 좌표계는 위로 갈수록 y가 커지죠.
        # 이 차이를 맞추기 위해 h에서 계산된 픽셀값을 빼주어 상하를 뒤집습니다(Inversion).
        py = int(np.clip(round(h - (y - ymin) / (ymax - ymin) * h), 0, h))
        
        return px, py # 변환된 픽셀 좌표 (x, y) 반환

    @staticmethod
    def calculate_iou(corners_a_px, corners_b_px, img_shape=(256, 256)):
        # ★ 이 파일의 존재 이유! IoU(Intersection over Union, 교집합/합집합 비율)를 계산합니다.
        # 물체가 목표 다각형에 얼마나 완벽하게 겹쳐있는지 0.0 ~ 1.0 사이의 퍼센트로 알려줍니다.
        
        # 1. 텅 빈 검은색 도화지(마스크) 두 장을 준비합니다. (하나는 현재 물체용, 하나는 목표용)
        mask_a = np.zeros(img_shape, dtype=np.uint8)
        mask_b = np.zeros(img_shape, dtype=np.uint8)
        
        # 2. OpenCV가 이해할 수 있는 형태(3차원 배열)로 꼭짓점 리스트를 변환합니다.
        poly_a = np.array(corners_a_px, dtype=np.int32).reshape((-1, 1, 2))
        poly_b = np.array(corners_b_px, dtype=np.int32).reshape((-1, 1, 2))
        
        # 3. 빈 도화지에 다각형의 외곽선을 그리고, 그 안을 하얗게(255) 꽉 채워서 색칠합니다. (-1은 꽉 채우라는 뜻)
        cv2.drawContours(mask_a, [poly_a], -1, 255, -1)
        cv2.drawContours(mask_b, [poly_b], -1, 255, -1)
        
        # 4. 교집합(Intersection) 영역 넓이 구하기
        # 두 이미지에서 동시에 하얗게 칠해진 부분(AND 연산)의 픽셀 개수를 셉니다.
        intersection = np.count_nonzero(cv2.bitwise_and(mask_a, mask_b))
        
        # 5. 합집합(Union) 영역 넓이 구하기
        # 두 이미지 중 하나라도 하얗게 칠해진 부분(OR 연산)의 픽셀 개수를 셉니다.
        union = np.count_nonzero(cv2.bitwise_or(mask_a, mask_b))
        
        # 6. IoU 계산 = 교집합 / 합집합
        # 겹치는 면적이 하나도 없으면 0.0, 완전히 똑같이 겹치면 1.0(100%)이 됩니다.
        return 0.0 if union == 0 else float(intersection / union)

    @staticmethod
    def make_circle_polygon(center, radius, num_points=64):
        # 외부 모듈(geometry_utils)에 있는 원형 다각형 생성 함수를 
        # 이 클래스 안에서도 쉽게 가져다 쓸 수 있도록 감싸놓은(Wrapper) 함수입니다.
        return make_circle_polygon(center, radius, num_points)