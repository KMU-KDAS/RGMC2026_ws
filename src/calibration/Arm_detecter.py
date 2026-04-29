import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------------
# 1. 모델 로드 및 영상/이미지 읽기
# -----------------------------------
model_path = r"C:\Users\wlsdud\Desktop\gripper_dataset_yolo\gripper_project\train_v14\weights\last.pt"
img_path = r"C:\Users\wlsdud\Desktop\Dataset example\data-sample\sample_CloudGripper-Rope-100\robot01\episode00000000\image_bottom\data\5.jpg"

# YOLO 모델 로드
model = YOLO(model_path)

# OpenCV로 이미지 읽기
image = cv2.imread(img_path)

if image is None:
    print("Image read failed")
    exit()

output = image.copy()

# -----------------------------------
# 2. YOLO 모델 추론 (기존 HSV 변환 및 마스크 생성 대체)
# -----------------------------------
# 이미지를 모델에 넣고 결과 리스트 반환
results = model(image)

detected = False

# -----------------------------------
# 3. 마스크(Contour) 및 센터 좌표 추출
# -----------------------------------
for result in results:
    # 마스크(세그멘테이션 결과)가 존재하는지 확인
    if result.masks is not None:
        # masks.xy는 외곽선 좌표들의 리스트입니다. 첫 번째 객체의 좌표를 가져옵니다.
        # 기존 cv2.findContours()로 찾은 contour와 동일한 역할을 합니다.
        contour_points = result.masks.xy[0]
        
        # OpenCV 함수에서 사용할 수 있도록 float32 -> int32 데이터 타입으로 변환
        contour = np.array(contour_points, dtype=np.int32)

        # OpenCV moments를 이용해 마스크의 무게중심(Center) 계산
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            center = (cX, cY)
            detected = True
            
            # -----------------------------------
            # 4. 결과 출력 및 시각화
            # -----------------------------------
            print("Gripper detected")
            print("Center:", center)

            # 외곽선 표시 (노란색)
            cv2.drawContours(output, [contour], -1, (0, 255, 255), 2)

            # 중심점 표시 (빨간색)
            cv2.circle(output, center, 5, (0, 0, 255), -1)

            cv2.putText(
                output,
                f"Center: {center}",
                (center[0] + 10, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )
            
            # 하나의 객체만 처리한다고 가정하고 루프 종료
            break

if not detected:
    print("No gripper detected in this image")

# -----------------------------------
# 5. 화면 출력
# -----------------------------------
cv2.imshow("Original", image)
cv2.imshow("Detected Gripper Center", output)

cv2.waitKey(0)
cv2.destroyAllWindows()