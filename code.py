import cv2, os, urllib.request, numpy as np, mediapipe as mp, time, threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# 모델 경로
POSE_MODEL_PATH = 'pose_landmarker_heavy.task'
FACE_MODEL_PATH = 'face_landmarker_v2_with_blendshapes.task'
HAND_MODEL_PATH = 'gesture_recognizer.task'

# 모델 다운로드 함수
def download_model(model_path, url):
    if not os.path.exists(model_path):
        print(f"모델 다운로드 중: {os.path.basename(model_path)}...")
        urllib.request.urlretrieve(url, model_path)
        print(f"모델 다운로드 완료: {os.path.basename(model_path)}")

# 모델 다운로드 링크
download_model(POSE_MODEL_PATH, 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task')
download_model(FACE_MODEL_PATH, 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task')
download_model(HAND_MODEL_PATH, 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task')

# 미디어파이프 설정
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# 랜드마크 표시 함수
def draw_landmarks_on_image(rgb_image, pose_detection_result, face_detection_result, hand_detection_result):
    annotated_image = rgb_image.copy()  # 필요한 경우에만 복사

    if pose_detection_result.pose_landmarks:
      pose_landmarks_list = pose_detection_result.pose_landmarks
      for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            mp_drawing.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing_styles.get_default_pose_landmarks_style())

    if face_detection_result.face_landmarks:
       face_landmarks_list = face_detection_result.face_landmarks
       for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

    if hand_detection_result.hand_landmarks:
        for hand_landmarks in hand_detection_result.hand_landmarks:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    return annotated_image

# MediaPipe 모델 초기화 (NPU 및 GPU 우선 사용, 없을 경우 CPU 사용)
use_npu = False  # NPU 사용 여부 확인 (현재 MediaPipe에서 NPU 직접 지원은 제한적)
use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0

print(f"NPU 사용: {use_npu}")
print(f"GPU 사용: {use_gpu}")

if use_npu:
    # NPU 사용 (향후 지원 시)
    # TODO: NPU delegate 지원 추가
    pose_base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    face_base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
    hand_base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
elif use_gpu:
     # CUDA 사용
    pose_base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH, delegate=python.Delegate.GPU)
    face_base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH, delegate=python.Delegate.GPU)
    hand_base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH, delegate=python.Delegate.GPU)
else:
    # CPU 사용
    pose_base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    face_base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
    hand_base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)


pose_options = vision.PoseLandmarkerOptions(base_options=pose_base_options,
                                            output_segmentation_masks=False)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

face_options = vision.FaceLandmarkerOptions(base_options=face_base_options,
                                            output_face_blendshapes=False,
                                            output_facial_transformation_matrixes=False,
                                            num_faces=1)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

hand_options = vision.GestureRecognizerOptions(base_options=hand_base_options, num_hands=2)
hand_recognizer = vision.GestureRecognizer.create_from_options(hand_options)

# 웹캠 실행
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라 시스템 오류 발생! 카메라 시스템을 점검하십시오")
    exit()

# 창 설정
cv2.namedWindow("HMOS_Pilot1", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HMOS_Pilot1", 800, 600)

# 프레임 처리 빈도 조절 (프레임 건너뛰기)
frame_skip_interval = 2  # 프레임 건너뛰기 간격 설정 (1이면 모든 프레임 처리, 2이면 한 프레임 건너뛰고 처리)
frame_count = 0
prev_time = 0

# 도움말 메시지
help_text = "Press 'q' to quit, Click Mouse to capture"

# 마우스 클릭 이벤트 핸들러
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Mouse clicked. Triggering capture...")
        # 마우스 클릭 시 'c' 키를 누른 것과 동일한 동작 수행
        global frame_count
        frame_count = 0  # 강제로 프레임 처리

cv2.setMouseCallback("HMOS_Pilot1", mouse_callback)

# 추론 및 시각화 함수 (멀티쓰레딩)
def process_frame(frame, mp_image,annotated_frame):
    # 추론 실행
    pose_detection_result = pose_detector.detect(mp_image)
    face_detection_result = face_detector.detect(mp_image)
    hand_detection_result = hand_recognizer.recognize(mp_image)
     # 인식 결과 시각화
    annotated_frame = draw_landmarks_on_image(rgb_frame, pose_detection_result, face_detection_result, hand_detection_result)

    # 제스처 출력
    if hand_detection_result.gestures:
        for hand_index, gestures in enumerate(hand_detection_result.gestures):
            if gestures:
                top_gesture = gestures[0]
                gesture_name = top_gesture.category_name
                gesture_score = top_gesture.score
                cv2.putText(annotated_frame, f"Hand {hand_index+1}: {gesture_name} ({gesture_score:.2f})", (10, 30+hand_index*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return annotated_frame
   
while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임 오류 발생! 카메라 시스템을 점검하십시오")
        continue
    frame_count += 1
    if frame_count % frame_skip_interval != 0:
        # 건너뛸 프레임인 경우 처리하지 않고 다음 프레임으로 이동
         continue

    # 미디어파이프 입력 형식으로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

     # 멀티쓰레딩을 위한 함수 호출
    annotated_frame = process_frame(frame, mp_image, rgb_frame.copy())
    # 프레임 정보 (FPS) 표시
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 도움말 메시지 표시
    cv2.putText(annotated_frame, help_text, (10, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 결과 프레임 표시
    cv2.imshow("HMOS_Pilot1", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()