import cv2, os, urllib.request, numpy as np, mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


## 모델 파트 ##
# 파일 경로 (포즈/얼굴/손)
POSE_MODEL_PATH = 'pose_landmarker_heavy.task'
FACE_MODEL_PATH = 'face_landmarker_v2_with_blendshapes.task'
HAND_MODEL_PATH = 'gesture_recognizer.task'

# 모델 파일 다운로드 함수
def download_model(model_path, url):
    if not os.path.exists(model_path):
        print(f"모델 다운로드 중중 {os.path.basename(model_path)}...")
        urllib.request.urlretrieve(url, model_path)

# 모델 다운로드 링크 (포즈/얼굴/손)
download_model(POSE_MODEL_PATH, 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task')
download_model(FACE_MODEL_PATH, 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task')
download_model(HAND_MODEL_PATH, 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task')

# 미디어파이프 설정
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

## 랜드마크 파트 ##
# 랜드마크 표시 함수
def draw_landmarks_on_image(rgb_image, pose_detection_result, face_detection_result, hand_detection_result):
    annotated_image = np.copy(rgb_image)

    # 랜드마크 측정 (포즈/얼굴/손)
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

## 모델 초기화 파트 ##
# 모델 초기화 (포즈/얼굴/손)
pose_base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
pose_options = vision.PoseLandmarkerOptions(base_options=pose_base_options,
                                            output_segmentation_masks=False)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

face_base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
face_options = vision.FaceLandmarkerOptions(base_options=face_base_options,
                                            output_face_blendshapes=False,
                                            output_facial_transformation_matrixes=False,
                                            num_faces=1)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

hand_base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
hand_options = vision.GestureRecognizerOptions(base_options=hand_base_options, num_hands=2) 
hand_recognizer = vision.GestureRecognizer.create_from_options(hand_options)

## 실행 파트 ##
# 웹캠 실행
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라 시스템 오류 발생! 카메라 시스템을 점검하십시오")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임 오류 발생! 카메라 시스템을 점검하십시오")
        continue

    # 미디어파이프 입력 형식으로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 추론 실행 (포즈/얼굴/손)
    pose_detection_result = pose_detector.detect(mp_image)
    face_detection_result = face_detector.detect(mp_image)
    hand_detection_result = hand_recognizer.recognize(mp_image)

    # 인식 결과 시각화
    annotated_frame = draw_landmarks_on_image(rgb_frame, pose_detection_result, face_detection_result, hand_detection_result)

    # 제스처 출력 (Print only the highest)
    if hand_detection_result.gestures:
        for hand_index, gestures in enumerate(hand_detection_result.gestures):
            if gestures:
                top_gesture = gestures[0]
                gesture_name = top_gesture.category_name
                gesture_score = top_gesture.score
                cv2.putText(annotated_frame, f"Hand {hand_index+1}: {gesture_name} ({gesture_score:.2f})", (10, 30+hand_index*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 결과 프레임 표시
    cv2.imshow("HMOS_Pilot1", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()