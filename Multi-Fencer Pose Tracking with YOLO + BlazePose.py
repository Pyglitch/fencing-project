'''
pip install torch torchvision torchaudio
pip install opencv-python mediapipe numpy
git clone https://github.com/ultralytics/yolov5  # Clone YOLOv5
cd yolov5
pip install -r requirements.txt
'''
import torch

# Load YOLOv5 model (v5s is fastest; switch to v5m/v5l for accuracy)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def detect_people_yolo(frame):
    results = yolo_model(frame)
    people = []

    for *box, conf, cls in results.xyxy[0]:
        if int(cls) == 0 and conf > 0.5:  # class 0 is 'person'
            x1, y1, x2, y2 = map(int, box)
            people.append((x1, y1, x2, y2))  # Bounding box
    return people

import mediapipe as mp

mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

import cv2
import numpy as np
from yolov5 import detect_people_yolo  # if saved separately

cap = cv2.VideoCapture("Simon_trim.mp4")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    people_boxes = detect_people_yolo(frame_rgb)

    for (x1, y1, x2, y2) in people_boxes:
        person_crop = frame_rgb[y1:y2, x1:x2]
        if person_crop.shape[0] == 0 or person_crop.shape[1] == 0:
            continue  # Skip bad crops

        # Resize to 256x256 for better BlazePose input
        resized_crop = cv2.resize(person_crop, (256, 256))
        results = pose_model.process(resized_crop)

        if results.pose_landmarks:
            # Scale landmarks back to original coordinates
            for lm in results.pose_landmarks.landmark:
                lx = int(lm.x * (x2 - x1)) + x1
                ly = int(lm.y * (y2 - y1)) + y1
                cv2.circle(frame, (lx, ly), 3, (0, 255, 0), -1)

            # Optional: draw pose connections (rough estimate)
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

    cv2.startWindowThread()

    cv2.imshow("YOLO + BlazePose (Multi-Fencer)", frame)



    
    
    
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
