import cv2
import torch
import numpy as np
import mediapipe as mp

# Load YOLOv5 model
def load_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

# Detect people using YOLOv5
def detect_people_yolo(yolo_model, frame):
    results = yolo_model(frame)
    people = []

    for *box, conf, cls in results.xyxy[0]:
        if int(cls) == 0 and conf > 0.5:  # 'person' class
            x1, y1, x2, y2 = map(int, box)
            people.append((x1, y1, x2, y2))
    return people

# Run BlazePose on a cropped person image
def run_blazepose_on_crop(pose_model, crop_image):
    crop_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
    results = pose_model.process(crop_rgb)
    return results

# Main driver
def process_video_with_yolo_blazepose(video_path):
    yolo_model = load_yolo_model()
    
    mp_pose = mp.solutions.pose
    pose_model = mp_pose.Pose(
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Video file could not be opened.")
        return

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame (maybe end of video).")
            break

        print(f"📽 Processing frame {frame_count}")

        people_boxes = detect_people_yolo(yolo_model, frame)
        print(f"🔎 Detected {len(people_boxes)} person(s) in frame {frame_count}")


        for (x1, y1, x2, y2) in people_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.imshow("YOLO + BlazePose (Multi-Fencer)", frame)

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.shape[0] == 0 or person_crop.shape[1] == 0:
                continue

            resized_crop = cv2.resize(person_crop, (256, 256))
            results = run_blazepose_on_crop(pose_model, resized_crop)

            if results.pose_landmarks:
                # Reproject landmarks to original frame
                for lm in results.pose_landmarks.landmark:
                    lx = int(lm.x * (x2 - x1)) + x1
                    ly = int(lm.y * (y2 - y1)) + y1
                    cv2.circle(frame, (lx, ly), 3, (0, 255, 0), -1)

                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

        cv2.imshow("YOLO + BlazePose (Multi-Fencer)", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("👋 Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()
