import cv2
import torch
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

print("Starting test script...")

def load_yolo_model():
    print("🔄 Inside load_yolo_model()...")
    model=YOLO("yolov5s.pt")
#    model = torch.hub.load('yolov5', 'yolov5s', source='local', force_reload=True)
#    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    print("✅ YOLO model returned.")
    model.conf = 0.3  # Confidence threshold
    model.iou = 0.45  # IoU threshold for NMS
    model.classes = [0]  # Person only
    model.imgsz = 640  # Try 640 or even 960

    return model

def detect_people_yolo(yolo_model, frame):
    print("Running YOLO detection...")
    results = yolo_model(frame)
    people = []
    for *box, conf, cls in results.xyxy[0]:
        if int(cls) == 0 and conf > 0.3:
            x1, y1, x2, y2 = map(int, box)
            people.append((x1, y1, x2, y2))
    print(f"Detected {len(people)} person(s)")
    return people

def run_blazepose_on_crop(pose_model, crop_image):
    crop_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
    return pose_model.process(crop_rgb)

def process_video_with_yolo_blazepose(video_path):
    print("Opening video...")
    cap = cv2.VideoCapture(video_path)
    print(f"Trying to open video: {video_path}")

    if not cap.isOpened():
        print("❌ Error: Video could NOT be opened.")
        return
    else:
        print("✅ Video opened successfully!")

    print("Loading YOLO model...")
    yolo_model = load_yolo_model()
    print("✅ YOLO model loaded!")

    mp_pose = mp.solutions.pose
    pose_model = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("End of video.")
            break

        print(f"Frame {frame_count}")
        people_boxes = detect_people_yolo(yolo_model, frame)
        print(f"🔎 Detected {len(people_boxes)} person(s) in frame {frame_count}")

        for (x1, y1, x2, y2) in people_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Fencer", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLO + BlazePose (Multi-Fencer)", frame)

        

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

# Actually run the function here
if __name__ == "__main__": 
    print("Starting test script...")
    process_video_with_yolo_blazepose("C:/Users/flora/Documents/FencingProject/Sam_video_trim.mp4")
   