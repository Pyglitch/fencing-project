import cv2
import json
import csv
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Initialize MediaPipe Pose modules globally
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Load YOLO Model ---
def load_yolo_model():
    model = YOLO("yolov5su.pt")
    model.conf = 0.3
    model.iou = 0.45
    model.classes = [0]  # Person only
    model.imgsz = 640
    return model

# --- Detect people using YOLO ---
def detect_people_yolo(yolo_model, frame):
    results = yolo_model(frame)[0]
    people = []
    for det in results.boxes:
        cls_id = int(det.cls[0].item())
        conf = float(det.conf[0].item())
        if cls_id == 0 and conf > 0.3:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            people.append((x1, y1, x2, y2))
    return people

# --- Run BlazePose on cropped person image ---
def run_blazepose_on_crop(pose_model, crop_image):
    crop_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
    return pose_model.process(crop_rgb)

# --- Draw landmarks and export ---
def draw_landmarks_on_frame(frame, results, x1, y1, crop_w, crop_h, resized_crop, frame_idx, person_idx, landmark_data):
    if not results.pose_landmarks:
#        print("⚠️ No landmarks detected.")
        return

#   print("✅ Landmarks found! Drawing now...")

    # Step 1: Draw landmarks on the resized crop
    annotated_crop = resized_crop.copy()
    mp_drawing.draw_landmarks(
        annotated_crop,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )

    # Step 2: Resize the annotated crop back to original crop size
    annotated_crop_resized = cv2.resize(annotated_crop, (crop_w, crop_h))

    # Step 3: Paste it back into the original frame
    frame[y1:y1 + crop_h, x1:x1 + crop_w] = annotated_crop_resized

    # Step 4: Save landmark data for export
    for id, lm in enumerate(results.pose_landmarks.landmark):
        landmark_data.append({
            "frame": frame_idx,
            "person": person_idx,
            "id": id,
            "x": lm.x,
            "y": lm.y,
            "z": lm.z,
            "visibility": lm.visibility
        })

# --- Main pipeline ---
def process_video_with_yolo_blazepose(video_path, output_video_path, json_path, csv_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
#        print("❌ Error: Video could NOT be opened.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    yolo_model = load_yolo_model()
    pose_model = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    frame_idx = 0
    landmark_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        people_boxes = detect_people_yolo(yolo_model, frame)

        for person_idx, (x1, y1, x2, y2) in enumerate(people_boxes):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Fencer", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.shape[0] == 0 or person_crop.shape[1] == 0:
                continue

            crop_h, crop_w = person_crop.shape[:2]
            if crop_h < 100 or crop_w < 100:
#                print("❌ Skipping tiny crop - likely a false positive")
                continue

            resized_crop = cv2.resize(person_crop, (384, 384))
            results = run_blazepose_on_crop(pose_model, resized_crop)

            draw_landmarks_on_frame(
                frame, results, x1, y1, crop_w, crop_h, resized_crop,
                frame_idx, person_idx, landmark_data
            )

        cv2.putText(frame, "DRAW TEST", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)
        cv2.imshow("YOLO + BlazePose (Multi-Fencer)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # --- Export JSON ---
    with open(json_path, "w") as jf:
        json.dump(landmark_data, jf, indent=2)

    # --- Export CSV ---
    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["frame", "person", "id", "x", "y", "z", "visibility"])
        writer.writeheader()
        writer.writerows(landmark_data)

# --- Run the Script ---
if __name__ == "__main__":
    process_video_with_yolo_blazepose(
        video_path="C:/Users/flora/Documents/FencingProject/Sam_video_trim_shorter.mp4",
        output_video_path="C:/Users/flora/Documents/FencingProject/output_annotated.mp4",
        json_path="C:/Users/flora/Documents/FencingProject/pose_landmarks.json",
        csv_path="C:/Users/flora/Documents/FencingProject/pose_landmarks.csv"
    )
