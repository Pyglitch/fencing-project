import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# --- Load YOLO Model ---
def load_yolo_model():
    model = YOLO("yolov5su.pt")
    model.conf = 0.3  # Confidence threshold
    model.iou = 0.45  # IoU threshold for NMS
    model.classes = [0]  # Person only
    model.imgsz = 640  # Input image size
    return model

# --- Detect people using YOLO ---
def detect_people_yolo(yolo_model, frame):
    results = yolo_model(frame)[0]  # Get first prediction object
    people = []

    for det in results.boxes:
        cls_id = int(det.cls[0].item())
        conf = float(det.conf[0].item())

        if cls_id == 0 and conf > 0.3:  # Person
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            people.append((x1, y1, x2, y2))

    return people
#ultralytics.YOLO format:
#results.boxes → list of boxes
#det.xyxy[0] → bounding box
#det.cls[0] → class ID
#det.conf[0] → confidence

# --- Run BlazePose on cropped person image ---
def run_blazepose_on_crop(pose_model, crop_image):
    crop_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
    return pose_model.process(crop_rgb)

# --- Reproject and draw landmarks on full frame ---
def draw_landmarks_on_frame(frame, results, x1, y1, crop_w, crop_h):
    if not results.pose_landmarks:
        print("⚠️ No landmarks detected.")
        return

    print("✅ Landmarks found! Drawing now...")
    cv2.circle(frame, (x1 + 10, y1 + 10), 5, (0, 0, 255), -1)  # Red dot for crop corner

    for landmark in results.pose_landmarks.landmark:
        lx = int(landmark.x * crop_w) + x1
        ly = int(landmark.y * crop_h) + y1

    #    print(f"Landmark pixel: ({lx}, {ly})")

        # Only draw if in frame bounds
        if 0 <= lx < frame.shape[1] and 0 <= ly < frame.shape[0]:
            cv2.circle(frame, (lx, ly), 8, (0, 0, 255), -1)


    # Optional: draw connections
    mp.solutions.drawing_utils.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
    )

# --- Main video processing pipeline ---
def process_video_with_yolo_blazepose(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Video could NOT be opened.")
        return

    yolo_model = load_yolo_model()
    mp_pose = mp.solutions.pose
    pose_model = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        people_boxes = detect_people_yolo(yolo_model, frame)

        for (x1, y1, x2, y2) in people_boxes:
            # Step 1: Draw bounding box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Fencer", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Step 2: Run BlazePose on cropped person
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.shape[0] == 0 or person_crop.shape[1] == 0:
                continue


#            Make sure the crop dimensions match the crop you're drawing from
            crop_h, crop_w = person_crop.shape[:2]
            resized_crop = cv2.resize(person_crop, (384, 384))
            cv2.imwrite(f"crop_debug_{x1}_{y1}.jpg", resized_crop)

            results = run_blazepose_on_crop(pose_model, resized_crop)
            draw_landmarks_on_frame(frame, results, x1, y1, crop_w, crop_h)
             #crop_w = 256, crop_h = 256, x1, y1 = top-left corner of the original box

        cv2.putText(frame, "DRAW TEST", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imwrite("frame_debug.jpg", frame)
        
        cv2.imshow("YOLO + BlazePose (Multi-Fencer)", frame)

        
       
        #print("✅ Frame displayed with boxes/landmarks.")

        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Run the script ---
if __name__ == "__main__":
    process_video_with_yolo_blazepose("C:/Users/flora/Documents/FencingProject/Sam_video_trim.mp4")
