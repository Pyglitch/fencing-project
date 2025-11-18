import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Blade and body length constants
REAL_BLADE_LENGTH = 1.1  # meters
REAL_ARM_LENGTH = 0.35   # meters
REAL_TORSO_LENGTH = 0.6  # meters
EXPECTED_ENGARDE_DISTANCE_METERS = 4.0

# En Garde detection state
is_engarde = False
fencer1_score = 0
fencer2_score = 0

# Define handedness
fencer1_hand = 'R'
fencer2_hand = 'R'


def calculate_scale_factor(landmarks, image_width, image_height):
    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

    Sx, Sy = int(shoulder.x * image_width), int(shoulder.y * image_height)
    Ex, Ey = int(elbow.x * image_width), int(elbow.y * image_height)

    pixel_arm_length = np.sqrt((Ex - Sx) ** 2 + (Ey - Sy) ** 2)
    return REAL_ARM_LENGTH / pixel_arm_length if pixel_arm_length > 0 else 1


def calculate_engarde_scale_factor(landmarks, image_width, image_height):
    neck = landmarks[mp_pose.PoseLandmark.NOSE]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    hip_x = (left_hip.x + right_hip.x) / 2
    hip_y = (left_hip.y + right_hip.y) / 2

    neck_x, neck_y = int(neck.x * image_width), int(neck.y * image_height)
    hip_x, hip_y = int(hip_x * image_width), int(hip_y * image_height)

    torso_length_pixels = np.sqrt((hip_x - neck_x) ** 2 + (hip_y - neck_y) ** 2)
    return REAL_TORSO_LENGTH / torso_length_pixels if torso_length_pixels > 0 else 1


def detect_en_garde(fencer1_landmarks, fencer2_landmarks, image_width, image_height):
    fencer1_foot = fencer1_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE] if fencer1_hand == "R" else fencer1_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    fencer2_foot = fencer2_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE] if fencer2_hand == "R" else fencer2_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

    fencer1_x = int(fencer1_foot.x * image_width)
    fencer1_y = int(fencer1_foot.y * image_height)
    fencer2_x = int(fencer2_foot.x * image_width)
    fencer2_y = int(fencer2_foot.y * image_height)

    distance_between_feet = np.sqrt((fencer2_x - fencer1_x) ** 2 + (fencer2_y - fencer1_y) ** 2)

    scale_factor = calculate_engarde_scale_factor(fencer1_landmarks, image_width, image_height)
    expected_distance_pixels = EXPECTED_ENGARDE_DISTANCE_METERS / scale_factor

    return expected_distance_pixels * 0.8 <= distance_between_feet <= expected_distance_pixels * 1.2


def draw_flexible_fencing_blade(image, landmarks, image_width, image_height, scale_factor, frame_count):
    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    pinky = landmarks[mp_pose.PoseLandmark.RIGHT_PINKY]
    index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX]

    Wx, Wy = int(wrist.x * image_width), int(wrist.y * image_height)
    Px, Py = int(pinky.x * image_width), int(pinky.y * image_height)
    Ix, Iy = int(index.x * image_width), int(index.y * image_height)

    P0_x = int((Px + Ix) / 2)
    P0_y = int((Py + Iy) / 2)

    direction_vector = np.array([P0_x - Wx, P0_y - Wy])
    norm = np.linalg.norm(direction_vector)

    unit_vector = direction_vector / norm if norm > 0 else np.array([1, 0])
    blade_length_pixels = int(REAL_BLADE_LENGTH / scale_factor)

    P2_x = P0_x + int(unit_vector[0] * blade_length_pixels)
    P2_y = P0_y + int(unit_vector[1] * blade_length_pixels)

    bend_amplitude = blade_length_pixels * 0.06
    wave_offset = -bend_amplitude * np.sin(frame_count * 0.06)

    P1_x = int((P0_x + P2_x) / 2 + unit_vector[1] * wave_offset)
    P1_y = int((P0_y + P2_y) / 2 - unit_vector[0] * wave_offset)

    num_segments = 100
    bezier_points = []

    for t in np.linspace(0, 1, num_segments):
        x = int((1 - t) ** 2 * P0_x + 2 * (1 - t) * t * P1_x + t ** 2 * P2_x)
        y = int((1 - t) ** 2 * P0_y + 2 * (1 - t) * t * P1_y + t ** 2 * P2_y)
        bezier_points.append((x, y))

    for i in range(len(bezier_points) - 1):
        cv2.line(image, bezier_points[i], bezier_points[i + 1], (255, 255, 255), 2)

    return image


def process_flexible_blade_video(video_file):
    global is_engarde

    cap = cv2.VideoCapture(video_file)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            scale_factor = calculate_scale_factor(results.pose_landmarks.landmark, w, h)
            draw_flexible_fencing_blade(image, results.pose_landmarks.landmark, w, h, scale_factor, frame_count)

            # Simulate both fencers using the same pose landmarks
            is_engarde = detect_en_garde(results.pose_landmarks.landmark, results.pose_landmarks.landmark, w, h)

            if is_engarde:
                cv2.putText(image, "En Garde!", (w // 2 - 120, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

        cv2.imshow("Fencing Blade Overlay with En Garde Detection", image)
        frame_count += 1

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# === Run the integrated video processor ===
process_flexible_blade_video("Sam_video_trim_shorter.mp4")
'''

MediaPipe BlazePose or
A hybrid OpenCV + YOLO approach to detect bounding boxes first, then apply pose estimation per box.'''