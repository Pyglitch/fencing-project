import cv2
import mediapipe as mp
import numpy as np
import time

# ---------------------- Fencer Class ---------------------- #
class Fencer:
    def __init__(self, lunge=False, duck=False, toe_touch=False, flech=False):
        self.lunge = lunge
        self.duck = duck
        self.toe_touch = toe_touch
        self.flech = flech

    def same_frame(self, feedback_list):
        poses = {
            'lunge': self.lunge,
            'duck': self.duck,
            'toe_touch': self.toe_touch,
            'flech': self.flech
        }
        feedback_list.append(poses)

# ---------------------- Globals ---------------------- #
lunge_count = 0
duck_count = 0
toe_touch_count = 0
flech_count = 0
previous_wrist_x = None
previous_frame_time = None
wrist_velocity = []
feedback = []
hands = 0  # Left=0, Right=1

# ---------------------- Mediapipe Setup ---------------------- #
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---------------------- Angle Calculation ---------------------- #
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(angle, 1)

# ---------------------- Pose Detection ---------------------- #
def check_pose(landmarks, frame):
    global lunge_count, duck_count, toe_touch_count, flech_count

    fencer = Fencer()

    # Left leg
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    angle_left_leg = calculate_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))

    # Right leg
    hip_r = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    knee_r = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    ankle_r = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    angle_right_leg = calculate_angle((hip_r.x, hip_r.y), (knee_r.x, knee_r.y), (ankle_r.x, ankle_r.y))

    # Right arm
    shoulder_r = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    elbow_r = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    wrist_r = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    right_arm_angle = calculate_angle((shoulder_r.x, shoulder_r.y), (elbow_r.x, elbow_r.y), (wrist_r.x, wrist_r.y))

    # Left arm
    shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    elbow_l = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    wrist_l = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    left_arm_angle = calculate_angle((shoulder_l.x, shoulder_l.y), (elbow_l.x, elbow_l.y), (wrist_l.x, wrist_l.y))

    # Pose checks
    if hands == 0:
        if angle_right_leg > 170 and angle_left_leg < 100 and 170 < left_arm_angle < 190:
            fencer.lunge = True
            lunge_count += 1
    else:
        if angle_right_leg < 100 and angle_left_leg > 170 and right_arm_angle > 170:
            fencer.lunge = True
            lunge_count += 1

    if angle_left_leg < 90 and angle_right_leg < 90:
        fencer.duck = True
        duck_count += 1

    if 100 < angle_right_leg and 150 < angle_left_leg < 190 and 170 < left_arm_angle < 190 and hip.y < wrist_l.y:
        fencer.toe_touch = True
        toe_touch_count += 1

    fencer.same_frame(feedback)

# ---------------------- Scale Factor ---------------------- #
def calculate_scale_factor(hands, landmarks, image_width, image_height):
    if hands == 0:
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    else:
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

    sx, sy = int(shoulder.x * image_width), int(shoulder.y * image_height)
    ex, ey = int(elbow.x * image_width), int(elbow.y * image_height)
    pixel_length = np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
    real_length = 0.35  # meters
    return real_length / pixel_length if pixel_length > 0 else 1

# ---------------------- Blade Drawing ---------------------- #
def draw_flexible_fencing_blade(hands, image, landmarks, width, height, scale_factor, frame_count, bend_intensity=0.008, bend_speed=0.008):
    blade_length_m = 1.1

    if hands == 0:
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        pinky = landmarks[mp_pose.PoseLandmark.LEFT_PINKY]
        index = landmarks[mp_pose.PoseLandmark.LEFT_INDEX]
    else:
        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        pinky = landmarks[mp_pose.PoseLandmark.RIGHT_PINKY]
        index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX]

    Wx, Wy = int(wrist.x * width), int(wrist.y * height)
    Px, Py = int(pinky.x * width), int(pinky.y * height)
    Ix, Iy = int(index.x * width), int(index.y * height)

    P0_x = int((Px + Ix) / 2)
    P0_y = int((Py + Iy) / 2)

    direction = np.array([P0_x - Wx, P0_y - Wy])
    norm = np.linalg.norm(direction)
    unit_vector = direction / norm if norm != 0 else np.array([1, 0])

    blade_pixels = int(blade_length_m / scale_factor)
    P2_x = P0_x + int(unit_vector[0] * blade_pixels)
    P2_y = P0_y + int(unit_vector[1] * blade_pixels)

    wave_offset = blade_pixels * bend_intensity * np.sin(frame_count * bend_speed)
    P1_x = int((P0_x + P2_x) / 2 + unit_vector[1] * wave_offset)
    P1_y = int((P0_y + P2_y) / 2 - unit_vector[0] * wave_offset)

    bezier_points = []
    for t in np.linspace(0, 1, 200):
        x = int((1 - t) ** 2 * P0_x + 2 * (1 - t) * t * P1_x + t ** 2 * P2_x)
        y = int((1 - t) ** 2 * P0_y + 2 * (1 - t) * t * P1_y + t ** 2 * P2_y)
        bezier_points.append((x, y))

    for i in range(len(bezier_points) - 1):
        cv2.line(image, bezier_points[i], bezier_points[i + 1], (255, 255, 255), 2)

    return image

# ---------------------- Video Processing ---------------------- #
def process_video():
    global hands, previous_wrist_x, previous_frame_time

    hands = int(input("Enter 0 for left hand or 1 for right hand: "))
    cap = cv2.VideoCapture("Toe_touch_v2.mp4")
    frame_count = 0

    while cap.isOpened():
        current_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape

        if results.pose_landmarks:
            check_pose(results.pose_landmarks.landmark,)