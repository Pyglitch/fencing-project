import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_scale_factor(landmarks, is_left_handed, image_width, image_height):
    """
    Compute scale factor based on the shoulder-to-elbow distance.
    """
    if is_left_handed:
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    else:
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    
    shoulder_px = (int(shoulder.x * image_width), int(shoulder.y * image_height))
    elbow_px = (int(elbow.x * image_width), int(elbow.y * image_height))
    
    distance = np.linalg.norm(np.array(shoulder_px) - np.array(elbow_px))
    
    REAL_ARM_LENGTH = 0.35
    return distance / REAL_ARM_LENGTH

def draw_flexible_fencing_blade(image, landmarks, is_left_handed, image_width, image_height, scale_factor, frame_count, bend_intensity=0.2, bend_speed=0.15):
    """
    Draw a flexible fencing blade using a quadratic Bézier curve with tunable bend intensity and frequency.
    """
    REAL_BLADE_LENGTH = 1.1  # Standard fencing blade length in meters
    
    if is_left_handed:
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        pinky = landmarks[mp_pose.PoseLandmark.LEFT_PINKY]
        index = landmarks[mp_pose.PoseLandmark.LEFT_INDEX]
    else:
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
    
    if norm > 0:
        unit_vector = direction_vector / norm
    else:
        unit_vector = np.array([1, 0])
    
    blade_length_pixels = int(REAL_BLADE_LENGTH / scale_factor)
    
    P2_x = P0_x + int(unit_vector[0] * blade_length_pixels)
    P2_y = P0_y + int(unit_vector[1] * blade_length_pixels)
    
    bend_amplitude = blade_length_pixels * bend_intensity
    wave_offset = bend_amplitude * np.sin(frame_count * bend_speed)
    
    P1_x = int((P0_x + P2_x) / 2 + unit_vector[1] * wave_offset)
    P1_y = int((P0_y + P2_y) / 2 - unit_vector[0] * wave_offset)
    
    num_segments = 200
    bezier_points = []
    
    for t in np.linspace(0, 1, num_segments):
        x = int((1 - t) ** 2 * P0_x + 2 * (1 - t) * t * P1_x + t ** 2 * P2_x)
        y = int((1 - t) ** 2 * P0_y + 2 * (1 - t) * t * P1_y + t ** 2 * P2_y)
        bezier_points.append((x, y))
    
    for i in range(len(bezier_points) - 1):
        cv2.line(image, bezier_points[i], bezier_points[i + 1], (255, 255, 255), 2)
    
    return image

def detect_score(blade_tip, opponent_landmarks, image_width, image_height):
    """Detects if the blade tip enters the opponent's valid target area."""
    global fencer1_score, fencer2_score

    # Define torso target area (average of shoulder and hip position)
    left_shoulder = opponent_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = opponent_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = opponent_landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = opponent_landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    torso_x = int((left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4 * image_width)
    torso_y = int((left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4 * image_height)

    # Check if blade tip is close to the torso (within 30 pixels)
    if abs(blade_tip[0] - torso_x) < 30 and abs(blade_tip[1] - torso_y) < 30:
        return True
    return False

def process_video():
    cap = cv2.VideoCapture("flech_Sam.mp4")  # Load video
    frame_count = 0  # Track frames for animation
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
                
                scale_factor = calculate_scale_factor(results.pose_landmarks.landmark, fencer1_is_left_handed, w, h)
                draw_flexible_fencing_blade(image, results.pose_landmarks.landmark, fencer1_is_left_handed, w, h, scale_factor, frame_count, bend_intensity=0.008, bend_speed=0.008)
                
                # Determine handedness of fencer 2 and draw their blade
                #fencer2_is_left_handed = not fencer1_is_left_handed  # Assuming fencer 2 is opposite handed
                #draw_flexible_fencing_blade(image, results.pose_landmarks.landmark, fencer2_is_left_handed, w, h, scale_factor, frame_count, bend_intensity=0.008, bend_speed=0.008)

            cv2.startWindowThread()
            cv2.namedWindow("Fencing Blade Overlay", cv2.WINDOW_NORMAL)

            cv2.imshow('Fencing Blade Overlay', image)
            frame_count += 1
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    fencer1_side = input("Is Fencer 1 on the left or right side? (L/R): ").strip().upper()
    fencer1_is_left_handed = input("Is Fencer 1 left-handed or right-handed? (L/R): ").strip().upper() == "L"
    print(fencer1_is_left_handed)
    process_video()