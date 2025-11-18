import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Blade length in meters (standard épée or foil length)
REAL_BLADE_LENGTH = 1.10  

def calculate_scale_factor(landmarks, image_width, image_height):
    """
    Compute scale factor based on the shoulder-to-elbow distance.
    This factor ensures the fencing blade length is proportional to the fencer's body size.

    :param landmarks: List of pose landmarks from MediaPipe
    :param image_width: Width of the image
    :param image_height: Height of the image
    :return: Scale factor for fencing blade length
    """
    # Get shoulder and elbow landmarks (right arm assumed)
    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

    # Convert normalized coordinates to pixel values
    Sx, Sy = int(shoulder.x * image_width), int(shoulder.y * image_height)
    Ex, Ey = int(elbow.x * image_width), int(elbow.y * image_height)

    # Compute Euclidean distance in pixels
    pixel_arm_length = np.sqrt((Ex - Sx) ** 2 + (Ey - Sy) ** 2)

    # Real-world shoulder-to-elbow length (adjustable, e.g., 0.35m)
    REAL_ARM_LENGTH = 0.35  

    # Compute scale factor (meters per pixel)
    scale_factor = REAL_ARM_LENGTH / pixel_arm_length if pixel_arm_length > 0 else 1

    return scale_factor

import numpy as np
import cv2
import mediapipe as mp

def draw_fencing_blade(image, landmarks, image_width, image_height, scale_factor):
    """
    Draw a fencing blade extending from the hand orientation.

    :param image: OpenCV image frame
    :param landmarks: MediaPipe landmarks
    :param image_width: Frame width
    :param image_height: Frame height
    :param scale_factor: Scaling factor to adjust blade size based on fencer's body
    """
    REAL_BLADE_LENGTH = 1.1  # Meters, standard fencing blade length

    # Get wrist, pinky, and index finger landmarks
    wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
    pinky = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_PINKY]
    index = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_INDEX]

    # Convert to pixel coordinates
    Wx, Wy = int(wrist.x * image_width), int(wrist.y * image_height)
    Px, Py = int(pinky.x * image_width), int(pinky.y * image_height)
    Ix, Iy = int(index.x * image_width), int(index.y * image_height)

    # Step 1: Compute midpoint of index and pinky fingers
    mid_x = int((Px + Ix) / 2)
    mid_y = int((Py + Iy) / 2)

    # Step 2: Compute the direction vector from the wrist to the midpoint
    direction_vector = np.array([mid_x - Wx, mid_y - Wy])
    norm = np.linalg.norm(direction_vector)

    if norm > 0:
        unit_vector = direction_vector / norm
    else:
        unit_vector = np.array([1, 0])  # Default rightward direction

    # Compute blade length in pixels
    blade_length_pixels = int(REAL_BLADE_LENGTH / scale_factor)

    # Compute blade tip position
    blade_tip_x = mid_x + int(unit_vector[0] * blade_length_pixels)
    blade_tip_y = mid_y + int(unit_vector[1] * blade_length_pixels)

    # Draw the fencing blade (white line)
    cv2.line(image, (mid_x, mid_y), (blade_tip_x, blade_tip_y), (255, 255, 255), 1)

    return image

def draw_flexible_fencing_blade(image, landmarks, image_width, image_height, scale_factor, frame_count):
    """
    Draw a flexible fencing blade using a quadratic Bézier curve.

    :param image: OpenCV image frame
    :param landmarks: MediaPipe landmarks
    :param image_width: Frame width
    :param image_height: Frame height
    :param scale_factor: Scaling factor
    :param frame_count: Used for oscillation effect
    """
    REAL_BLADE_LENGTH = 1.1  # Standard fencing blade length in meters

    # Get wrist, pinky, and index finger landmarks
    wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
    pinky = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_PINKY]
    index = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_INDEX]

    # Convert to pixel coordinates
    Wx, Wy = int(wrist.x * image_width), int(wrist.y * image_height)
    Px, Py = int(pinky.x * image_width), int(pinky.y * image_height)
    Ix, Iy = int(index.x * image_width), int(index.y * image_height)

    # Compute midpoint of index and pinky fingers (base of the blade)
    P0_x = int((Px + Ix) / 2)
    P0_y = int((Py + Iy) / 2)

    # Compute direction vector from wrist to midpoint
    direction_vector = np.array([P0_x - Wx, P0_y - Wy])
    norm = np.linalg.norm(direction_vector)

    if norm > 0:
        unit_vector = direction_vector / norm
    else:
        unit_vector = np.array([1, 0])  # Default rightward direction

    # Compute blade length in pixels
    blade_length_pixels = int(REAL_BLADE_LENGTH / scale_factor)

    # Compute blade tip position (P2)
    P2_x = P0_x + int(unit_vector[0] * blade_length_pixels)
    P2_y = P0_y + int(unit_vector[1] * blade_length_pixels)

    # Compute control point (P1) for Bézier curve - Simulates tip bending
    bend_amplitude = blade_length_pixels * 0.06  # Adjustable bend intensity (2% of blade length)
    wave_offset = -bend_amplitude * np.sin(frame_count * 0.06)  # Oscillating effect

    P1_x = int((P0_x + P2_x) / 2 + unit_vector[1] * wave_offset)  # Perpendicular to blade
    P1_y = int((P0_y + P2_y) / 2 - unit_vector[0] * wave_offset)  # Perpendicular to blade

    # Draw Quadratic Bézier Curve
    num_segments = 1000  # Number of segments for smooth curve
    bezier_points = []

    for t in np.linspace(0, 1, num_segments):
        x = int((1 - t) ** 2 * P0_x + 2 * (1 - t) * t * P1_x + t ** 2 * P2_x)
        y = int((1 - t) ** 2 * P0_y + 2 * (1 - t) * t * P1_y + t ** 2 * P2_y)
        bezier_points.append((x, y))

    # Draw the flexible Bézier curve
    for i in range(len(bezier_points) - 1):
        cv2.line(image, bezier_points[i], bezier_points[i + 1], (255, 255, 255), 2)

    return image

def process_video():
    cap = cv2.VideoCapture("flech_Sam.mp4")   # Webcam feed (change to 'video.mp4' for a video file)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back for OpenCV

        h, w, _ = image.shape  # Get frame dimensions

        if results.pose_landmarks:
            # Draw the existing pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Compute scale factor using the fencer's shoulder-to-elbow distance
            scale_factor = calculate_scale_factor(results.pose_landmarks.landmark, w, h)

            # Draw fencing blade
            draw_fencing_blade(image, results.pose_landmarks.landmark, w, h, scale_factor)


        # Display the output
        cv2.imshow('Fencing Blade Overlay', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_flexible_balde_video(vidoe_flie):
    cap = cv2.VideoCapture(vidoe_flie)  # Load video
    frame_count = 0  # Track frames for oscillation effect

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back for OpenCV

        h, w, _ = image.shape  # Get frame dimensions

        if results.pose_landmarks:
            # Draw pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Compute scale factor based on fencer's shoulder-to-elbow distance
            scale_factor = calculate_scale_factor(results.pose_landmarks.landmark, w, h)

            # Draw flexible fencing blade
            draw_flexible_fencing_blade(image, results.pose_landmarks.landmark, w, h, scale_factor, frame_count)

        # Display the output
        cv2.imshow('Fencing Blade Overlay', image)

        frame_count += 1  # Update frame count for animation

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the video processing function
#process_video()
process_flexible_balde_video("flech_Sam.mp4")
