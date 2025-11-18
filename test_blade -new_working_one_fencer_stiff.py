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

def draw_fencing_blade(image, landmarks, image_width, image_height, scale_factor):
    """
    Draw a fencing blade extending from the wrist in the arm's direction.

    :param image: OpenCV image frame
    :param landmarks: MediaPipe landmarks
    :param image_width: Frame width
    :param image_height: Frame height
    :param scale_factor: Scaling factor to adjust blade size based on fencer's body
    """
    # Define real-world fencing blade length (approx. 1.10m for an épée or foil)
    #The size 5 épée fence is the correct legally approved length to be used by modern adult fencers. 
    # #The blade has a length of 90 cm (3 ft) and the overall length of the fence is 110 cm (3.6ft).
    REAL_BLADE_LENGTH = 1.1

    # Get wrist and elbow landmarks
    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

    # Get landmark positions for shoulders
    
    right_pinky = landmarks[mp_pose.PoseLandmark.RIGHT_PINKY]  #18
    right_index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX]  #20
    # Step 1: Compute the midpoint of segment 17-19
    mid_x =int((right_pinky.x + right_index.x) / 2 * image_width)
    mid_y =int((right_pinky.y + right_index.y) / 2 * image_height)
    #14 - right elbow
    #15 - left wrist
    #16 - right wrist
    
    # Convert to pixel coordinates
    Wx, Wy = int(wrist.x * image_width), int(wrist.y * image_height)
    Ex, Ey = int(elbow.x * image_width), int(elbow.y * image_height)

    # Compute unit direction vector from elbow to wrist
    direction_vector = np.array([Wx - Ex, Wy - Ey])
    norm = np.linalg.norm(direction_vector)

    if norm > 0:
        unit_vector = direction_vector / norm
    else:
        unit_vector = np.array([1, 0])  # Default to rightward direction if division by zero

    # Compute blade length in pixels
    blade_length_pixels = int(REAL_BLADE_LENGTH / scale_factor)
    #print("scale factor is", scale_factor)

    # Compute blade tip position
    blade_tip_x = mid_x + int(unit_vector[0] * blade_length_pixels)
    blade_tip_y = mid_y + int(unit_vector[1] * blade_length_pixels)

    # Draw the fencing blade (white line)
    cv2.line(image, (mid_x, mid_y), (blade_tip_x, blade_tip_y), (255, 255, 255), 2)

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

# Run the video processing function
process_video()
