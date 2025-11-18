import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize Pose Model with confidence thresholds
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

'''''
# Function to calculate custom points A and B based on existing pose landmarks
def calculate_new_points(landmarks, image_width, image_height):
    """
    Calculate two new points A and B from existing landmarks.
    
    Point A: Midpoint between left shoulder (11) and right shoulder (12).
    Point B: Midpoint between left hip (23) and right hip (24).

    :param landmarks: List of pose landmarks from MediaPipe
    :param image_width: Width of the image
    :param image_height: Height of the image
    :return: Tuple containing coordinates of points A and B
    """

    # Get landmark positions for shoulders
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # Compute midpoint (A) between shoulders
    Ax = (left_shoulder.x + right_shoulder.x) / 2 * image_width
    Ay = (left_shoulder.y + right_shoulder.y) / 2 * image_height

    # Get landmark positions for hips
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    # Compute midpoint (B) between hips
    Bx = (left_hip.x + right_hip.x) / 2 * image_width
    By = (left_hip.y + right_hip.y) / 2 * image_height

    # Return integer coordinates for OpenCV drawing
    return (int(Ax), int(Ay)), (int(Bx), int(By))
'''''

# Ask user for fencer handedness (Left or Right)
#fencer1_hand = input("Is Fencer 1 left-handed or right-handed? (L/R): ").strip().upper()
#fencer2_hand = input("Is Fencer 2 left-handed or right-handed? (L/R): ").strip().upper()

def get_wrist_and_elbow(landmarks, is_left_handed, image_width, image_height):
    """
    Returns wrist and elbow coordinates based on handedness.
    :param landmarks: MediaPipe landmark list
    :param is_left_handed: True for left-handed, False for right-handed
    :param image_width: Image width
    :param image_height: Image height
    :return: (Wx, Wy, Ex, Ey) wrist and elbow coordinates
    """
    if is_left_handed:
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    else:
        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

    Wx, Wy = int(wrist.x * image_width), int(wrist.y * image_height)
    Ex, Ey = int(elbow.x * image_width), int(elbow.y * image_height)

    return Wx, Wy, Ex, Ey

def calculate_new_points(landmarks, image_width, image_height):
    #    Point A: Midpoint between left shoulder (11) and right shoulder (12).
    #    Point B: Midpoint between left hip (23) and right hip (24).

    #:param landmarks: List of pose landmarks from MediaPipe
    #:param image_width: Width of the image
    #:param image_height: Height of the image
    #:return: Tuple containing coordinates of points A and B
    #15 - left wrist
    #16 - right wrist
    #17 - left pinky
    #18 - right pinky
    #19 - left index

    # Get landmark positions for shoulders
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]  #15
    left_pinky = landmarks[mp_pose.PoseLandmark.LEFT_PINKY]  #17
    left_index = landmarks[mp_pose.PoseLandmark.LEFT_INDEX]  #19


    # Step 1: Compute the midpoint of segment 17-19
    mid_x = (left_pinky.x + left_index.x) / 2
    mid_y = (left_pinky.y + left_index.y) / 2

    # Step 2: Compute the direction vector of segment 17-19
    Vx = (left_index.x - left_pinky.x)
    Vy = (left_index.y - left_pinky.y)

    # Step 3: Compute the perpendicular vector by rotating 90 degrees
    V_perp_x = Vx
    V_perp_y = -Vy

    # Step 4: Normalize the perpendicular vector
    magnitude = np.sqrt(V_perp_x**2 + V_perp_y**2)
    V_perp_x_norm = V_perp_x / magnitude
    V_perp_y_norm = V_perp_y / magnitude

    # Step 5: Determine the correct side (extend away from point 15)
    vec_to_15_x = left_wrist.x - mid_x
    vec_to_15_y = left_wrist.y - mid_y

    # Compute dot product to check alignment
    dot_product = vec_to_15_x * V_perp_x_norm + vec_to_15_y * V_perp_y_norm

    # If dot product is **negative**, flip direction (so it moves away from 15)
    if dot_product < 0:
        V_perp_x_norm *= 1
        V_perp_y_norm *= 1

    L=0.9 #blade length 90cm
    # Step 6: Scale the perpendicular vector to length L
    V_perp_x_scaled = V_perp_x_norm * L
    V_perp_y_scaled = V_perp_y_norm * L

    # Step 7: Compute the endpoint of the perpendicular line (starting from node 15)
    P_end_x = (left_wrist.x + V_perp_x_scaled) * image_width
    P_end_y = (left_wrist.y + V_perp_y_scaled) * image_height

   # Step 8: starting from node 15
    P_start_x = int(left_wrist.x) * image_width
    P_start_y = int(left_wrist.y) * image_height
    #you need to apply * image_width and * image_height to the existing point left_wrist to convert the normalized coordinate (ranging from 0.0 to 1.0 from MediaPipe) into actual pixel coordinates for OpenCV drawing.

   # MediaPipe outputs landmark coordinates in normalized form, meaning:

   # left_wrist.x is a ratio of the image width (e.g., 0.5 means halfway across the image).
   # left_wrist.y is a ratio of the image height.
   # OpenCV, on the other hand, expects pixel coordinates, so you must scale them to the actual image size.

    # Return integer coordinates for OpenCV drawing
    return (int(P_start_x), int(P_start_y),int(P_end_x), int(P_end_y))

    # Plotting
    #plt.figure(figsize=(6, 6))

    # Plot original segment 17-19
    #plt.plot([x17, x19], [y17, y19], 'bo-', label='Segment 17-19')

    # Plot perpendicular line extending away from node 15
    #plt.plot([x15, P_end_x], [y15, P_end_y], 'r-', linewidth=2,
    #         label='Perpendicular Line (Away from 15, Length L)')

    # Plot original points, midpoint, and point 15
    #plt.scatter([x15, x17, x19, x15, P_end_x], [y15, y17, y19, y15, P_end_y], color='black', s=100,
    #            label='Points 15, 17, 19 & blade start and end points')


    # Labels and display
    #plt.axhline(0, color='gray', linewidth=0.5)
    #plt.axvline(0, color='gray', linewidth=0.5)
    #plt.grid(True, linestyle='--', alpha=0.6)
    #plt.legend()
    #plt.title("Perpendicular Line Extended Away from Node 15 (Triangle Structure)")
    #plt.show()

# Example: Define coordinates for nodes 15, 17, and 19, ensuring they form a triangle
#x15, y15 = 2, 5  # Coordinates for point 15 (distinct from line 17-19)
#x17, y17 = 3, 3  # Coordinates for point 17
#x19, y19 = 7, 6  # Coordinates for point 19
#L = 40  # Length of the perpendicular line

# Call the function to plot perpendicular line extending away from node 15
#plot_perpendicular_line_away_from_15(x15, y15, x17, y17, x19, y19, L)

def calculate_scale_factor(landmarks, image_width, image_height):
    """
    Compute scale factor based on the shoulder-to-elbow distance.
    This factor ensures the fencing blade length is proportional to the fencer's body size.

    :param landmarks: List of pose landmarks from MediaPipe
    :param image_width: Width of the image
    :param image_height: Height of the image
    :return: Scale factor for fencing blade length
    """
    # Get shoulder and elbow landmarks (assuming right-handed fencer)
    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

    # Convert normalized coordinates to pixel coordinates
    Sx, Sy = int(shoulder.x * image_width), int(shoulder.y * image_height)
    Ex, Ey = int(elbow.x * image_width), int(elbow.y * image_height)

    # Compute Euclidean distance in pixels
    pixel_arm_length = np.sqrt((Ex - Sx) ** 2 + (Ey - Sy) ** 2)

    # Real-world reference arm length (adjust as needed, e.g., average adult arm length ≈ 0.35m)
    #Reduce the value (e.g., 0.30m for younger fencers). Increase the value (e.g., 0.38m for taller fencers).
    REAL_ARM_LENGTH = 0.40  

    # Compute scale factor: meters per pixel
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

    # Compute blade tip position
    blade_tip_x = mid_x + int(unit_vector[0] * blade_length_pixels)
    blade_tip_y = mid_y + int(unit_vector[1] * blade_length_pixels)

    # Draw the fencing blade (white line)
    cv2.line(image, (mid_x, mid_y), (blade_tip_x, blade_tip_y), (255, 255, 255), 3)

def process_video():
    cap = cv2.VideoCapture("flech_Sam.mp4")  # Load fencing match video

    if not cap.isOpened():
        print("Error: Video file not found or cannot be opened.")
        return  # Exit if video is not found

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape  # Get frame dimensions

        if results.pose_landmarks:
            print("Detected fencer(s)... Processing landmarks")

            # Ensure fencers are sorted based on left-right positioning
            fencer_positions = [(landmark.x * w, i) for i, landmark in enumerate(results.pose_landmarks.landmark)]
            fencer_positions.sort(key=lambda x: x[0])  # Sort by x-coordinate

            if len(fencer_positions) >= 2:
                # Assign fencers based on x-coordinates
                fencer1_landmarks = results.pose_landmarks.landmark
                fencer2_landmarks = results.pose_landmarks.landmark

                # Get wrist and elbow positions
                Wx1, Wy1, Ex1, Ey1 = get_wrist_and_elbow(fencer1_landmarks, fencer1_is_left_handed, w, h)
                Wx2, Wy2, Ex2, Ey2 = get_wrist_and_elbow(fencer2_landmarks, fencer2_is_left_handed, w, h)

                # Compute scaling factor (to convert meters to pixels)
                scale_factor1 = np.sqrt((Ex1 - Wx1) ** 2 + (Ey1 - Wy1) ** 2) / 0.35  # Arm length as reference
                scale_factor2 = np.sqrt((Ex2 - Wx2) ** 2 + (Ey2 - Wy2) ** 2) / 0.35  # Arm length as reference

                # Draw fencing blades
                draw_fencing_blade(image, Wx1, Wy1, Ex1, Ey1, scale_factor1, (0, 0, 255))  # Red for Fencer 1
                draw_fencing_blade(image, Wx2, Wy2, Ex2, Ey2, scale_factor2, (255, 0, 0))  # Blue for Fencer 2

            # Draw pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show the video frame
        cv2.imshow("Fencing Blade Overlay", image)

        key = cv2.waitKey(30) & 0xFF  # Delay for smooth playback
        if key == ord("q"):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()


process_video()    