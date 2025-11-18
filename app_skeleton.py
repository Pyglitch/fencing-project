import cv2
import mediapipe as mp
import numpy as np
import time
previous_wrist_x=None
previous_frame_time=None
wrist_velocity=[]
feedback={}
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# Blade length in meters (standard épée or foil length)
REAL_BLADE_LENGTH = 1.10
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(angle, 1)

# global var1
var1 = ''
L=0
def check_pose(landmarks, frame):

    #creating empty list for frame to then append data to
    #so each frame starts with an empty list to add wrist/pose/whatever to
    #this is because each frame is not guarranteed to even have the other data
    feedback[frame] = [] #this is the list to append the data to from the frame (if it exists)

    global var1
    # if not isinstance(landmarks, list) or len(landmarks) < 33:
    #     print("Invalid or missing landmarks in check_pose.")
    #    # return
    global L


    hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
    ankle = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
    knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE]

    angle=calculate_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))

    # #right

    hip_right  = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    ankle_right = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    knee_right = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    
    angle_right=calculate_angle((hip_right.x, hip_right.y), (knee_right.x, knee_right.y), (ankle_right.x, ankle_right.y))
    arm_part_1=landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    arm_part_2=landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
    arm_part_3=landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
    right_full_arm=calculate_angle((arm_part_1.x, arm_part_1.y),(arm_part_2.x, arm_part_2.y),(arm_part_3.x, arm_part_3.y))
    
    left_arm_part_1=landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    left_arm_part_2=landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
    left_arm_part_3=landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    left_full_arm=calculate_angle((left_arm_part_1.x, left_arm_part_1.y),(left_arm_part_2.x, left_arm_part_2.y),(left_arm_part_3.x, left_arm_part_3.y))
    # #duck
    # if angle < 90 and angle_right < 90:
    #     if var1!="duck":
    #         var1="duck"
    #         feedback.append("You did a duck. Frame "+str(frame))
    # if angle_right>100:
    #     feedback.append('RightLaeg. Frame'+str(frame))
    # if angle>150 and angle<190:
    #     feedback.append('LeeftLaeg. Frame'+str(frame))
    # if left_full_arm>170 and left_full_arm<190:
    #     feedback.append('LeeftAerm. Frame'+str(frame))
    #infighting
    # if abs(left_arm_part_1.x-hip.x)<=0.025: 







        # print(str(abs(left_arm_part_1.x-hip.x))+" frame "+str(frame))



    if abs(left_arm_part_1.x-hip.x)>=0.025:
        # print("You leaned. Frame "+str(frame))
        
        feedback[frame].append('lean')
        #new feedback style
        
        L+=1
    else:
        feedback[frame].append('No lean')
        
        wrist_velocity
    if angle_right>100 and angle>150 and angle<190 and left_full_arm>170 and left_full_arm<190 and hip.y<left_arm_part_3.y:
        if var1!="Foot":
            var1="Foot"
            feedback[frame].append('Toe touch')

    # #left knee angle
    # angle=calculate_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
    # #right knee angle
    # angle_right=calculate_angle((hip_right.x, hip_right.y), (knee_right.x, knee_right.y), (ankle_right.x, ankle_right.y))
    
    # # print(angle)
    
    
   
def calculate_scale_factor(hands, landmarks, image_width, image_height):
    """
    Compute scale factor based on the shoulder-to-elbow distance.
    This factor ensures the fencing blade length is proportional to the fencer's body size.

    :param landmarks: List of pose landmarks from MediaPipe
    :param image_width: Width of the image
    :param image_height: Height of the image
    :return: Scale factor for fencing blade length
    """
    if hands==0:
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    else:

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

def draw_flexible_fencing_blade(hands, image, landmarks, image_width, image_height, scale_factor, frame_count, bend_intensity=0.2, bend_speed=0.15):
    """
    Draw a flexible fencing blade using a quadratic Bézier curve with tunable bend intensity and frequency.

    :param image: OpenCV image frame
    :param landmarks: MediaPipe landmarks
    :param image_width: Frame width
    :param image_height: Frame height
    :param scale_factor: Scaling factor for blade length
    :param frame_count: Used for oscillation effect
    :param bend_intensity: Controls how much the blade bends (default = 0.2)
    :param bend_speed: Controls the speed of tip oscillation (default = 0.15)
    """
    REAL_BLADE_LENGTH = 1.1  # Standard fencing blade length in meters
    if hands==0:
        wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        pinky = landmarks[mp.solutions.pose.PoseLandmark.LEFT_PINKY]
        index = landmarks[mp.solutions.pose.PoseLandmark.LEFT_INDEX]
        # print(wrist)
    else:
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
    bend_amplitude = blade_length_pixels * bend_intensity  # Adjustable bend intensity
    wave_offset = bend_amplitude * np.sin(frame_count * bend_speed)  # Oscillating effect

    P1_x = int((P0_x + P2_x) / 2 + unit_vector[1] * wave_offset)  # Perpendicular to blade
    P1_y = int((P0_y + P2_y) / 2 - unit_vector[0] * wave_offset)  # Perpendicular to blade

    # Draw Quadratic Bézier Curve
    num_segments = 200  # Number of segments for smooth curve
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
    
    # print("Are you left or right handed?")
    hands=int(input('''0 is for the left hand while 1 is for the right hand'''))
    # cap = cv2.VideoCapture("flech_Sam.mp4")
    # cap = cv2.VideoCapture("Infights.mp4")
    cap = cv2.VideoCapture("Toe_touch_v2.mp4")
    # cap = cv2.VideoCapture("Toe_touch_right_handed.mp4")  # Load video
    # cap = cv2.VideoCapture("Toe_touch_v2.mp4")
    
    frame_count = 0  # Track frames for animation

    while cap.isOpened():
        current_time=time.time()
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
            scale_factor = calculate_scale_factor(hands, results.pose_landmarks.landmark, w, h)

            # Draw flexible fencing blade with Bézier curve
            draw_flexible_fencing_blade(hands, image, results.pose_landmarks.landmark, w, h, scale_factor, frame_count, bend_intensity=0.008, bend_speed=0.008)
            global previous_wrist_x, previous_frame_time, wrist_velocity



            # Get wrist landmark based on hand selection
            if hands == 0:
                wrist_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            else:
                wrist_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Current wrist x-coordinate (normalized)
            current_wrist_x = wrist_landmark.x

            # Calculate time delta if we have previous data
            if previous_wrist_x is not None and previous_frame_time is not None:
                time_delta = current_time - previous_frame_time
                if time_delta > 0:
                    vx = abs((current_wrist_x - previous_wrist_x) / time_delta)  # velocity in normalized units/sec
                    print(frame_count, vx)
                    wrist_velocity.append((frame_count, vx))  # store frame and velocity
                    feedback[frame_count].append(vx)

            
            # Update previous wrist x and time
            previous_wrist_x = current_wrist_x
            previous_frame_time = current_time




            #pose checking
            # print("run pose check")
            check_pose(results.pose_landmarks.landmark, frame_count)
            
        # Display the output
        cv2.imshow('Fencing Blade Overlay', image)

        frame_count += 1  # Update frame count for animation

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the video processing function
#process_video()
process_video()
# print(feedback)
with open("feedback.txt", 'w') as file:
    file.write("")
file.close()
print(feedback)
#now writing to file after its cleared
with open("feedback.txt", 'w') as file:

    # for line in wrist_velocity:
    #     file.write(str(line)+ "\n") 
    for line in feedback:
        file.write(str(line) + "\n")

    file.close()

print("\n\nDONE WRITING\n\n")
print('???')
# print(feedback[0])
# print(wrist_velocity[0])

print(len(wrist_velocity))
print(len(feedback))

#first two
# print("wrist vx")
# print(wrist_velocity[0])
# print(wrist_velocity[1])
# print()
# print(wrist_velocity[-1])
# print(wrist_velocity[-2])
# print()
# print()
# print("feedback")
# print(feedback[0])
# print(feedback[1])
# print()
# print(feedback[-1])
# print(feedback[-2])


max=wrist_velocity[0][1]
# print(L)
# for line in wrist_velocity:
#     if line[1]>max:
#         max=line[1]
# print(max)
max_frame=1

'''
list 1              list 2 
feedback            wrist_velocity
1 pose              1 1.3
2 pose              2 1.54658
3 lean?             3 1.5
4 no lean?          4 4.6



****

new_list with all of it

1, pose, wrist_vx





'''



for i in range(len(feedback)):
    if "No lean" in feedback[i]:
        # print("HERE1")
        if max<=wrist_velocity[i][1]:
            # print("HERE2")
            max=wrist_velocity[i][1]
            max_frame=wrist_velocity[i][0]
            # print(max)
            # print(max_frame)
print(max)
print(max_frame)
# print(wrist_velocity)

# print(len(wrist_velocity))
# print(len(feedback))

