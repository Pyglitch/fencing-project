from flask import Flask, json, jsonify
from flask.sansio.scaffold import T_template_context_processor
from flask_cors import CORS
from flask_cors.extension import make_after_request_function
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(angle, 1)



name=Flask(__name__)
CORS(name)

def video_processer(video_source):
    LR=int(input('''Are you on the left side or the right side?
    Enter 1 for right and 0 for left:'''))
    while LR>=2:
        LR=int(input('Do it again'))
    after_y=1
    after_ya=1
    #previous frame knee
    pre_1=0
    pre_2=0
    pre_1a=0
    pre_2a=0
    #previous toes
    pre_foot=0
    pre_foot2=0
    pre_foota=0
    pre_foot2a=0
    feedback=[] #this is what will store the poses
    #standard pose
    #lunge pose
    mpDraw = mp.solutions.drawing_utils
    model_path = "pose_landmarker_full.task"
    num_poses = 2
    min_pose_detection_confidence = 0.5
    min_pose_presence_confidence = 0.5
    min_tracking_confidence = 0.5
    current_frame_output = None
    last_timestamp_ms = 0
    base_options = python.BaseOptions(model_asset_path=model_path) 
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(video_source)
    fps = int(cap.get(5))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width,frame_height)
    output = cv2.VideoWriter('Sam.avi', cv2.VideoWriter_fourcc(*'MP4V'), fps, frame_size)
    var1=""
    var2=""
    famerconmure=0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_landmarks_list = result.pose_landmarks
            frame = np.copy(mp_image.numpy_view())
            noses = []
            for person in range(len(pose_landmarks_list)):
                noses.append(pose_landmarks_list[person][0].x)
            noses.sort()
            #person is for each fencer
            for person in range(len(pose_landmarks_list)):
                pose_landmarks = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks.landmark.extend([
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x,
                        y=landmark.y,
                        z=landmark.z) for landmark in pose_landmarks_list[person]
                ])
                if len(noses)==1 or len(noses)==0:
                    continue

                
                # nose_x = pose_landmarks_list[person][0].x
                # if abs(nose_x - noses[LR]) < 0.01:

                if (pose_landmarks_list[person][0].x == noses[LR]):
                    
                    #stuff for player 1
                    #left hip
                    hip = pose_landmarks_list[person][23]
                    #right hip
                    hip_right = pose_landmarks_list[person][24]
                    #left ankle
                    ankle = pose_landmarks_list[person][27]
                    #right ankle
                    ankle_right = pose_landmarks_list[person][28]
                    #left knee
                    knee = pose_landmarks_list[person][25]
                    #right knee
                    knee_right = pose_landmarks_list[person][26]

                    print(knee.x)
                    
                    #angles
                    #left knee angle
                    angle=calculate_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
                    #right knee angle
                    angle_right=calculate_angle((hip_right.x, hip_right.y), (knee_right.x, knee_right.y), (ankle_right.x, ankle_right.y))
                    #arm
                    arm_part_1=pose_landmarks_list[person][12]
                    arm_part_2=pose_landmarks_list[person][14]
                    arm_part_3=pose_landmarks_list[person][16]
                    full_arm=calculate_angle((arm_part_1.x, arm_part_1.y),(arm_part_2.x, arm_part_2.y),(arm_part_3.x, arm_part_3.y))
                    #checking for duck
                    #duck for YOU (or player 1)
                
                    if full_arm>=170 and full_arm<=190:
                        feedback.append("straight right arm " + str(famerconmure))
                        # if LR == 1: #right side
                            #  if ankle_right.x >= pre_foot2.x or ankle<=pre_foot.x:
                                #  if var1!="countattack":
                                    # var1="countattack"
                                    # feedback.append("You did a counterattack. Frame "+str(famerconmure))
                        # elif LR == 0: #left side
                    #        if var1!="countattack":
                     #               var1="countattack"
                      #              feedback.append("Your opponent did a counterattack. Frame "+str(famerconmure))                        
                        if ankle_right.x<=pre_foot2.x or ankle.x<=pre_foot.x:
                            if var1!="countattack":
                                var1="countattack"
                                feedback.append("You did a counterattack. Frame"+str(famerconmure))
                    #could probably just use angle var
                    if angle < 90 and angle_right < 90:
                        if var1!="duck":
                            var1="duck"
                            feedback.append("You did a duck. Frame "+str(famerconmure))
#Add lunge code to the enemy
                    if 100<angle_right and angle>170 and full_arm>170:
                        if ankle_right>=pre_foot2:    
                            if var1!="lunge":
                                var1="lunge"
                                feedback.append("You did a lunge. Frame"+str(famerconmure))
#player detection for lunge
                    arm_y=pose_landmarks_list[person][16].y
                    if arm_y-after_y>0.1:
                        if var1!="parry":
                            var1="parry"
                            feedback.append("You did a parry")
                    after_y=arm_y
                    #right knee
                    knee_1=pose_landmarks_list[person][26].x
                    #left knee
                    knee_2=pose_landmarks_list[person][25].x



                    '''
                    ***FLECH CODE*** copied from line 14-50
                    update flech using hand/arm angle
                        and footwork
                        
                    have a slight error for angle
                    doesnt have to be completely 180 degrees

                    #EASIER stuff
                    '''
                    #player 1

                    #left side
                    wrist = pose_landmarks_list[person][15]
                    elbow = pose_landmarks_list[person][13]
                    shoulder = pose_landmarks_list[person][11]

                    #right side
                    #just add the rest
                    wrist_right = pose_landmarks_list[person][16]
                    elbow_right = pose_landmarks_list[person][14]
                    shoulder_right = pose_landmarks_list[person][12]

                    #player 2

                    #left side
                    wrista = pose_landmarks_list[person][15]
                    elbowa = pose_landmarks_list[person][13]
                    shouldera = pose_landmarks_list[person][11]

                    #right side
                    wrista_right = pose_landmarks_list[person][16]
                    elbowa_right = pose_landmarks_list[person][14]
                    shouldera_right = pose_landmarks_list[person][12]

                    if knee_1<knee_2 and pre_1>pre_2 and 170 <= calculate_angle((wrist.x,wrist.y), (elbow.x, elbow.y), (shoulder.x,shoulder.y)) <= 185:
                        if var1!="flech":
                            var1="flech"
                            feedback.append("You did a flech. Franme "+str(famerconmure))

                    if knee_1>knee_2 and pre_1<pre_2 and 170 <= calculate_angle((wrist.x,wrist.y), (elbow.x, elbow.y), (shoulder.x,shoulder.y)) <= 185:
                        if var1!="flech":
                            var1="flech"
                            feedback.append("You did a flech")
                    
                    # print(var1)
                    
                    '''

                    #for update flech
                    #check angle of hand/arm for straight

                    #if knee_1<knee_2 and pre_1>pre_2:
                    #    feedback.append("You did a flech")
                    #if knee_1>knee_2 and pre_1<pre_2:
                    #    feedback.append("You did a flech")

                    '''
                    '''
                    landmarks 11, 13, 15 for left side
                    12, 14, 16 for right side
                    '''

                    pre_foot=ankle
                    pre_foot2=ankle_right
                    pre_1=knee_1
                    pre_2=knee_2

                    mpDraw.draw_landmarks(frame, pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS, mp.solutions.drawing_styles.get_default_pose_landmarks_style())
                    

                    #opponents stuff (player 2)
                    #left hip
                    hipa = pose_landmarks_list[person][23]
                    #right hip
                    hipa_right = pose_landmarks_list[person][24]
                    #left ankle
                    anklea = pose_landmarks_list[person][27]
                    #right ankle
                    anklea_right = pose_landmarks_list[person][28]
                    #left knee
                    kneea = pose_landmarks_list[person][25]
                    #right knee
                    kneea_right = pose_landmarks_list[person][26]
                    #left knee angle
                    anglea=calculate_angle((hipa.x, hipa.y), (kneea.x, kneea.y), (anklea.x, anklea.y))
                    #right knee angle
                    anglea_right=calculate_angle((hipa_right.x, hipa_right.y), (kneea_right.x, kneea_right.y), (anklea_right.x, anklea_right.y))

                    #checking for duck (player 2)
                    if anglea < 90 and anglea_right < 90:
                        if var2!="ducke":
                            var2="ducke"
                            feedback.append("Your oponent ducked. Frame "+str(famerconmure))

                    if 100<anglea:
                         if var2!="lungedyu":
                            var2="lungedyu"
                            feedback.append("Your oponent lunged. Frame "+str(famerconmure))
                    arm_ya=pose_landmarks_list[person][16].y
                    if arm_ya-after_ya>0.1:
                        if var2!="parried":
                            var2="parried"
                            feedback.append("Your oponent parried")
                    after_ya=arm_ya
                    knee_1a=pose_landmarks_list[person][26].x
                    knee_2a=pose_landmarks_list[person][25].x
                    if knee_1<knee_2 and pre_1>pre_2 and 170 <= calculate_angle((wrist.x,wrist.y), (elbow.x, elbow.y), (shoulder.x,shoulder.y)) <= 185:
                        if var2!="flech":
                            var2="flech"
                        feedback.append("You did a flech")

                    if knee_1>knee_2 and pre_1<pre_2 and 170 <= calculate_angle((wrist.x,wrist.y), (elbow.x, elbow.y), (shoulder.x,shoulder.y)) <= 185:
                        if var2!="flech":
                            var2="flech"
                            feedback.append("You did a flech")


                    # lean testing
                    if abs(shoulder.x-hip.x)>=1:
                        feedback.append("You leaned on frame "+str(famerconmure))
                    else:
                        feedback.append("Not lean "+str(famerconmure))

                    pre_1a=knee_1a
                    pre_2a=knee_2a
                    mpDraw.draw_landmarks(frame, pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS, mp.solutions.drawing_styles.get_default_pose_landmarks_style())
                    #this is printing pose landmark coords
                    # print(pose_landmarks_list[person][1].x)
            output.write(frame)
        else:
            print("Stream disconnected")
            break
        famerconmure+=1
    cap.release()
    cv2.destroyAllWindows()


    
    #clearing file for new output
    with open("feedback.txt", 'w') as file:
        file.write("")
    file.close()

    #now writing to file after its cleared
    with open("feedback.txt", 'w') as file:
        for line in feedback:
            file.write(str(line) + "\n")
        # file.write(feedback)
        file.close()
    
    print("\n\nDONE WRITING\n\n")

    return feedback


#getting video frame
def get_frame(frame_number_input, video_file_name):
    print("getting frame")
    myFrameNumber = frame_number_input -1
    cap = cv2.VideoCapture(video_file_name)

    # get total number of frames
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # check for valid frame number
    if myFrameNumber >= 0 & myFrameNumber <= totalFrames:
        # set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES,myFrameNumber)

    ret, frame = cap.read()
    # cv2.imshow("Frame", frame)
#    cv2.imwrite("video_frame.jpg", frame)
    cv2.imwrite("video_frame.jpg", frame)
    cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()
    print("frame got")



video_processer("Toe_touch_v2.mp4")
# video_processer("counter_attack.mp4")
# video_processer("toe_touch_new.mp4")

#reminder: divide line by 2 to get the frame
# get_frame(135,"counter_attack.mp4")
#get_frame(770,"Video of ducks and arrows.mp4")
# get_frame(19,"Simon_trim.mp4")



#can do this much later
# if __name__ == '__main__':
#   name.run(host='0.0.0.0')
#   video_processer("Sam_video_trim_shorter.mp4")

#   video_processer("Sam_video_trim.mp4")