import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math


import cv2

model_path = r"C:\Users\imish\Python_Local\Hackathon\pose_landmarker_full.task"

import time

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None #result
latest_timestamp = 0 #time
LEFT_HIP = 23 #setting the points to each part of body q
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
def distance(a,b):
    return math.sqrt((a.x-b.x)**2+(a.y-b.y)**2) #distance formula 

# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    global latest_timestamp

    latest_result = result #save the latest result
    latest_timestamp = timestamp_ms #save latest timestamp 
    print("Detected poses:", len(result.pose_landmarks)) #prints how mnay people were detected


def draw(frame, result):
    if result is None:
        return frame
    if not result.pose_landmarks:
        return frame
    height, width, _ = frame.shape #reutrn hiehgt,wdith, and color
    for person  in result.pose_landmarks:
        for landmark in person:
            x = int(landmark.x * width) #reutrns graphing points like 0.25 meaning 25% into the frame, but needs pixels so multiplies by widht and height
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    return frame

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_poses=4,
    min_pose_detection_confidence=0.4, #atleast 50 confidence to detec t apose
    min_pose_presence_confidence=0.5, #50% presense to detect post
    min_tracking_confidence=0.5,
    result_callback=print_result #intilization
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


if not cap.isOpened():
    print("Camera could not be opened.")
    exit()

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        works, frame = cap.read()

        if not works:
            break

        
        frame = cv2.flip(frame, 1) #mirror it

        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
        )  #convert BGR to RGB for mediapipe

        
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        #milliseconds
        timestamp_ms = int(time.time() * 1000)

       #frame to mediapipe
        landmarker.detect_async(mp_image, timestamp_ms) #send to mediaframe for processing

        #draw latest result
        frame = draw(frame, latest_result)

        draw(frame, latest_result)
        cv2.imshow("Live Pose Landmarker", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
        







    