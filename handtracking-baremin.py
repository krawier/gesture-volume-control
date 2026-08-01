import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# using hand detecing model for now
# we should later use our own modul

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

#paramteres of Hands() -> static_image_mode when True it detects all the time so really slow
# max_num_hands -> self explenatory
# min_detection_confidence -> defeault 50% below 50% it will do the detection agains
# min_tracking_confidence -> defeault 50% below 50% it will retrack the hand

hands = mpHands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

prev_Time = 0
curr_Time = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # we want to convert to rgb bc hands only uses rgb imgs
    result = hands.process(imgRGB)

    #print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    curr_Time = time.time()
    fps = 1/(curr_Time-prev_Time)
    prev_Time = curr_Time

    cv2.putText(img, str(int(fps)), (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255), 1)

    cv2.imshow("cam", img)
    cv2.waitKey(1)



