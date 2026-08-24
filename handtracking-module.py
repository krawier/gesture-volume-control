import cv2
import mediapipe as mp
import time

class handDetector():

    def __init__(self, mode = False, maxHands = 2, detectionConf = 0.5, trackConf = 0.5 ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 
                                        self.detectionConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # we want to convert to rgb bc hands only uses rgb imgs
        result = self.hands.process(imgRGB)
        
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

                # for id, lm in enumerate(handLms.landmark):
                        
                #     height, width, channels = img.shape
                #     cx, cy = int(lm.x *width), int(lm.y * height)
                #     print(id,cx, cy)
        
                #     if id == 0:
                #         cv2.circle(img, (cx,cy), 25 , (255,0,255), cv2.FILLED)
        


def main():
    prev_Time = 0
    curr_Time = 0

    cap = cv2.VideoCapture(0)


    while True:
        success, img = cap.read()
    
    curr_Time = time.time()
    fps = 1/(curr_Time-prev_Time)
    prev_Time = curr_Time

    cv2.putText(img, str(int(fps)), (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255), 1)

    cv2.imshow("cam", img)
    cv2.waitKey(1)


if __name__== "__main__":
    main()