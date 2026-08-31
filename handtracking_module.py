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
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, 
                                        max_num_hands=self.maxHands, 
                                        min_detection_confidence=self.detectionConf, 
                                        min_tracking_confidence=self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # we want to convert to rgb bc hands only uses rgb imgs
        self.result = self.hands.process(imgRGB)
        
        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img        

                
    def findPos(self, img, handNo = 0, draw =True):

        bound = ()
        
        xList = []
        yList = []


        lmList = []
        if self.result.multi_hand_landmarks:  
            myHand = self.result.multi_hand_landmarks[handNo]               
            for id, lm in enumerate(myHand.landmark):    
                             height, width, channels = img.shape
                             cx, cy = int(lm.x *width), int(lm.y * height)
                             xList.append(cx)
                             yList.append(cy)
                             lmList.append([id, cx, cy])
                             if draw:    
                              cv2.circle(img, (cx,cy), 5 , (255,0,255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bound = xmin, ymin, xmax, ymax

        return lmList, bound
        



def main():
    prev_Time = 0
    curr_Time = 0

    cap = cv2.VideoCapture(0)

    detector = handDetector()


    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        landList = detector.findPos(img)
        if len(landList) != 0: 
            print(landList[4])
    
        curr_Time = time.time()
        fps = 1/(curr_Time-prev_Time)
        prev_Time = curr_Time

        cv2.putText(img, str(int(fps)), (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255), 1)

        cv2.imshow("cam", img)
        cv2.waitKey(1)


if __name__== "__main__":
    main()