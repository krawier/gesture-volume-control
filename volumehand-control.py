import cv2
import time
import numpy as np
import handtracking_module as htm


wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector()



while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPos(img, draw = False)
    if len(lmList) != 0:
        print(lmList[4], lmList[8])

        x1, y1 =  lmList[4][1], lmList[4][2]
        x2, y2 =  lmList[8][1], lmList[8][2] 

        cv2.circle(img, (x1,y1), 15 , (225,0,5), cv2.FILLED )
        cv2.circle(img, (x2,y2), 15,  (225,0,5), cv2.FILLED )

    cTime = time.time()
    
    if cTime - pTime > 0:
        fps = 1 / (cTime - pTime)
    else:
        fps = 0
        
    pTime = cTime

    cv2.putText(img, f'fps: {int(fps)}', (40, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
    
    cv2.imshow("img", img)

    cv2.waitKey(1)
    
    if cv2.getWindowProperty("img", cv2.WND_PROP_VISIBLE) < 1:
        break
    

cap.release()
cv2.destroyAllWindows()