import cv2
import time
import numpy as np
import handtracking_module as htm


wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0


while True:
    success, img = cap.read()

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