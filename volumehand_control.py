import cv2
import time
import numpy as np
import handtracking_module as htm
import math

from pycaw.pycaw import AudioUtilities

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionConf=0.8)

device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findHands(img)
    lmList, bound = detector.findPos(img, draw=False)
    
    if len(lmList) != 0:

        # filter on size? TODO

        print(bound)

        # find distance -> methodize it TODO

        #convert volume from lenght to actual volume -> reduce resolutin to make it smoother TODO

        #check fingers up? TODO

        #if pinky is down set volume

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2] 

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 10, (225, 0, 5), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (225, 0, 5), cv2.FILLED)
        cv2.circle(img, (cx, cy), 10, (225, 0, 5), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 5), 3)

        lenght = math.hypot(x2 - x1, y2 - y1)

        vol = np.interp(lenght, [25, 200], [minVol, maxVol])
        print(vol)
        volume.SetMasterVolumeLevel(float(vol), None)

        if lenght < 25:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

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