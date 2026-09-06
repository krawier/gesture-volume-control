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
area = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findHands(img)
    lmList, bound = detector.findPos(img, draw=True)
    
    if len(lmList) != 0:

        #print(bound)
        wB,hB = bound[2]-bound[0], bound[3]-bound[1]
        area = (wB*hB)//100
        #print(area)

        if 350<area<1000:
            lenght, img, lineInfo = detector.findDistance(4,8,img)
            


            #convert volume from lenght to actual volume -> reduce resolutin to make it smoother TODO

            volPer = np.interp(lenght, [30,250], [0,100])
            volume.SetMasterVolumeLevelScalar(volPer/100, None)
            #check fingers up? TODO

            #if pinky is down set volume





            if lenght < 25:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)

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