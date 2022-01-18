import cv2
from HandTrackingModule import handDetector
import math
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = handDetector(detectionCon=0.8, maxHands=1)
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)

def putTextRect(img, text, pos, scale=3, thickness=3, colorT=(255, 255, 255),
                colorR=(255, 0, 255),font=cv2.FONT_HERSHEY_PLAIN,
                offset=10, border=None, colorB=(0, 255, 0)):
    ox, oy = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset
    cv2.rectangle(img, (x1, y1), (x2, y2), colorR, cv2.FILLED)
    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), colorB, border)
    cv2.putText(img, text, (ox, oy), font, scale, colorT, thickness)
    return img, [x1, y2, x2, y1]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands = detector.findHands(img, draw=False)

    if hands:
        lmList = hands[0]["lmList"]
        x, y, w, h = hands[0]["bbox"]
        x1, y1 = lmList[5]
        x2, y2 = lmList[17]

        distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
        A, B, C = coff
        distanceCM = A * distance**2 + B * distance + C
        # print(distanceCM, distance)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
        putTextRect(img, f'{int(distanceCM)} cm', (x + 5, y - 10))
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break