import cv2
import numpy as np


def findface(img):
    faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    ####create lists
    myFaceListC = []
    myFacelistArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 225), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0, 225, 0), cv2.FILLED)
        # the below lists is created to find the maximum area
        # and we only want to send that value back.
        myFaceListC.append([cx, cy])
        myFacelistArea.append(area)
    if len(myFacelistArea) != 0:
        i = myFacelistArea.index(max(myFacelistArea))  # find the index of max value in myfacelist area, stored in i
        return img, [myFaceListC[i], myFacelistArea[i]]  ##get the index of area and center of that area
    else:
        return img, [[0, 0], 0]  # [[cx,cy],area]


cap = cv2.VideoCapture(1)
while True:
    _, img = cap.read()
    #img = cv2.flip(img, 1)
    #print(img.shape)
    img, info = findface(img)
    print("Center", info[0], "Area", info[1])
    cv2.imshow("Output", img)
    cv2.waitKey(1)
