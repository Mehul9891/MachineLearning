import numpy as np
import  cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([20,20,0])
    upper_red = np.array([255,255,255])
    mask = cv2.inRange(hsv, lower_red,upper_red)
    res = cv2.bitwise_and(frame,frame,mask=mask)
    cv2.imshow('hsv',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break