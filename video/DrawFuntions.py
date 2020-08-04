from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("G:\\Jupyter\\pic.JPG",cv2.IMREAD_COLOR)
img = cv2.resize(img,(1024,1024))
img = cv2.circle(img,(220,384),63,(0,255,0),-1)
img = cv2.putText(img,'circle',(220,460), cv2.FONT_ITALIC,4,(255,255,255),2,cv2.LINE_AA)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()