import cv2
import numpy as np


img1 = cv2.resize(cv2.imread("G:\\Jupyter\\index1.jpg"), (512,512))
img2 = cv2.resize(cv2.imread("G:\\Jupyter\\index2.jpeg"), (512,512))
img3 = cv2.imread("G:\\Jupyter\\pic.JPG")
#img3 = img1 + img2
img4 = cv2.add(img1,img2)

cv2.imshow('img4', img4)
cv2.waitKey(0)
cv2.destroyAllWindows()

ret, threshold = cv2.threshold(img3,120,255,cv2.THRESH_BINARY)
cv2.imshow('original',img3)
cv2.imshow('img3',threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()