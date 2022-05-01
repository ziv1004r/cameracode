import numpy as np
import cv2
import cv

cap = cv2.VideoCapture(0)

while True:
    ret,img = cap.read()
    cv2.imshow('Original ziv Video',img)
    img2=cv2.flip(img,-1)
    cv2.imshow('Flipped ziv video',img2)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
