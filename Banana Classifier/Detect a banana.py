import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('/home/shaxpy/Desktop/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/shaxpy/Desktop/haarcascade_eye.xml')
banana_cascade=cv2.CascadeClassifier('/home/shaxpy/Desktop/cascade.xml')

font=cv2.FONT_HERSHEY_PLAIN
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 14)
    
    # add this
    # image, reject levels level weights.
    banana = banana_cascade.detectMultiScale(gray,30,30)
    
    # add this
    for (x,y,w,h) in banana:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        cv2.putText(img,'Banana',(x,y-10),font,0.5,(0,0,255),2,cv2.LINE_8)
    for (x,y,w,h) in faces:
        
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(img,'Face',(x+100,y-10),font,2.5,(255,0,0),4,cv2.LINE_AA)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 12)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.putText(img,'Eyes',(ex+200,ey+200),font,1.5,(0,200,0),2,cv2.LINE_8)
    
    
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break

cap.release()
cv2.destroyAllWindows()
