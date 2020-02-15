

import numpy as np
import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#this is the cascade we just made. Call what you want
watch_cascade = cv2.CascadeClassifier('cascade.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # add this
    # image, reject levels level weights.

    watches = watch_cascade.detectMultiScale(gray, 120,200)
    
    # add this
    for (x,y,w,h) in watches:
       
        cv2.putText(img,'Watch',(x-w,y-h), font, 0.5, (11,255,255), 2, cv2.LINE_AA)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,200),2)

    for (x, y, w, h) in faces: # For each detected face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2) # We paint a rectangle around the face.
        roi_gray = gray[y:y+h, x:x+w] # We get the region of interest in the black and white image.
        roi_color = img[y:y+h, x:x+w] # We get the region of interest in the colored image.
        cv2.putText(img,'Face',(x+100,y-10),font,2.5,(255,0,0),4,cv2.LINE_AA)
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 24) # We apply the
        for (ex, ey, ew, eh) in eyes: # For each detected eye:
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2) # We paint a rectangle around the eyes, but inside the referential of the face.
            cv2.putText(img,'Eyes',(ex+200,ey+200),font,1.5,(0,200,0),2,cv2.LINE_8)
 

    cv2.imshow('video',img)
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break # We stop the loop.

cap.release()
cv2.destroyAllWindows()