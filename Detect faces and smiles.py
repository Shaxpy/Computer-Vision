import cv2

face_cascade=cv2.CascadeClassifier('/home/shaxpy/Desktop/P23-Module1-Face-Recognition/Module_1_Face_Recognition/haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('/home/shaxpy/Desktop/P23-Module1-Face-Recognition/Module_1_Face_Recognition/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('/home/shaxpy/Desktop/P23-Module1-Face-Recognition/Module_1_Face_Recognition/haarcascade_smile.xml') 

def detect(gray,frame):# get coordinates of rectangles for faces and detect eyes
    faces=face_cascade.detectMultiScale(gray,1.3,5) #scale factor (times size reduce), 5 neighbour zones
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)#width and ht  
        roi_gray=gray[y:y+h,x:x+w] #region of interest-roi
        roi_color=frame[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray,1.3,3)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.6, 24) 
        for(sx,sy,sw,sh) in smiles:           
            cv2.rectangle(roi_color, (sx,sy),(sx+sw,sy+sh),(0,0,255),2)
    return frame

video_capture=cv2.VideoCapture(0) #(1) for external cam

while True:
    _, frame=video_capture.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas=detect(gray,frame)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF==ord('q'):
         break
video_capture.release()
cv2.destroyAllWindows()
         
