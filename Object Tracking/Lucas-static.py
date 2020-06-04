import numpy as np
import cv2

corner=dict(maxCorners=2,qualityLevel=0.15,minDistance=7,blockSize=3)
lk=dict(winSize=(300,400),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS| cv2.TERM_CRITERIA_COUNT,4,0.03))


cap=cv2.VideoCapture(0)
ret,prev=cap.read()
prevg=cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)

#Points for tracking

prevpts=cv2.goodFeaturesToTrack(prevg,mask=None,**corner)

mask=np.zeros_like(prev)

while True:
    ret,frame=cap.read()
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    nextpts,status,err=cv2.calcOpticalFlowPyrLK(prevg,frame_gray,prevpts,None,**lk)
    good_new=nextpts[status==1]
    good_prev=prevpts[status==1]
    for i ,(new,prev) in enumerate(zip(good_new,good_prev)):
        x_new,y_new=new.ravel()
        x_prev,y_prev=prev.ravel()
        mask = cv2.line(mask, (x_new,y_new),(x_prev,y_prev), (0,5,0), 1)
        # Draw red circles at corner points
        frame = cv2.circle(frame,(x_new,y_new),8,(0,0,255),-1)
    # Display the image along with the mask we drew the line on.
    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break # We stop the loop.
   
    # Now update the previous frame and previous points
    prev_gray = frame_gray.copy()
    prevPts = good_new.reshape(-1,1,2)
    
    
cv2.destroyAllWindows()
cap.release()
