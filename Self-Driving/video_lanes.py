'''
Steps involved in this 


1. Convert to grayscale
2. Remove some noise from the lane image
--Smoothen -weighted average of pixels-Gaussian blur

3. Apply Canny method for edges(to outline strong grad)
Performs a derivative on our function in both  and y directions measuring change in
intensity w.r.t adjacent to pixels
If gradient is larger than high threshold it is accepted as an edge pixel

4.Region of interest
MAke a mask for white triangle for the region of interest

5.Finding Lane Lines
--Binary representation -- Bitwise_and operation

6-Finding Lanes using -- 
** Hough transform-- Set a threshold --min no of lines to 
accept a candidate line

--or by providing votes to max intersection of lines in bin
--or using polar coordinates (measure angle from origin 
to line)
--line_image=dlines(lane_image,lines)
combine=cv2.addWeighted()--- add the line image and the original canny filtered image
The function addWeighted calculates the weighted sum of two arrays

7. Optimization lanes--function make having paramters and averageim function

8. Find lanes in video


'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

def make(image,line_parameters):
    slope,intercept=line_parameters
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])


def averageim(image,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        paramters=np.polyfit((x1,x2),(y1,y2),1)     
        slope=paramters[0]
        intercept=paramters[1]
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_av=np.average(left_fit,axis=0)
    right_fit_av=np.average(right_fit,axis=0)
    left_line=make(image,left_fit_av)
    right_line=make(image,right_fit_av)
    return np.array([left_line,right_line])


def canny2(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    canny=cv2.Canny(blur,50,150)
    return canny
def dlines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),10)
    return line_image

def region(image):
    ht=image.shape[0]
    polygons=np.array([
    [(200,ht),(1100,ht),(550,250)]] )
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255) #WHITE TRIANGLE
    masked_image=cv2.bitwise_and(image,mask)
    return masked_image

cap=cv2.VideoCapture('Road-1101.mp4')
while(cap.isOpened()):
    _,frame=cap.read()
    canny=canny2(frame)
    crop=region(canny)
    lines=cv2.HoughLinesP(crop,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    average=averageim(frame,lines)

    line_image=dlines(frame,average)
    combine=cv2.addWeighted(frame,0.8,line_image,1,1)# gamma values- to add to the sum
  cv2.imshow('result',combine)
  cv2.waitKey(1)
# plt.imshow(canny)  #----for plotting graph 
# plt.show()
