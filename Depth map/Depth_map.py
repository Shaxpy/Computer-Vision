#      ▄▀▄     ▄▀▄
#     ▄█░░▀▀▀▀▀░░█▄
# ▄▄  █░░░░░░░░░░░█  ▄▄
# █▄▄█ █░░▀░░┬░░▀░░█ █▄▄█

###################################
##### Authors:                #####
##### Stephane Vujasinovic    #####
##### Frederic Uhrweiller     #####
#####                         #####
##### Creation: 2017          #####
###################################


# ***********************
# **** Main Programm ****
# ***********************


# Package importation
import numpy as np
import cv2
from openpyxl import Workbook  # Used for writing data into an Excel file
from sklearn.preprocessing import normalize
from pyimagesearch.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import turtle
from PIL import Image
from turtle import Screen, Turtle


# Filtering
kernel = np.ones((3, 3), np.uint8)


def coords_mouse_disp(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # print x,y,disp[y,x],filteredImg[y,x]
        average = 0
        for u in range(-1, 2):
            for v in range(-1, 2):
                average += disp[y+u, x+v]
        average = average/9
        Distance = -593.97*average**(3) + 1506.8 * \
            average**(2) - 1373.1*average + 522.06
        Distance = np.around(Distance*0.01, decimals=2)
        print('Distance: ' + str(Distance)+' m')

# def coords_mouse_disp(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         # print x,y,disp[y,x],filteredImg[y,x]
#         average = 0
#         for u in range(-1, 2):
#             for v in range(-1, 2):
#                 average += disp[y+u, x+v]
#         average = average/9
#         Distance = -593.97*average**(3) + 1506.8 * \
#             average**(2) - 1373.1*average + 522.06
#         Distance = np.around(Distance*0.01, decimals=2)
#         print('Distance: ' + str(Distance)+' m')

# This section has to be uncommented if you want to take mesurements and store them in the excel
##        ws.append([counterdist, average])
##        print('Measure at '+str(counterdist)+' cm, the dispasrity is ' + str(average))
# if (counterdist <= 85):
##            counterdist += 3
# elif(counterdist <= 120):
##            counterdist += 5
# else:
##            counterdist += 10
##        print('Next distance to measure: '+str(counterdist)+'cm')


# Mouseclick callback
wb = Workbook()
ws = wb.active

# *************************************************
# ***** Parameters for Distortion Calibration *****
# *************************************************

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_EPS +
                   cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []   # 3d points in real world space
imgpointsR = []   # 2d points in image plane
imgpointsL = []

# Start calibration from the camera
print('Starting calibration for the 2 cameras... ')
# Call all saved images
for i in range(0, 14):   # Put the amount of pictures you have taken for the calibration inbetween range(0,?) wenn starting from the image number 0
    t = str(i)
    ChessImaR = cv2.imread('chessboard-R'+t+'.png', 0)    # Right side
    ChessImaL = cv2.imread('chessboard-L'+t+'.png', 0)    # Left side
    retR, cornersR = cv2.findChessboardCorners(ChessImaR,
                                               (9, 6), None)  # Define the number of chess corners we are looking for
    retL, cornersL = cv2.findChessboardCorners(ChessImaL,
                                               (9, 6), None)  # Left side
    if (True == retR) & (True == retL):
        objpoints.append(objp)
        cv2.cornerSubPix(ChessImaR, cornersR, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(ChessImaL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)

# Determine the new values for different parameters
#   Right Side
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                        imgpointsR,
                                                        ChessImaR.shape[::-1], None, None)
hR, wR = ChessImaR.shape[:2]
OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR,
                                            (wR, hR), 1, (wR, hR))

#   Left Side
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                        imgpointsL,
                                                        ChessImaL.shape[::-1], None, None)
hL, wL = ChessImaL.shape[:2]
OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

print('Cameras Ready to use')

# ********************************************
# ***** Calibrate the Cameras for Stereo *****
# ********************************************

# StereoCalibrate function
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
#flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
#flags |= cv2.CALIB_USE_INTRINSIC_GUESS
#flags |= cv2.CALIB_FIX_FOCAL_LENGTH
#flags |= cv2.CALIB_FIX_ASPECT_RATIO
#flags |= cv2.CALIB_ZERO_TANGENT_DIST
#flags |= cv2.CALIB_RATIONAL_MODEL
#flags |= cv2.CALIB_SAME_FOCAL_LENGTH
#flags |= cv2.CALIB_FIX_K3
#flags |= cv2.CALIB_FIX_K4
#flags |= cv2.CALIB_FIX_K5
retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                           imgpointsL,
                                                           imgpointsR,
                                                           mtxL,
                                                           distL,
                                                           mtxR,
                                                           distR,
                                                           ChessImaR.shape[::-1],
                                                           criteria_stereo,
                                                           flags)

# StereoRectify function
rectify_scale = 0  # if 0 image croped, if 1 image nor croped
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                  ChessImaR.shape[::-1], R, T,
                                                  rectify_scale, (0, 0))  # last paramater is alpha, if 0= croped, if 1= not croped
# initUndistortRectifyMap function
Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                              ChessImaR.shape[::-1], cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                               ChessImaR.shape[::-1], cv2.CV_16SC2)
# *******************************************
# ***** Parameters for the StereoVision *****
# *******************************************

# Create StereoSGBM and prepare all parameters
window_size = 5
min_disp = 2
num_disp = 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=window_size,
                               uniquenessRatio=10,
                               speckleWindowSize=100,
                               speckleRange=32,
                               disp12MaxDiff=5,
                               P1=8*3*window_size**2,
                               P2=32*3*window_size**2)

# Used for the filtered image
# Create another stereo for right this time
stereoR = cv2.ximgproc.createRightMatcher(stereo)

# WLS FILTER Parameters
lmbda = 55000
sigma = 2.24000
visual_multiplier = 1.8

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

# *************************************
# ***** Starting the StereoVision *****
# *************************************

# Call the two camera
CamR = cv2.VideoCapture(2)  # 2 # Wenn 0 then Right Cam and wenn 2 Left Cam
CamL = cv2.VideoCapture(0)  # 0

# Call other 2 cameras
# cap = cv2.VideoCapture(6)
# cap1 = cv2.VideoCapture(4)

# .......................................................................................

outputFrame = None
outputFrame_normal = None
# outputFramew = None
lock = threading.Lock()
lock_normal = threading.Lock()
# lockw = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup

#vs = VideoStream(src=0).start()
time.sleep(2.0)


@app.route("/", methods=['GET', 'POST'])
def index():
    # return the rendered template
    return render_template("index.html")


def detect_motion(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock
    md = SingleMotionDetector(accumWeight=0.4)
    # initialize the motion detector and the total number of frames
    # read thus far
    total = 0
    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        # frame = filt_Color

        frame = depth_map()
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (7,7), 0)
        gray = cv2.bilateralFilter(gray, 3, 126, 126)
        cv2.imwrite('ii.jpg', gray)
        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        # if the total number of frames has reached a sufficient
        # number to construct a reasonable background model, then
        # continue to process the frame
        if total > frameCount:
            # detect motion in the image
            motion = md.detect(gray)

            # cehck to see if motion was found in the frame
            if motion is not None:
                # unpack the tuple and draw the box surrounding the
                # "motion area" on the output frame
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY),
                              (0, 0, 255), 2)
        #cv2.setMouseCallback("Filtered Color Depth",coords_mouse_disp,frame)

        # update the background model and increment the total number
        # of frames read thus far
        md.update(gray)
        total += 1

        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = frame.copy()


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


def detect_motion_normal(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame_normal, lock_normal

    # initialize the motion detector and the total number of frames
    # read thus far
    md = SingleMotionDetector(accumWeight=0.4)
    total = 0

    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it

        # ........................ RBG TEST #2
        # frame = cv2.cvtColor(normal_frame,cv2.COLOR_GRAY2RGB)

        # frame = normal_frame
        frame = frameR
        frame = imutils.resize(frame, width=600)
        gray = frame
        image = frame
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (7, 7), 0)
        # gray=cv2.bilateralFilter(gray,4,5,24)

        # grab the current timestamp and r it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        # if the total number of frames has reached a sufficient
        # number to construct a reasonable background model, then
        # continue to process the frame
        if total > frameCount:
            # detect motion in the image
            motion = md.detect(gray)

            # cehck to see if motion was found in the frame
            if motion is not None:
                # unpack the tuple and draw the box surrounding the
                # "motion area" on the output frame
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY),
                              (0, 0, 255), 2)

        # update the background model and increment the total number
        # of frames read thus far
        md.update(gray)
        total += 1

        # acquire the lock, set the output frame, and release the
        # lock

        with lock_normal:
            outputFrame_normal = frame.copy()


def generate_normal():
    # grab global references to the output frame and lock variables
    global outputFrame_normal, lock_normal

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock_normal:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame_normal is None:
                continue

            # encode the frame in JPEG format
            (flag_normal, encodedImage_normal) = cv2.imencode(
                ".jpg", outputFrame_normal)

            # ensure the frame was successfully encoded
            if not flag_normal:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage_normal) + b'\r\n')


@app.route("/depth_map")
def depth_map():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_stream")
def video_stream():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate_normal(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# @app.route("/video")
# def video():
#     # return the response generated along with the specific media
#     # type (mime type)
#     return Response(generate_warp(),
#                     mimetype="multipart/x-mixed-replace; boundary=frame")


#############################################################################################

    # ..................NORMAL RGB FRAME --- TEST 5 --- TO TEST
    # normal_disp = stereo.compute(frameL,frameR) ### next try with frameL and frameR
    # normal_disp = np.concatenate((frameR, frameL), axis=1)
    # normal_frame = np.uint8(normal_disp)
    # normal_frame = normal_frame.astype(np.uint8)

def depth_map():
    retR, frameR = CamR.read()
    retL, frameL = CamL.read()
    # Rectify the image using the kalibration parameters founds during the initialisation
    Left_nice = cv2.remap(
        frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Right_nice = cv2.remap(
        frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    normal_disp = stereo.compute(frameL, frameR)
    normal_dispL = normal_disp
    normal_dispR = stereoR.compute(frameR, frameL)
    normal_dispL = np.int16(normal_dispL)
    normal_dispR = np.int16(normal_dispR)
    # Using the WLS filter
    normal_filteredImg = wls_filter.filter(
        normal_dispL, frameL, None, normal_dispR)
    normal_filteredImg = cv2.normalize(
        src=normal_filteredImg, dst=normal_filteredImg, beta=5, alpha=120, norm_type=cv2.NORM_MINMAX)
    normal_filteredImg = np.uint8(normal_filteredImg)
    normal_frame = cv2.applyColorMap(normal_filteredImg, cv2.COLORMAP_HSV)
    # normal_frame = cv2.cvtColor(normal_frame, cv2.COLOR_HSV2RGB)
    return normal_frame

    # ...................................

    # Show the result for the Depth_image
    #cv2.imshow('Disparity', disp)
    # cv2.imshow('Closing',closing)
    #cv2.imshow('Color Depth',disp_Color)
    #cv2.imshow('Filtered Color Depth',filt_Color)
# ................................................................................................................


while True:
    retR, frameR = CamR.read()

    # retL, frameL= CamL.read()
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=19,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

# start a thread that will perform motion detection
    t = threading.Thread(target=detect_motion, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()

    t2 = threading.Thread(target=detect_motion_normal, args=(
        args["frame_count"],))
    t2.daemon = True
    t2.start()

    # t3 = threading.Thread(target=detect_warp, args=(
    #     args["frame_count"],))
    # t3.daemon = True
    # t3.start()


# start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

    # Mouse click
    # cv2.setMouseCallback("Filtered Color Depth", coords_mouse_disp, filt_Color)

    # End the Programme
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# Save excel
# wb.save("data4.xlsx")

# Release the Cameras
CamR.release()
CamL.release()
# cap.release()
# cap1.release()
# cv2.waitKey(0)
cv2.destroyAllWindows()
