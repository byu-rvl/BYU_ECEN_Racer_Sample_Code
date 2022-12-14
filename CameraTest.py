# $ python3 CameraTest.py
'''
******* Realsense camera as the sensor ***************
The Intel Realsense 435i camera provides
    RGB Data
    Depth Data
	Gyroscope Data
	Accelerometer Data
This example only shows RGB and Depth data.
You can hit
    'r' to record video for development and training
    's' to stop recording
    'i' to save an image
    'q' to quit the program
***********************************************
'''
# import the necessary packages
from RealSense import *
import cv2
import imutils	

enableDepth = True
rs = RealSense("/dev/video2", RS_VGA, enableDepth)    # RS_VGA, RS_720P, or RS_1080P
writer = None
recording = False
frameIndex = 0
while True:
    (time, rgb, depth, accel, gyro) = rs.getData(enableDepth)

    if writer is None and recording is True:
        # initialize our video writer
        writer = cv2.VideoWriter('Video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15, (rgb.shape[1], rgb.shape[0]), True)

    cv2.imshow("RGB", rgb)
    cv2.imshow("Depth", depth)
    
    if recording == True:
        # write the output frame to disk
        writer.write(rgb)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):     # Start Recording
        recording = True
    if key == ord("s"):     # Stop Recording
        recording = False
    if key == ord("i"):     # Save image
        filename = "image" + str(frameIndex) + ".jpg"
        cv2.imwrite(filename, rgb)
        frameIndex += 1
    if key == ord("q"):
        break

if writer:
    writer.release()
del rs
cv2.destroyAllWindows()
