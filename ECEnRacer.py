# python3 ECEnRacer.py
''' 
This program is for ECEN-631 BYU Race
*************** RealSense Package ***************
From the Realsense camera:
	RGB Data
	Depth Data
	Gyroscope Data
	Accelerometer Data
*************** Arduino Package ****************
	Steer(int degree) : -30 (left) to +30 (right) degrees
	Drive(float speed) : -3.0 to 3.0 meters/second
	Zero(int PWM) : Sets front wheels going straight around 1500
	Encoder() : Returns current encoder count.  Reset to zero when stop
	Pid(int flag) : 0 to disable PID control, 1 to enable PID control
	KP(float p) : Proporation control 0 ~ 1.0 : how fast to reach the desired speed.
	KD(float d) : How smoothly to reach the desired speed.

    EXTREMELY IMPORTANT: Read the user manual carefully before operate the car
**************************************
'''

# import the necessary packages
from Arduino import Arduino
from RealSense import *
import numpy as np
import imutils
import cv2

enableDepth = True
rs = RealSense("/dev/video2", RS_VGA, enableDepth)		# RS_VGA, RS_720P, or RS_1080P
writer = None

# Use $ ls /dev/tty* to find the serial port connected to Arduino
Car = Arduino("/dev/ttyUSB0", 115200)                # Linux
#Car = Arduino("/dev/tty.usbserial-2140", 115200)    # Mac

Car.zero(1500)      # Set car to go straight.  Change this for your car.
Car.pid(1)          # Use PID control
# You can use kd and kp commands to change KP and KD values.  Default values are good.
# loop over frames from Realsense
while True:
    (time, rgb, depth, accel, gyro) = rs.getData(enableDepth)

    cv2.imshow("RGB", rgb)
    cv2.imshow("Depth", depth)

    '''
    Add your code to process rgb, depth, IMU data
    '''

    '''
    Control the Car
    '''

    '''
   	IMPORTANT: Never go full speed. Use CarTest.py to selest the best speed for you.
    Car can switch between positve and negative speed (go reverse) without any problem.
    '''
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
del rs
del Car

