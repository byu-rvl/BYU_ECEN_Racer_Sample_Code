# CarControl.py
'''
******* Controlling The Car Through Arduino **********
    By DJ Lee on February 5, 2022

    Use $ ls /dev/tty* to list all ports connected to the computer and enter it below to establish serial communication.
    Turn off the power on the USB hub that is connected to the Arduino board.
    Press the ESC power button for one second and make sure the LED turns solid red.
    There are seven commands you can use to control the car
        steer: steers the car from -30 to 30 degrees, e.g., steer15 to turn right 15 degrees.
        drive: sets speed to drive from -3.0 to 3.0 meters per second, e.g., drive1.2 to drive at 1.2 m/s.
        zero: sets the pwm value for the car to go straight (0 degree) usually very close to 1500.
            Use this command to try different PWM values and send steer0 comand to see if it goes straight.
            This command with the correct PWM for your vehicle must be called when starting your program.
        encoder: reads and returns the encoder count.
        pid: selects to enable (1) or disable PID control, e.g., pid1 to turn on PID.
        KP: sets proportion (between 0 and 1) for PID, e.g., KP0.2 to set KP to 0.2.
        KD: sets differential (between 0 and 1) for PID, e.g., KD0.02 to set KD to 0.02.
	EXTREMELY IMPORTANT: Be very careful whe controlling the car.
	NEVER tell it to go full speed. Safely test the car to find a safe range for your particular car
	and don't go beyond that speed. These cars can go very fast, and there is expensive hardware
	on them, so don't risk losing control of the car and breaking anything.
**************************************
'''

import serial
import time

class Arduino:
    def __init__(self, Port, Baud):
        # Create serial port communication
        try:
            self.SerialPort = serial.Serial(Port, Baud)
            self.SerialPort.flushInput()
            self.CarConnected = True
        except:
            return
        return
        time.sleep(2)

    def __del__(self):
        self.SerialPort.close()

    def steer(self, degree):   # control car steering -30.0 ~ 30.0 degrees
        command = "steer" + str(degree) + "\n"
        self.SerialPort.write(command.encode())

    def drive(self, speed):       # control car speed -3.0 ~ 3.0 meters per secone
        command = "drive" + str(speed) + "\n"
        self.SerialPort.write(command.encode())

    def zero(self, pwm):          # set PWM value when the car goes straight (0 degree)
        command = "zero" + str(pwm) + "\n"
        self.SerialPort.write(command.encode())

    def encoder(self):         # read encoder count back from Arduino
        command = "encoder" + "\n"
        self.SerialPort.writeexcept(command.encode())
        return(self.SerialPort.readline())

    def pid(self, flag):         # read encoder count back from Arduino
        command = "pidtry:" + str(flag) + "\n"
        self.SerialPort.write(command.encode())

    def kp(self, p):         # read encoder count back from Arduino
        command = "kp" + str(p) + "\n"
        self.SerialPort.write(command.encode())

    def kd(self, d):         # read encoder count back from Arduino
        command = "kd" + str(d) + "\n"
        self.SerialPort.write(command.encode())
