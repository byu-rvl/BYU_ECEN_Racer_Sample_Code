# $ python3 CarTest.py

'''
******* Controlling The Car **********
    By DJ Lee on February 5, 2022

    Use $ ls /dev/tty* to list all ports connected to the computer and enter it below to establish serial communication.
    Turn off the power on the USB hub that is connected to the Arduino board.
    Press the ESC power button for one second and make sure the LED turns solid red.
    There are seven commands you can use to communicate with the car
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

from Arduino import Arduino

# Use $ ls /dev/tty* to find the serial port connected to Arduino
Car = Arduino("/dev/ttyUSB0", 115200)                # Linux
#Car = Arduino("/dev/tty.usbserial-2140", 115200)     # Mac

while True:
    command = input("Enter a command:\n")
    if command == 's':
        angle = input("Enter a steering angle (-30 ~ 30):\n")
        Car.steer(float(angle))
    elif command == 'd':
        speed = input("Enter a drive speed (-3.0 ~ 3.0):\n")
        Car.drive(float(speed))
    elif command == 'z':
        pwm = input("Enter a PWM value (~1500):\n")
        Car.zero(int(pwm))
    elif command == 'p':
        flag = input("Enter 1 to turn on PID and 0 to turn off:\n")
        Car.pid(int(flag))
    elif command == 'e':
        print(int(Car.encoder().strip()))   # need to strip character of \r or \n
    elif command == 'q':
        if Car.CarConnected:
            del Car
        break


