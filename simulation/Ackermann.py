from .polygonCalculator import coordinate
from typing import Optional, Union
import time
import numpy as np

def monotonic_ms(currentTime: Optional[float] = None):
    if (currentTime is None):
        currentTime = time.monotonic()
    return int(round(currentTime * 1e3))

class AckermannState:
    def __init__(self, x, y, theta, delta, v, steeringOffset=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.delta = delta
        self.v = v
        self.steeringOffset = steeringOffset
    
    def update(self, L):
        smallV = self.v/1000
        self.x += smallV * np.cos(self.theta)
        self.y += smallV * np.sin(self.theta)
        self.theta += smallV/L * np.tan(self.delta + self.steeringOffset)
    
    def setSteering(self, delta):
        self.delta = delta

    def setVelocity(self, v):
        self.v = v

class Ackermann:
    def __init__(self,
        wheelbase: float = 6.0,
        maxSteering: float = 30.0,
        steeringOffset: float = 0.0,
        startSteering: float = 0.0,
        minVelocity: float = 0.0,
        maxVelocity: float = 60.0,
        startVelocity: float = 0.0,
        startBearing: float = 0.0,
        startPoint: Optional[coordinate] = None,
        modelTime: Optional[int] = None
    ):
        """
        Wheelbase is the distance from the front to the back axle in inches.
        maxSteering is the highest angle at which the vehicle can turn (+/-) in degrees
        startSteering is - in degrees - the intial steering angle of the vehicle
        maxVelocity is the top speed of the car in pix/s
        startVelocity is - in pix/s - the starting velocity of the vehicle
        startBearing is - in degrees - the initial world-frame bearing of the vehicle
        startPoint is where the car starts, positionally.
        modelTime is the current/starting time of the model in ms
        """
        if (startPoint is None):
            startPoint = coordinate()
        if (modelTime is None):
            modelTime = monotonic_ms()
        self.wheelbase = wheelbase * 8 # convert from in to pixels
        self.maxSteering = maxSteering
        self.minVelocity = minVelocity
        self.maxVelocity = maxVelocity
        self.modelTime = modelTime
        self.currentState = AckermannState(
            startPoint.x, # State x
            startPoint.y, # State y
            np.deg2rad(startBearing), # State theta (world angle)
            np.deg2rad(startSteering), # State delta (steering angle, relative)
            startVelocity, # State velocity
            np.deg2rad(steeringOffset) # usually the vehicle doesn't go perfectly straight
        ) # x, y, theta (world angle), delta (steering angle),
    
    def update(self, currentTime: Union[int, float, None] = None):
        if (currentTime is None):
            currentTime = monotonic_ms()
        if (type(currentTime) is float):
            currentTime = monotonic_ms(currentTime)
        timeDiff = int(currentTime - self.modelTime)
        self.modelTime = currentTime

        # handle mins/maxes:
        v = self.currentState.v
        d = self.currentState.delta

        # If we exceed max velocity, cap
        if (abs(v) > self.maxVelocity):
            self.currentState.v = v/abs(v) * self.maxVelocity
        
        # If we're beneath min velocity, we don't move
        if (abs(v) < self.minVelocity):
            self.currentState.v = 0
        
        # If we're past max steering, we cap
        if (abs(d) > self.maxSteering):
            self.currentState.delta = d/abs(d) * self.maxSteering

        for timestep in range(timeDiff):
            # For every millisecond we update the motion model
            self.currentState.update(self.wheelbase)

    def setSteering(self, delta): # delta in degrees, we'll threshhold and convert
        toUse = max(min(delta, self.maxSteering), -self.maxSteering)
        self.currentState.setSteering(np.deg2rad(toUse))
    
    def setVelocity(self, v):
        # we receive a float from -3 to +3, we need to map it to the max speed
        # first, be sure it's actually in that range
        realV = min(3.0, max(-3.0, v))

        # Then map it over the range
        toUse = self.maxVelocity * realV/3.0
        self.currentState.setVelocity(toUse)
    
    def getCoord(self):
        return coordinate(float(self.currentState.x), float(self.currentState.y))
    
    def getFacing(self):
        """ Returns the direction the car is currently facing, in radians """
        return self.currentState.theta
    
    def pose(self):
        return self.getCoord(), self.getFacing()

# import cv2
# img = np.zeros((1000, 1000, 3))
# cv2.namedWindow("output", cv2.WINDOW_NORMAL)

# car = Ackermann(startVelocity = 100, modelTime=0, startSteering = 5)

# def drawCar():
#     global car, img
#     newImg = img.copy()
#     carPlace = car.getCoord()
#     newImg = cv2.line(newImg, carPlace.asInt(), (carPlace + 100*coordinate.unitFromAngle(car.currentState.theta, isRadians = True)).asInt(), (0, 255, 0), 3)
#     newImg = cv2.line(newImg, carPlace.asInt(), (carPlace + 100*coordinate.unitFromAngle(car.currentState.theta + car.currentState.delta, isRadians = True)).asInt(), (0, 0, 255), 3)
#     cv2.imshow("output", newImg)

# for t in range(5000): # 5 seconds?
#     car.update(t)
#     drawCar()
#     if (cv2.waitKey(1) == ord('q')):
#         break