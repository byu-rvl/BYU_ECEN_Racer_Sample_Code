import cv2

from .Ackermann import Ackermann
from .MapGenerator import Map
from .camera import Camera
from .polygonCalculator import coordinate
import numpy as np

class RealSenseSimulator:
    # init should create the map
    def __init__(self, parent, cameraSettings = None): # TODO should pass params like camera height, x/y/z offsets, etc.
        self.parent = parent
        if (cameraSettings is None):
            cameraSettings = {}
        self.cameraParams = cameraSettings

    def makeMap(self, seed=None, startData=None, mapParameters=None): # need to pass map generation params
        if (mapParameters is None):
            mapParameters = {}
        self.currentMap = Map(seed, **mapParameters)
        self.currentImg = self.currentMap.getImage()
        self.camera = Camera(self.currentImg, **self.cameraParams)
        return self.currentMap.getStart(startData)

    def getFrame(self):
        if (self.parent.ackermann is None):
            raise Exception("Cannot get frame, you must call the start() method on the simulator")
        self.parent.advanceTime(1/30)
        where, facing = self.parent.ackermann.pose()
        return self.camera.getImage(where, facing)

class ArduinoSimulator:
    def __init__(self, parent):
        self.parent = parent
        pass

    def setSpeed(self, speed):
        # Passes speed (-3.0 to +3.0) into ackermann
        self.parent.setSpeed(speed)
    
    def setSteering(self, steering):
        # handle random offset here?  Probably.
        self.parent.setSteering(steering)
    
class Simulator:
    def __init__(self, cameraSettings = None):
        self.Arduino = ArduinoSimulator(self)
        self.RealSense = RealSenseSimulator(self, cameraSettings)
        self.ackermann = None
        self.currentSteering = 0
        self.currentSpeed = 0
        self.windowsMade = False
    
    def start(
        self,
        mapSeed = None,
        startPoint = None,
        mapParameters = None,
        carParameters = None
    ):
        self.time = 0
        if (mapParameters is None):
            mapParameters = {}
        if (carParameters is None):
            carParameters = {}
        startCoords, startAngle = self.RealSense.makeMap(mapSeed, startPoint, mapParameters)
        self.ackermann = Ackermann(
            modelTime = float(self.time),
            startPoint = startCoords,
            startBearing = startAngle,
            startSteering = self.currentSteering,
            startVelocity = self.currentSpeed,
            **carParameters
        )
        return self.RealSense.getFrame()
    
    def setSpeed(self, speed):
        self.currentSpeed = speed
        if (self.ackermann is not None):
            self.ackermann.setVelocity(self.currentSpeed)
    
    def setSteering(self, steering):
        self.currentSteering = steering
        if (self.ackermann is not None):
            self.ackermann.setSteering(self.currentSteering)
    
    # Every time "getFrame" is called we advance by 1/30 second
    def advanceTime(self, timeStep):
        self.time += timeStep
        self.ackermann.update(float(self.time))
    
    def getStats(self):
        # extract data about the car to be used in reward schemes.
        carPosition = self.ackermann.getCoord()
        carBearing = self.ackermann.getFacing()
        distanceToCenter, bearingOffset = self.RealSense.currentMap.getStatistics(carPosition, carBearing)
        return distanceToCenter, bearingOffset

    def step(self,steer,speed,display=False):
        #for network training to step the simulation with steering and speed as input
        self.Arduino.setSpeed(speed)
        self.Arduino.setSteering(steer)
        frame = self.RealSense.getFrame()
        if display:
            if not self.windowsMade:
                cv2.namedWindow("map", cv2.WINDOW_NORMAL)
                cv2.namedWindow("car", cv2.WINDOW_NORMAL)
            cv2.imshow("car",frame)
            carPlace = self.ackermann.getCoord()
            map = self.RealSense.currentImg.copy()
            map = cv2.line(map, carPlace.asInt(), (
                        carPlace + 100 * coordinate.unitFromAngle(self.ackermann.currentState.theta, isRadians=True)).asInt(),
                              (0, 255, 0), 3)
            map = cv2.line(map, carPlace.asInt(), (carPlace + 100 * coordinate.unitFromAngle(
                self.ackermann.currentState.theta + self.ackermann.currentState.delta + self.ackermann.currentState.steeringOffset,
                isRadians=True)).asInt(), (0, 0, 255), 3)

            cv2.imshow("map",map)
            cv2.waitKey(1)
        return frame



