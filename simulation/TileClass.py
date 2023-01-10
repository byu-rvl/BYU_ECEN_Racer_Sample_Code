from __future__ import annotations
from .tileGenerator import sideLen, junctionTileImg, junctionOrientation, straightTileImg, straightOrientation, curveTileImg, curveOrientation, blankTileImg
import cv2
import numpy as np
from .polygonCalculator import coordinate

tileSizeCoord = coordinate(float(sideLen), float(sideLen))

startPoints = { # x, y, angle
    "straight": [ # n/s
        (sideLen/2, sideLen/3, 90.0),
        (sideLen/2, 2*sideLen/3, 270.0)
    ],
    "turn": [ #n n/e
        (sideLen - sideLen/2 * np.cos(np.deg2rad(30)), sideLen/2 * np.sin(np.deg2rad(30)), 60.0),
        (sideLen - sideLen/2 * np.cos(np.deg2rad(60)), sideLen/2 * np.sin(np.deg2rad(60)), 30.0)
    ],
    "t-junc": [ # n/e/s
        (sideLen/2, sideLen/4, 90.0),
        (3*sideLen/4, sideLen/2, 0),
        (sideLen/2, 3*sideLen/4, 270.0)
    ]
}

startPointOrientations = {
    "straight": "ns",
    "turn": "ne",
    "t-junc": "nes"
}

def minimizeAngle(a, degrees=True, toRound=6):
    offset = 180.0 if degrees else np.pi
    a += offset
    a = round(a, toRound)
    a %= round(2 * offset, toRound)
    a -= offset
    return round(a, toRound)

def absAngleDiff(a, b, degrees=True, toRound=6):
    realDiff = abs(a - b)
    return abs(minimizeAngle(realDiff, degrees, toRound))

def addSide(sideStr, count):
    sides = "nesw"
    out = ""
    for side in sideStr:
        index = (sides.index(side) + count) % len(sides)
        out += sides[index]
    return out

class Tile:
    def __init__(self, openings: str = ""):
        self._n = "n" in openings
        self._e = "e" in openings
        self._s = "s" in openings
        self._w = "w" in openings

    def openSides(self, sides: str = ""):
        self._n |= ("n" in sides)
        self._e |= ("e" in sides)
        self._s |= ("s" in sides)
        self._w |= ("w" in sides)
    
    def closeSides(self, sides: str = ""):
        self._n &= (not ("n" in sides))
        self._e &= (not ("e" in sides))
        self._s &= (not ("s" in sides))
        self._w &= (not ("w" in sides))
    
    def openCount(self):
        count = 0
        for letter in "nesw":
            if (self.isOpen(letter)):
                count += 1
        return count
    
    def isOpen(self, sides):
        total = True
        for side in sides:
            total &= self.__getattribute__("_" + side)
        return total
    
    def getOpenSides(self):
        out = ""
        for side in "nesw":
            if (self.isOpen(side)):
                out += side
        return out

    def compatible(self, side, tile: Tile, connectedFilter = False):
        openSide = self.isOpen(side)
        otherTileOpenSide = tile.isOpen(addSide(side, 2))
        if (connectedFilter):
            return (
                (openSide == otherTileOpenSide)
                and openSide
            )
        else:
            return openSide == otherTileOpenSide            
    
    def connected(self, side, tile: Tile):
        return self.compatible(side, tile, connectedFilter = True)
    
    def type(self):
        if self.openCount() == 0:
            return "empty"
        elif (self.openCount() == 3):
            return "t-junc"
        else:
            if (
                self.isOpen("ns")
                or self.isOpen("ew")
            ):
                return "straight"
            else:
                return "turn"
    
    def getStart(self, startNum, dir):
        # there are the same number of starting points as there are openings.
        type = self.type()
        numTurns = 0
        if (type == "empty"):
            raise Exception("Cannot get starting point for blank tile")

        # figure out how many times we need to turn the reference to get to the actual orientation
        openSides = self.getOpenSides()
        needSides = startPointOrientations[type]
        allIn = False
        while (not allIn):
            soFar = True
            for side in needSides:
                if (side not in openSides):
                    needSides = addSide(needSides, 1)
                    numTurns += 1
                    soFar = False
                    break
            allIn = soFar
        
        startPoint = startPoints[type][startNum]
        startCoord = coordinate(float(startPoint[0]), float(startPoint[1]))
        startCoord.rotateAboutPoint(90 * numTurns, coordinate(float(sideLen/2), float(sideLen/2)), True)
        startAngle = startPoint[2] + 90 * numTurns + 180 * dir
        return startCoord, startAngle

    @staticmethod
    def statHelper(tileType: str = "straight", numTurns: int = 0, carPos: coordinate = coordinate(), carBearing: float = 0.0):
        translateCompare = 0 # what the car has
        translateCompareTo = sideLen/2
        angleCompare = (0, 0) # options for what it should be

        if (tileType == "straight"):
            # If it's straight, we just find how close to the middle of the road it is
            translateCompare = carPos.y if (numTurns % 2) else carPos.x # check horizontal or vertical coord of car depending on road orientation
            angleCompare = (np.deg2rad(-90. + (90 * numTurns)), np.deg2rad(90. + (90 * numTurns))) # should be straight relative to road
        elif (tileType == "turn"):
            # if it's a turn, we start with a point in the to-right, since that represents the canonical configuration
            startCoord = coordinate(sideLen, 0)
            # we rotate that point about the center of the tile by (numTurns) times, now we have our actual start
            # for the tile the car is on.
            startCoord.rotateAboutPoint(90 * numTurns, coordinate(float(sideLen/2), float(sideLen/2)), True)
            # distance from this point to the car should be sideLen/2 if it's right in the center
            carVector = carPos - startCoord
            translateCompare = carVector.length()

            # Now we also need to see what angle the car is at relative to that point
            carAngle = carVector.angle()
            offset = np.deg2rad(90)
            angleCompare = (carAngle + offset, carAngle - offset) # should be at + or - 90 degrees to the angle
        elif (tileType == "t-junc"):
            # first, we'll get values if it was a straight road, and if it was two curved roadds.
            options = [Tile.statHelper(typeOption, turnCount, carPos, carBearing) for typeOption, turnCount in zip(["straight", "turn", "turn"], [numTurns, numTurns, numTurns + 1])]

            # Now we loop through these to see which one is best in distance, and we'll use that angle as well.
            bestDist = np.inf
            bestAngle = 0

            for option in options:
                dist, angle = option
                if (dist < bestDist):
                    bestDist = dist
                    bestAngle = angle
            
            return bestDist, bestAngle
        else:
            raise Exception(f"Cannot compute, unknown tile type '{tileType}'.")
        
        positionDiff = abs(translateCompare - translateCompareTo)
        angleDiff = np.inf
        for a in angleCompare:
            aDiff = absAngleDiff(a, carBearing, degrees=False)
            if (aDiff < angleDiff):
                angleDiff = aDiff
        return positionDiff, angleDiff


    def getStats(self, carPos: coordinate, carBearing: float):
        type = self.type()
        if (type == "empty"):
            # if the car is on a blank tile, it's bad:
            return np.inf, np.inf # distance to center line and bearing offset are both set to inf because we don't have anything to compare to
        elif (type == "straight"):
            numTurns = 0 if "n" in self.getOpenSides() else 1
            return self.statHelper(type, numTurns, carPos, carBearing)
        else:
            # figure out how many times we need to turn the reference to get to the actual orientation
            numTurns = 0
            openSides = self.getOpenSides()
            needSides = startPointOrientations[type]
            allIn = False
            while (not allIn):
                soFar = True
                for side in needSides:
                    if (side not in openSides):
                        needSides = addSide(needSides, 1)
                        numTurns += 1
                        soFar = False
                        break
                allIn = soFar
            
            return self.statHelper(type, numTurns, carPos, carBearing)
    
    def getImage(self):
        type = self.type()
        if (type == "empty"):
            img = blankTileImg.copy()
            return img
        elif (type == "straight"):
            img = straightTileImg.copy()
            currentOrientation = straightOrientation
        elif (type == "turn"):
            img = curveTileImg.copy()
            currentOrientation = curveOrientation
        elif (type == "t-junc"):
            img = junctionTileImg.copy()
            currentOrientation = junctionOrientation

        openSides = self.getOpenSides()
        foundMatch = False
        while (not foundMatch):
            currentOrientation = addSide(currentOrientation, 1)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            allMatch = True
            for s in openSides:
                allMatch = allMatch and (s in currentOrientation)
            foundMatch = foundMatch or allMatch
        return img
        
        # out = np.full((3, 3), 1)
        # if (self.openCount() > 0):
        #     out[1, 1] = 0

        # if (self._n):
        #     out[0, 1] = 0
        # if (self._e):
        #     out[1, 2] = 0
        # if (self._s):
        #     out[2, 1] = 0
        # if (self._w):
        #     out[1, 0] = 0

        # return out