from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class coordinate:
    x: float
    y: float

    @staticmethod
    def unitFromAngle(degrees: float | int = 0.0, isRadians: bool = False) -> coordinate:
        if (not isRadians):
            theta = np.radians(float(degrees))
        else:
            theta = float(degrees)
        c, s = np.cos(theta), np.sin(theta)
        return coordinate(float(c), float(s))

    def __init__(self, x: Optional[float | int | coordinate | tuple] = None, y: Optional[float | int] = 0.0) -> coordinate:
        if (
            type(x) is float
            or type(x) is int
        ):
            self.x = float(x)
            self.y = float(y)
        elif (type(x) is coordinate):
            self.x = float(x.x)
            self.y = float(x.y)
        elif (type(x) is tuple):
            self.x = float(x[0])
            self.y = float(x[1])
        elif (x is None):
            self.x = 0.0
            self.y = float(y)
        else:
            raise Exception(f"Could not make coorindate from passed arguments ({type(x)} and {type(y)})")
    
    def rotate(self, degrees: float = 0.0, mutate: bool = False) -> coordinate:
        theta = np.radians(degrees)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        homogenous = np.array([[self.x, self.y, 1]]).T
        rotated = np.squeeze((R@homogenous))
        newCoord = coordinate(float(rotated[0]), float(rotated[1]))
        if (mutate):
            self.x = newCoord.x
            self.y = newCoord.y
        return newCoord
    
    def rotateAboutPoint(self, degrees: float = 0.0, point: Optional[coordinate] = None, mutate: bool = False) -> coordinate:
        if (point is None):
            point = coordinate()

        newCoord = (self - point).rotate(degrees) + point
        if (mutate):
            self.x = newCoord.x
            self.y = newCoord.y
        
        return newCoord
    
    def __mul__(self, other: float | int | coordinate):
        if type(other) is coordinate:
            return coordinate(self.x * other.x, self.y * other.y)
        else:
            return coordinate(self.x * other, self.y * other)

    def __rmul__(self, other) -> coordinate:
        return self.__mul__(other)
    
    def __pow__(self, other) -> coordinate:
        return coordinate(self.x ** other, self.y ** other)
    
    def __add__(self, other: float | int | coordinate) -> coordinate:
        if type(other) is coordinate:
            return coordinate(self.x + other.x, self.y + other.y)
        else:
            return coordinate(self.x + other, self.y + other)
    
    def __radd__(self, other) -> coordinate:
        return self.__add__(other)
    
    def __invert__(self) -> coordinate:
        return coordinate(-self.x, -self.y)
    
    def __neg__(self) -> coordinate:
        return self.__invert__()

    def __sub__(self, other) -> coordinate:
        return self.__add__(-other)
    
    def __rsub__(self, other) -> coordinate:
        return (-self).__add__(other)
    
    def __truediv__(self, other: float | int | coordinate) -> coordinate:
        if type(other) is coordinate:
            return coordinate(self.x/other.x, self.y/other.y)
        else:
            return coordinate(self.x/other, self.y/other)
    
    def __floordiv__(self, other: float | int | coordinate) -> coordinate:
        if type(other) is coordinate:
            return coordinate(self.x//other.x, self.y//other.y)
        else:
            return coordinate(self.x//other, self.y//other)

    def isAboveLine(self, linePoint1: coordinate, linePoint2: coordinate) -> bool:
        b, a, c = linePoint1, linePoint2, self
        # would be a, b, c but inverted y in images means we should do it this way

        bma = b - a
        cma = c - a

        return ((bma.x * cma.y) - (bma.y * cma.x)) > 0
    
    def sum(self) -> float:
        return float(self.x + self.y)
    
    def length(self) -> float:
        return np.sqrt((self ** 2).sum())

    def angle(self, pFrom: Optional[coordinate] = None, pTo: Optional[coordinate] = None):
        if (pFrom is not None and pTo is not None):
            # print("3 way angle")
            # we have both, 3-way angle
            a1 = (pFrom - self)._angle()
            # print(f"angle from pFrom {pFrom} to self {self} is {a1}")
            a2 = (pTo - self)._angle()
            # print(f"angle from self {self} to pTo {pTo} is {a2}")

            first = a1 if a1 > a2 else a2
            second = a2 if a1 > a2 else a1
            # print(f"calling first {first} and second {second}")

            diff = (first - second)
            if (diff > round(np.pi, 6) or diff < 0):
                diff %= round(np.pi, 6) # angle can't be larger than pi, but it might BE pi...
            
            # print("calculated diff: ", diff)
            
            return diff
        elif (pFrom is None and pTo is None):
            # we have no givne points, assume it's from the origin to self
            pFrom = coordinate()
        
        if (pFrom is not None or pTo is not None):
            # we have one of them, can find a partial angle
            p1, p2 = None, None
            if (pFrom is not None):
                p1, p2 = pFrom, self
            else:
                p1, p2 = self, pTo

            diff = p2 - p1
            return diff._angle()
    
    def _angle(self):
        return round(np.arctan2(self.y, self.x), 6) % round(2*np.pi, 6)
    
    def asInt(self) -> tuple(int, int):
        return (round(self.x), round(self.y))

def calculatePolygon(h, w, f, xAngle, yAngleInv):
    # xAngle = -xAngleInv
    yAngle = -yAngleInv

    # pixelsPerDegree = h/fov_y
    yOffsetPixels = f * np.tan(np.deg2rad(-(xAngle - (-90.0))))

    # yOffsetPixels = -(xAngle - (-90.0)) * pixelsPerDegree
    horizonCenterOffset = coordinate(0, yOffsetPixels)
    imageCenter = coordinate(w/2, h/2)

    # print("calculating polygon with the following", xAngle, yAngle, fov_y, yOffsetPixels, h, w, horizonCenterOffset, imageCenter)

    horizonCenter = coordinate(horizonCenterOffset + imageCenter)
    horizonCenter.rotateAboutPoint(yAngle, imageCenter, True)
    leftDir = coordinate.unitFromAngle(yAngle + 180)
    rightDir = -leftDir

    leftXThresh = -1.0 if leftDir.x < 0 else w + 1.0
    rightXThresh = w + 1.0 if leftXThresh == -1.0 else -1.0

    toGoLeft = (leftXThresh - horizonCenter.x) / leftDir.x
    toGoRight = (rightXThresh - horizonCenter.x) / rightDir.x

    leftPoint = horizonCenter + toGoLeft * leftDir
    rightPoint = horizonCenter + toGoRight * rightDir

    toAdd = coordinate.unitFromAngle(yAngle + 90) * 1.0
    leftPoint += toAdd
    rightPoint += toAdd

    corners = [coordinate(x, y) for x in [-1.0, w+1.0] for y in [-1.0, h+1.0]]
    aboveCorners = []
    for corner in corners:
        if (corner.isAboveLine(leftPoint, rightPoint)):
            aboveCorners.append(corner)

    ordered = [leftPoint, rightPoint]
    # print("sorting corners: ", aboveCorners, "\n", ordered)
    while ((len(aboveCorners) + 2) > len(ordered)):
        closestIndex = -1
        largestAngle = 0
        last1 = ordered[-1]
        last2 = ordered[-2]
        for i, corner in enumerate(aboveCorners):
            # print(f"working on {i} which is {corner}")
            if (not (corner in ordered)):
                # print("it's not in the ordered list")
                thisAngle = last1.angle(last2, corner)
                # print(f"calculated angle {thisAngle}")
                if (thisAngle > largestAngle):
                    largestAngle = thisAngle
                    closestIndex = i
                    # print(f"updated to {i}")
        ordered.append(aboveCorners[closestIndex])

    if (len(ordered) == 2):
        return None
    else:
        output = []
        for coord in ordered:
            output.append([round(coord.x), round(coord.y)])
        return np.array(output)