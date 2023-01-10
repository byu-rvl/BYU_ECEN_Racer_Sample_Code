import cv2
import numpy as np
import math

tileInches = 24
pixelsPerInch = 8
borderInches = 2
centerWidth = 1
centerLength = 2
centerGap = 1

white = (255, 255, 255)
yellow = (0, 255, 255)
black = (0, 0, 0)
red = (0, 0, 255)

borderPixels = borderInches * pixelsPerInch
centerPixels = centerWidth * pixelsPerInch
sideLen = tileInches * pixelsPerInch

corner = np.zeros((sideLen, sideLen, 3), np.uint8) # Remember image is BGR
# If we had a path along the center, it would be (1/2 pi * 12 inches (radius)) inches long.
# (Arc length = angle * radius)
# That gives us 6pi inches.
# In general, we're trying to subdivide the center lines into ~ 3 inch segments.
# pi basically == 3, since we're engineers.
# That means I want to do a (pi-2)/2 inch gap, then a 2 inch strip, then another (pi-2)/2 inch gap
# Repeat 6 times for this.
lastAngle = math.pi
gapAngle = ((math.pi - 2)/2)/(tileInches/2)
lineAngle = 2/(tileInches/2)
for i in range(6):
    lastAngle += gapAngle
    startPoint = ((sideLen/2)*math.cos(lastAngle) + sideLen, -(sideLen/2)*math.sin(lastAngle))
    perpAngle = lastAngle + (lineAngle/2)
    lastAngle += lineAngle
    endPoint = ((sideLen/2)*math.cos(lastAngle) + sideLen, -(sideLen/2)*math.sin(lastAngle))
    lastAngle += gapAngle

    xDiff = centerPixels/2 * math.cos(perpAngle)
    yDiff = centerPixels/2 * math.sin(perpAngle)

    fractionalBits = 8
    polyPoints = np.array([
        [startPoint[0] - xDiff, startPoint[1] + yDiff],
        [startPoint[0] + xDiff, startPoint[1] - yDiff],
        [endPoint[0] + xDiff, endPoint[1] - yDiff],
        [endPoint[0] - xDiff, endPoint[1] + yDiff],
    ]) * (2 ** fractionalBits)
    polyPoints = np.rint(polyPoints)

    corner = cv2.fillPoly(corner, np.int32([polyPoints]), yellow, cv2.LINE_AA, fractionalBits)
cv2.circle(corner, (sideLen, 0), sideLen - borderPixels//2, white, borderPixels)
cv2.circle(corner, (sideLen, 0), borderPixels//2, white, borderPixels)

#############
straight = np.zeros((sideLen, sideLen, 3), np.uint8) # Remember image is BGR

#Yellow line in the middle
straight = cv2.line(straight, (0, sideLen//2), (sideLen, sideLen//2), yellow, centerPixels, cv2.LINE_AA)

#Chop up that line
step = (centerGap + centerLength)
toIter = np.arange(0, tileInches+step/2, 3) # +step/2 is an ugly hack to get the endpoint included in the range, but whatever.
for i in toIter:
    xCoord = round(i*pixelsPerInch)
    straight = cv2.line(straight, (xCoord, 0), (xCoord, sideLen), black, centerPixels, cv2.LINE_AA)

#Top and bottom borders
straight = cv2.line(straight, (0, borderPixels//2), (sideLen, borderPixels//2), white, borderPixels)
straight = cv2.line(straight, (0, sideLen-borderPixels//2), (sideLen, sideLen-borderPixels//2), white, borderPixels)


##############
junction = np.zeros((sideLen, sideLen, 3), np.uint8) # Remember image is BGR

#Top red line
redPoints = np.array([
    [borderPixels, 0],
    [sideLen//2 - centerPixels//2, 0],
    [sideLen//2 - centerPixels//2, borderPixels],
    [borderPixels, borderPixels]
])
junction = cv2.fillPoly(junction, np.int32([redPoints]), red, cv2.LINE_AA)

#Right red line
redPoints = np.array([
    [sideLen - borderPixels, borderPixels],
    [sideLen, borderPixels],
    [sideLen, sideLen//2 - centerPixels//2],
    [sideLen - borderPixels, sideLen//2 - centerPixels//2]
])
junction = cv2.fillPoly(junction, np.int32([redPoints]), red, cv2.LINE_AA)

# Bottom red line
redPoints = np.array([
    [sideLen - borderPixels, sideLen],
    [sideLen//2 + centerPixels//2, sideLen],
    [sideLen//2 + centerPixels//2, sideLen - borderPixels],
    [sideLen - borderPixels, sideLen-borderPixels]
])
junction = cv2.fillPoly(junction, np.int32([redPoints]), red, cv2.LINE_AA)

# Left border
junction = cv2.line(junction, (borderPixels//2, 0), (borderPixels//2, sideLen), white, borderPixels, cv2.LINE_AA)

# Top Right square/border
whitePoints = np.array([
    [sideLen, 0],
    [sideLen - borderPixels, 0],
    [sideLen - borderPixels, borderPixels],
    [sideLen, borderPixels]
])
junction = cv2.fillPoly(junction, np.int32([whitePoints]), white, cv2.LINE_AA)

# Bottom Right square/border
whitePoints = np.array([
    [sideLen, sideLen],
    [sideLen - borderPixels, sideLen],
    [sideLen - borderPixels, sideLen - borderPixels],
    [sideLen, sideLen - borderPixels]
])
junction = cv2.fillPoly(junction, np.int32([whitePoints]), white, cv2.LINE_AA)

# Top yellow
yellowPoints = np.array([
    [sideLen//2 - centerPixels//2, 0],
    [sideLen//2 + centerPixels//2, 0],
    [sideLen//2 + centerPixels//2, borderPixels],
    [sideLen//2 - centerPixels//2, borderPixels]
])
junction = cv2.fillPoly(junction, np.int32([yellowPoints]), yellow, cv2.LINE_AA)

# Bottom yellow
yellowPoints = np.array([
    [sideLen//2 - centerPixels//2, sideLen],
    [sideLen//2 + centerPixels//2, sideLen],
    [sideLen//2 + centerPixels//2, sideLen - borderPixels],
    [sideLen//2 - centerPixels//2, sideLen - borderPixels]
])
junction = cv2.fillPoly(junction, np.int32([yellowPoints]), yellow, cv2.LINE_AA)

# Right yellow
yellowPoints = np.array([
    [sideLen - borderPixels, sideLen//2 - centerPixels//2],
    [sideLen, sideLen//2 - centerPixels//2],
    [sideLen, sideLen//2 + centerPixels//2],
    [sideLen - borderPixels, sideLen//2 + centerPixels//2]
])
junction = cv2.fillPoly(junction, np.int32([yellowPoints]), yellow, cv2.LINE_AA)



# cv2.namedWindow("test", cv2.WINDOW_NORMAL)
# cv2.imshow("test", corner)
# cv2.waitKey(0)
# cv2.imshow("test", straight)
# cv2.waitKey(0)
# cv2.imshow("test", junction)
# cv2.waitKey(0)



junctionTileImg = junction
junctionOrientation = "nes"

straightTileImg = straight
straightOrientation = "ew"

curveTileImg = corner
curveOrientation = "ne"

blankTileImg = np.zeros((sideLen, sideLen, 3), np.uint8)