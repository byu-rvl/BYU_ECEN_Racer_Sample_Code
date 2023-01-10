import numpy as np
import matplotlib
# matplotlib.use("qtAgg")
import matplotlib.pyplot as plt
import math
import random
import time
from .TileClass import Tile, addSide
from .tileGenerator import sideLen, junctionTileImg, junctionOrientation, straightTileImg, straightOrientation, curveTileImg, curveOrientation, blankTileImg

def followPath(outputMaze, pathBetween, startCoord):
    directions = "nesw"
    directionAdd = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    currentCoord = startCoord
    for newCoord in pathBetween:
        diff = (newCoord[0] - currentCoord[0], newCoord[1] - currentCoord[1])
        dir = directions[directionAdd.index(diff)]

        outputMaze[currentCoord].openSides(dir)
        outputMaze[newCoord].openSides(addSide(dir, 2))
        currentCoord = newCoord

    return outputMaze

def manDist(t1, t2):
    return abs(t1[0] - t2[0]) + abs(t1[1] - t2[1])

def recursiveFindPath(outputMaze, currentCoords, targetCoords, coordsToUse, usedCoords, lastDirection, startTime, searchPattern = "optimal", currentMinLen = math.inf):
    if ((time.monotonic_ns() - startTime) > 500e6): # longer than 500ms
        return False, []
    if ((len(usedCoords) + manDist(currentCoords, targetCoords)) >= currentMinLen):
        return False, []
    directions = "nesw"
    directionAdd = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    currentTile = outputMaze[currentCoords]
    acceptableDirections = []

    shortestPath = None
    combined = list(zip(directions, directionAdd))
    if (
        (searchPattern == "random")
        or (
            (searchPattern == "turn")
            and (lastDirection is None)
        )
    ):
        random.shuffle(combined)

    if (
        (searchPattern == "turn")
        and (lastDirection is not None)
    ):
        while (not (combined[0][0] == lastDirection)):
            combined.append(combined.pop(0))
        combined.reverse()
        combined.append(combined.pop(0))
        combined.append(combined.pop(0))

    for dir, dirAdd in combined:
        if (not currentTile.isOpen(dir)):
            # great, we can go in this direction
            acceptableDirections.append(dirAdd)
            nextCoord = (currentCoords[0] + dirAdd[0], currentCoords[1] + dirAdd[1])
            if (nextCoord == targetCoords):
                # we did it!
                returnPath = usedCoords.copy()
                returnPath.append(targetCoords)
                return True, returnPath

            # If it's not the target, we'll at least see if it's acceptable
            if (
                (nextCoord in coordsToUse)
                and (nextCoord not in usedCoords)
            ):
                # It's acceptable
                passCoords = usedCoords.copy()
                passCoords.append(nextCoord)

                # See if we can make our way there
                # print("recurisve find path starting at " + str(currentCoords) + " moved " + dir)
                foundPath, testPath = recursiveFindPath(outputMaze, nextCoord, targetCoords, coordsToUse, passCoords, dir, startTime, searchPattern, math.inf if shortestPath is None else len(shortestPath))
                if (foundPath and not (searchPattern == "optimal")):
                    return foundPath, testPath
                if (
                    foundPath
                    and (
                        (shortestPath is None)
                        or (len(testPath) < len(shortestPath))
                    )
                ):
                    shortestPath = testPath
    
    if (shortestPath is not None):
        return True, shortestPath
    else:
        return False, []

def recursiveFindTiles(outputMaze, fromTileCoords, canBridge, cannotBridge, openTiles):
    directions = "nesw"
    directionAdd = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    fromTile = outputMaze[fromTileCoords]
    for index, dir in enumerate(directions):

        if (not fromTile.isOpen(dir)):
            dirAdd = directionAdd[index]
            newTileCoords = (fromTileCoords[0] + dirAdd[0], fromTileCoords[1] + dirAdd[1])
            if (
                (newTileCoords in cannotBridge)
                or (newTileCoords in canBridge)
                or (newTileCoords in openTiles)
            ): # short circuit exit if we've already found it
                continue
            
            if (
                (newTileCoords[0] < 0)
                or (newTileCoords[0] >= outputMaze.shape[0])
                or (newTileCoords[1] < 0)
                or (newTileCoords[1] >= outputMaze.shape[1])
            ):
                # can't go this way, out of bounds.  Don't try it.
                continue
            
            newTile = outputMaze[newTileCoords]
            open = newTile.openCount()
            if (open > 2):
                # can't bridge to this, it already has 3 connections
                cannotBridge.append(newTileCoords)
                continue
            
            if (open == 0):
                openTiles.append(newTileCoords)
                # we need to go deeper and check all directions from here
                canBridge, cannotBridge, openTiles = recursiveFindTiles(outputMaze, newTileCoords, canBridge, cannotBridge, openTiles)
                continue
            
            # if we get here, the open tile must have two openings.  That means it's compatible
            canBridge.append(newTileCoords)
    
    return canBridge, cannotBridge, openTiles

def addLoop(outputMaze):
    loopAdded = False
    canStart = []
    # maxCount = np.prod(np.array(outputMaze.shape))
    # first, find everywhere we can start
    for y, row in enumerate(outputMaze):
        for x, tile in enumerate(row):
            if (tile.openCount() == 2):
                canStart.append((y, x))
    
    # We'll try starting at every one of them if we have to.
    while (len(canStart)):
        attempt = np.random.randint(0, len(canStart))
        y, x = canStart[attempt]
        startTile = outputMaze[y, x]
        # Now we find what we can reach from this point.  Can only travel in the directions
        canBridge = []
        cannotBridge = [(y, x)]
        openTiles = []

        directions = "nesw"
        directionAdd = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for index, dir in enumerate(directions):
            if (startTile.isOpen(dir)):
                # this is the direction we can't go
                dirAdd = directionAdd[index]
                cannotBridge.append((y + dirAdd[0], x + dirAdd[1]))
        # now that we've defined that, we can start looking recursively along other tiles for ones we can bridge to
        canBridge, cannotBridge, openTiles = recursiveFindTiles(outputMaze, (y, x), canBridge, cannotBridge, openTiles)
        # print("Can: \n", canBridge, "\nCannot: \n", cannotBridge, "\n Open: \n", openTiles, "\n-----")

        madePath = False
        while (len(canBridge)):
            bridgeTarget = np.random.randint(0, len(canBridge))
            targetY, targetX = canBridge[bridgeTarget]
            # print("Trying to bridge from (" + str(y) + ", " + str(x) + ") to (" + str(targetY) + ", " + str(targetX) + ")")
            # foundPath, pathBetween = recursiveFindPath(outputMaze, (y, x), (targetY, targetX), openTiles, [], None, searchPattern="optimal")
            foundPath, pathBetween = recursiveFindPath(outputMaze, (y, x), (targetY, targetX), openTiles, [], None, time.monotonic_ns(), searchPattern="turn")
            if (not foundPath):
                canBridge.pop(bridgeTarget)
            else:
                outputMaze = followPath(outputMaze, pathBetween, (y, x))
                madePath = True
                break
        
        if (madePath):
            loopAdded = True
            break
        else:
            canStart.pop(attempt)
        
        # failHereNow()

    return outputMaze, loopAdded

def getBounds(outputMaze):
    left = top = outputMaze.shape[1]
    right = bottom = 0

    for y, row in enumerate(outputMaze):
        for x, tile in enumerate(row):
            if (tile.type() != "empty"):
                if (x < left):
                    left = x
                if (x >= right):
                    right = x + 1
                if (y < top):
                    top = y
                if (y >= bottom):
                    bottom = y + 1
    
    return left, right, top, bottom

def addExpansion(outputMaze):
    hasExpanded = False
    expansionFailures = 0
    while (
        (not hasExpanded)
        and (expansionFailures < 100)
    ):
        left, right, top, bottom = getBounds(outputMaze)
        if (np.random.randint(0, 2) == 1):
            # doing a horizontal expansion: adding a vertical slice
            toExpand = np.random.randint(left + 1, right)
            addTile = Tile("ew")
            indexDiff = 0
            if (left > 0):
                outputMaze[:, 0:toExpand-1] = outputMaze[:, 1:toExpand]
                indexDiff += 1
            elif (right < (outputMaze.shape[1])):
                outputMaze[:, toExpand+1:] = outputMaze[:, toExpand:-1]
            else:
                # raise Exception("Can't expand horizontally.")
                expansionFailures += 1
                continue
            for y in range(outputMaze.shape[0]):
                leftTile = outputMaze[y, toExpand-1]
                comp = leftTile.connected("e", addTile)
                if (comp):
                    outputMaze[y, toExpand-indexDiff] = Tile("ew")
                else:
                    outputMaze[y, toExpand-indexDiff] = Tile()
        else:
            # doing a vertical expansion: adding a horizontal slice
            toExpand = np.random.randint(top + 1, bottom)
            addTile = Tile("ns")
            indexDiff = 0
            if (top > 0):
                outputMaze[0:toExpand-1, :] = outputMaze[1:toExpand, :]
                indexDiff += 1
            elif (bottom < (outputMaze.shape[0])):
                outputMaze[toExpand+1:, :] = outputMaze[toExpand:-1, :]
            else:
                expansionFailures += 1
                continue
            for x in range(outputMaze.shape[1]):
                topTile = outputMaze[toExpand-1, x]
                comp = topTile.connected("s", addTile)
                if (comp):
                    outputMaze[toExpand-indexDiff, x] = Tile("ns")
                else:
                    outputMaze[toExpand-indexDiff, x] = Tile()
        hasExpanded = True
    return outputMaze, hasExpanded

def addComplexity(outputMaze):
    # This looks something like:
    
    # 1. Pick a random tile
    # 2. Pick a random direction to move it
    # 3. Correct adjacent tiles so that doesn't cause problems

    startTime = time.monotonic_ns()

    complexityAdded = False

    possiblePositions = []
    for y, row in enumerate(outputMaze):
        for x, tile in enumerate(row):
            if (tile.openCount() > 0):
                possiblePositions.append((y, x))
    
    random.shuffle(possiblePositions)
    while (len(possiblePositions)):
        if ((time.monotonic_ns() - startTime) > 100e6):
            # if it's taken longer than 100ms, bail.  We'll find an easier one.
            break
        currentCoord = possiblePositions[0]
        tile = outputMaze[currentCoord]
        directions = "nesw"
        directionAdd = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        combined = list(zip(directions, directionAdd))
        random.shuffle(combined)
        moved = False
        while (len(combined)):
            dir, dirAdd = combined[0]
            # print(f"working with {dir} and {dirAdd}")
            newCoord = (currentCoord[0] + dirAdd[0], currentCoord[1] + dirAdd[1])
            if (
                (newCoord[0] < 0)
                or (newCoord[0] >= outputMaze.shape[0])
                or (newCoord[1] < 0)
                or (newCoord[1] >= outputMaze.shape[1])
            ):
                # it's out of bounds
                combined.pop(0)
                continue

            newTile = outputMaze[newCoord]
            oldType = tile.type()
            newType = newTile.type()

            if (newType == "t-junc"):
                # in this case, we're trying to move it into a tile that has 3 connections, and that might not fly.
                # Avoid the complexity.
                combined.pop(0)
                continue
            elif (newType == "straight"):
                if (oldType == "straight"):
                    # it's straight into straight.  That's... not going to add any complexity.
                    # Either it's boring, or it's impossible.
                    combined.pop(0)
                    continue
                if (oldType == "t-junc"):
                    # t-junc into straight is possible if they're connected, they might not be.
                    if (tile.connected(dir, newTile)):
                        # great!

                        # The t-junc will be open on 2 sides in directions it's not moving.
                        # we need to handle those sides.
                        unhandledSides = tile.getOpenSides().replace(dir, "")
                        allWork = True
                        for side in unhandledSides:
                            if (addSide(dir, 2) == side):
                                # if the direction is OPPOSITE the direction of movement, we need to add in a straight
                                allWork &= True
                            else:
                                # if the direction is not opposite, we need to close that connecting side,
                                # open up the side in the direction of movement, and add a new corner.
                                # check to see that we can add that new corner
                                checkDir = directionAdd[directions.index(side)]
                                checkCoord = (newCoord[0] + checkDir[0], newCoord[1] + checkDir[1])
                                allWork &= (outputMaze[checkCoord].type() == "empty")

                        if (allWork):
                            # We can actually make the changes
                            outputMaze[newCoord] = tile
                            outputMaze[currentCoord] = Tile()
                            for side in unhandledSides:
                                if (addSide(dir, 2) == side):
                                    outputMaze[currentCoord] = Tile(dir + side)
                                else:
                                    changeDir = directionAdd[directions.index(side)]
                                    changeTileCoords = (currentCoord[0] + changeDir[0], currentCoord[1] + changeDir[1])
                                    changeTile = outputMaze[changeTileCoords]
                                    changeTile.closeSides(addSide(side, 2))
                                    changeTile.openSides(dir)
                                    addTileCoords = (newCoord[0] + changeDir[0], newCoord[1] + changeDir[1])
                                    addTile = outputMaze[addTileCoords]
                                    addTile.openSides(addSide(side, 2) + addSide(dir, 2))
                            moved = True
                            break
                        else:
                            # can't do it
                            combined.pop(0)
                            continue
                    else:
                        # can't do it
                        combined.pop(0)
                        continue

                if (oldType == "turn"):
                    # turn into straight is possible if they're connected, they might not be.
                    if (tile.connected(dir, newTile)):
                        # great!

                        # The turn is also connected to something else.
                        # that something else needs to be closed where they're connected, and
                        # opened in the direction that the tile is moving.
                        # The tile it's now opened to better be avaialble.
                        unhandledSide = tile.getOpenSides().replace(dir, "")
                        # The direction is not opposite, we need to close that connecting side,
                        # open up the side in the direction of movement, and add a new corner.
                        # check to see that we can add that new corner
                        checkDir = directionAdd[directions.index(unhandledSide)]
                        checkCoord = (newCoord[0] + checkDir[0], newCoord[1] + checkDir[1])
                        allWork = (outputMaze[checkCoord].type() == "empty")

                        if (allWork):
                            # We can actually make the changes
                            outputMaze[newCoord] = tile
                            outputMaze[currentCoord] = Tile()
                            changeDir = directionAdd[directions.index(unhandledSide)]
                            changeTileCoords = (currentCoord[0] + changeDir[0], currentCoord[1] + changeDir[1])
                            changeTile = outputMaze[changeTileCoords]
                            changeTile.closeSides(addSide(unhandledSide, 2))
                            changeTile.openSides(dir)
                            addTileCoords = (newCoord[0] + changeDir[0], newCoord[1] + changeDir[1])
                            addTile = outputMaze[addTileCoords]
                            addTile.openSides(addSide(unhandledSide, 2) + addSide(dir, 2))
                            moved = True
                            break
                        else:
                            # can't do it
                            combined.pop(0)
                            continue
                    else:
                        # can't do it
                        combined.pop(0)
                        continue
            elif (newType == "turn"):
                # Moving something INTO a turn doesn't really work, it'd be ugly.
                combined.pop(0)
                continue
            elif (newType == "empty"):
                # this should be find, as long as the necessary adjacent tiles are open.
                # great!
                unhandledSides = tile.getOpenSides().replace(dir, "")
                allWork = True
                for side in unhandledSides:
                    if (addSide(dir, 2) == side):
                        # if the direction is OPPOSITE the direction of movement, we need to add in a straight
                        allWork &= True
                    else:
                        # if the direction is not opposite, we need to close that connecting side,
                        # open up the side in the direction of movement, and add a new corner.
                        # check to see that we can add that new corner
                        checkDir = directionAdd[directions.index(side)]
                        checkCoord = (newCoord[0] + checkDir[0], newCoord[1] + checkDir[1])
                        allWork &= (outputMaze[checkCoord].type() == "empty")

                if (allWork):
                    # We can actually make the changes
                    outputMaze[newCoord] = tile
                    outputMaze[currentCoord] = Tile()
                    for side in unhandledSides:
                        if (addSide(dir, 2) == side):
                            outputMaze[currentCoord] = Tile(dir + side)
                        else:
                            changeDir = directionAdd[directions.index(side)]
                            changeTileCoords = (currentCoord[0] + changeDir[0], currentCoord[1] + changeDir[1])
                            changeTile = outputMaze[changeTileCoords]
                            changeTile.closeSides(addSide(side, 2))
                            changeTile.openSides(dir)
                            addTileCoords = (newCoord[0] + changeDir[0], newCoord[1] + changeDir[1])
                            addTile = outputMaze[addTileCoords]
                            addTile.openSides(addSide(side, 2) + addSide(dir, 2))
                    moved = True
                    break
                else:
                    # can't do it
                    combined.pop(0)
                    continue



        if (moved):
            # great, we did it!
            complexityAdded = True
            break
        else:
            possiblePositions.pop(0)

    return outputMaze, complexityAdded

def makeRoads(
    # loops: int = 1,
    loops,
    # dimensions: tuple(int, int) = tuple([8, 8]), # in order rows x cols
    dimensions,
    expansion: int = 4,
    complexity: int = 10
):
    # print(f"MakeRoads called with loops {loops}, dimensions {dimensions}, expansion {expansion}, complexity {complexity}")
    baseHeight = 2
    baseWidth = 2
    # baseHeight += loops
    # if (baseHeight > dimensions[0]):
    #     raise Exception("Cannot generate " + str(loops) + " loops with given dimensions (" + str(dimensions) + ")")
    
    outputMaze = np.full(dimensions, Tile())
    for y in range(dimensions[0]):
        for x in range(dimensions[1]):
            outputMaze[y, x] = Tile()

    margins = (dimensions[0] - baseHeight, dimensions[1] - baseWidth)
    top = margins[0]//2
    left = margins[1]//2

    # Start with the simplest loop, in the center
    outputMaze[top, left].openSides("es")
    outputMaze[top, left+1].openSides("sw")
    # for i in range(top+1, top+baseHeight-1):
    #     outputMaze[i, left].openSides("nes")
    #     outputMaze[i, left+1].openSides("nsw")
    # outputMaze[top+baseHeight-1, left].openSides("ne")
    # outputMaze[top+baseHeight-1, left+1].openSides("nw")
    outputMaze[top+1, left].openSides("ne")
    outputMaze[top+1, left+1].openSides("nw")

    loopCount = 1
    right = left + 2
    bottom = top + 2

    # now we try to complicate things
    failures = 0
    while (
        (
            (loopCount < loops)
            or (expansion > 0)
            or (complexity > 0)
        ) and (
            failures < 10
        )
    ):
        options = []
        for i in range (loops - loopCount):
            options.append("loop")
            options.append("loop") # weighting it twice
            options.append("loop") # weighting it thrice
        for i in range(expansion):
            options.append("expansion")
        for i in range(complexity):
            options.append("complexity")

        which = options[np.random.randint(0, len(options))]
        if (which == "loop"):
            outputMaze, loopAdded = addLoop(outputMaze)
            if (loopAdded):
                loopCount += 1
                failures = 0
            else:
                failures += 1
        elif (which == "expansion"):
            outputMaze, expanded = addExpansion(outputMaze)
            if (expanded):
                expansion -= 1
                failures = 0
            else:
                failures += 1
        elif (which == "complexity"):
            outputMaze, complexityAdded = addComplexity(outputMaze)
            if (complexityAdded):
                complexity -= 1
                failures = 0
            else:
                failures += 1

    return outputMaze

def drawRoads(roads):
    outputImage = np.zeros((sideLen*len(roads), sideLen*len(roads[0]), 3), np.uint8)
    for y, row in enumerate(roads):
        for x, tile in enumerate(row):
            imgY = sideLen*y
            imgX = sideLen*x
            outputImage[imgY:imgY + sideLen, imgX:imgX + sideLen] = tile.getImage()
    return outputImage

def makeMap(
    loopCount: int = 1, # must be greater than 0
    size: tuple[int, int] = tuple([8, 8]), # number of tiles in map
    expansions: int = 0, # higher = more spread out map
    complications: int = 0, # higher = more convoluted map
):
    roads = makeRoads(loopCount, size, expansions, complications)
    img = drawRoads(roads)
    return img, roads
    

# # roads = makeRoads(2, (8, 8), 0, 0)
# # for i in range (2, 9):
# # for i in range (2, 7):
# now = time.monotonic_ns()
# # roads = makeRoads(4, (10, 10), 10, 3)
# # roads = makeRoads(4, (8, 8), 15, 5)
# # from currentMap import getMap
# # roads = getMap()
# # roads = makeRoads(40, (30, 30), 50, 50)
# roads = makeRoads(1, (8, 8), 20, 2)
# # roads = makeRoads(1, (10, 10), 1, 1)
# end = time.monotonic_ns()
# diff = (end - now)/1e6
# print("Diff was: ", diff)
# # roads = makeRoads(1, (8, 8), 0, 10)
# # roads = makeRoads(3, (8, 8), 10, 10)
# img = drawRoads(roads)
# cv2.namedWindow("test", cv2.WINDOW_NORMAL)
# cv2.imshow("test", img)
# cv2.waitKey(0)
# # cv2.imshow("test", straight)
# # cv2.waitKey(0)
# # cv2.imshow("test", junction)
# # cv2.waitKey(0)

# # cv2.imshow()
# # plt.imshow(img, interpolation='none')
# # plt.show()