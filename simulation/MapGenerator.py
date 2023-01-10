import numpy as np
import random
from .polygonCalculator import coordinate
from .generator import makeRoads, drawRoads
from .tileGenerator import sideLen
from .TileClass import tileSizeCoord

class Map():
    def __init__(
        self,
        seed = None,
        loops: int = 1,
        size: tuple[int, int] = tuple([8, 8]),
        expansions: int = 10,
        complications: int = 10
    ):
        if (seed is not None):
            if (seed == "real"):
                from .currentMap import getMap
                self.tiles = getMap()
                return
            else:
                np.random.seed(seed)
                random.seed(seed)
        
        if (loops < 1):
            raise Exception("Loops must be >= 1")
        if (size[0] < 2 or size[1] < 2):
            raise Exception("Both dimensions of size must be >= 2")
        if (expansions < 0):
            raise Exception("Expansions must be >= 0")
        if (complications < 0):
            raise Exception("Complications must be >= 0")

        # print("Map generation beginning")
        self.tiles = makeRoads(loops, size, expansions, complications)
        # print("Map generation finished.")

    def getStart(self, startData = None):
        if (startData is None):
            # print("no start data, adding some")
            # loop through our current map to find all possible start locations
            options = []
            for y, row in enumerate(self.tiles):
                for x, tile in enumerate(row):
                    openSides = tile.openCount()
                    if openSides !=3:
                        for i in range(openSides):
                            options.append((y, x, i, 0))
                            options.append((y, x, i, 1))
            # print(f"added all options, have {len(options)} options")
            random.shuffle(options)
            startData = options[0]
        # print("got start data: ", startData)
        y, x, i, dir = startData
        if (
            (len(self.tiles) <= y)
            or (len(self.tiles[y]) <= x)
        ):
            raise Exception(f"Could not get requested start point ({y}, {x}) because it is not in the map.")
        coord, angle = self.tiles[y, x].getStart(i, dir)
        tileCoord = coordinate(x*sideLen, y*sideLen)
        return (coord + tileCoord), angle

    def getImage(self):
        return drawRoads(self.tiles)
    
    def getStatistics(self, position: coordinate, bearing: float): # bearing is in radians
        # compare the current position to the tile at that position (each tile is 24 inches * 8 pixels/inch)
        # and figure out relevant metrics?
        tileIndex = position//tileSizeCoord
        offset = tileIndex * tileSizeCoord
        localCoords = position - offset
        pixelsToCenter, bearingOffset = self.tiles[round(tileIndex.y), round(tileIndex.x)].getStats(localCoords, bearing)
        return pixelsToCenter, bearingOffset