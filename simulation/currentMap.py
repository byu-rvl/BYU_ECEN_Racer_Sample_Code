from .TileClass import Tile, addSide
import numpy as np

def getMap():
    return np.array([
        [Tile(),      Tile("es"),  Tile("ew"),  Tile("ew"),  Tile("esw"), Tile("ew"),  Tile("ew"),  Tile("sw") ],
        [Tile("es"),  Tile("wn"),  Tile(),      Tile(),      Tile("ns"),  Tile(),      Tile(),      Tile("ns") ],
        [Tile("ns"),  Tile(),      Tile("es"),  Tile("ew"),  Tile("new"), Tile("ew"),  Tile("ew"),  Tile("nsw")],
        [Tile("ns"),  Tile(),      Tile("ns"),  Tile(),      Tile(),      Tile(),      Tile(),      Tile("ns") ],
        [Tile("ns"),  Tile(),      Tile("ns"),  Tile(),      Tile(),      Tile(),      Tile(),      Tile("ns") ],
        [Tile("nes"), Tile("ew"),  Tile("new"), Tile("ew"),  Tile("ew"),  Tile("esw"), Tile("ew"),  Tile("wn") ],
        [Tile("ns"),  Tile(),      Tile(),      Tile(),      Tile(),      Tile("ns"),  Tile(),  Tile() ],
        [Tile("ne"),  Tile("ew"),  Tile("ew"),  Tile("ew"),  Tile("ew"),  Tile("wn"),  Tile(),  Tile() ]
    ])