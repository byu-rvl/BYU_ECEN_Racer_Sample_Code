import math
import numpy as np
import warnings
import cv2
from .polygonCalculator import calculatePolygon
from scipy.spatial.transform import Rotation

class Camera:
    def __init__(self, image, resolution=(1920, 1080), fov={'diagonal': 77}, angle={'roll': 0, 'pitch': 0, 'yaw': 0}, height=48, M=None, D=None):
        """
        Initialization - intantiate the class for the simulation.
        @Params:
        resolution: tuple(int, int) - should be (cols, rows)
        fov: dict{string: int} holding "diagonal," "horizontal," or "vertical" - only one is necessary, priority goes to the largest dimension
        angle: dict{string: int} holding:
            "roll" (rotation about the axis pointing straight out of the camera),
            "pitch" (rotation that would point the camera more or less toward the ground)
            "yaw" (rotation that would point the camera away from the heading/bearing of the car)
            Units are the degrees offset from perfectly-aligned
        height: pixels the camera is placed above the ground (make sure the units match with the images you pass)
        """

        # Start with sanity checks on the resolution argument
        if (
            (len(resolution) != 2)
            or (type(resolution[0]) != int)
            or (type(resolution[1]) != int)
        ):
            raise Exception("Resolution should be a tuple of ints representing the number of horizontal and vertical pixels as (h, v)")
    
        # extract som enumbers for calculation and save
        x, y = resolution
        diag = math.sqrt(x**2 + y**2)
        self.resolution = resolution

        # Start handling the field of view
        saveFOV = {}
        degreesPerPixel = 0
        if ("diagonal" in fov):
            degreesPerPixel = fov["diagonal"] / diag
        elif (
            (x >= y)
            and ("horizontal" in fov)
        ):
            degreesPerPixel = fov["horizontal"] / x
        elif (
            (y >= x)
            and ("vertical" in fov)
        ):
            degreesPerPixel = fov["vertical"] / y
        elif ("horizontal" in fov):
            degreesPerPixel = fov["horizontal"] / x
        elif ("vertical" in fov):
            degreesPerPixel = fov["vertical"] / y
        else:
            raise Exception("Could not initialize camera simulator, field of view argument (fov) does not contain 'diagonal', 'horizontal', or 'vertical'")

        # Calculate exact fields of view and save
        saveFOV['diagonal'] = round(degreesPerPixel * diag)
        saveFOV['horizontal'] = round(degreesPerPixel * x)
        saveFOV['vertical'] = round(degreesPerPixel * y)
        self.fov = saveFOV

        # Check for all the angle arguments, should have all or else we warn
        if (not all(i in angle for i in ['roll', 'pitch', 'yaw'])):
            warnings.warn("Angle argument should be a dict containing keys 'roll' AND 'pitch' AND 'yaw' representing offset in degrees from perfect alignment - assuming 0 for missing values.")
        
        # Make sure it's in the right format and save.
        saveAngle = {
            'roll': 0,
            'pitch': 0,
            'yaw': 0
        }
        for i in ['roll', 'pitch', 'yaw']:
            if i in angle:
                saveAngle[i] = angle[i]
        self.angle = saveAngle

        if (height <= 0):
            raise Exception("Height is too low, must be greater than 0")
        self.height = height

        # Now that we have the FOV info, we can modify the image
        # to be the right aspect ratio, and figure out where the camera
        # must initially be positioned (and thus the focal length)
        # in order to see the whole thing.

        #FOV = 2 arctan (x / (2 f))
        # that's the diagonal fov, and x is the diagonal size.
        # We want both measurements in pixels.
        # We already have "diag" but that's of the OUTPUT, and we're
        # not quite there yet.  We need to make the input image match
        # by padding with black.

        imShape = image.shape[:2]
        ih, iw = imShape
        ar = x / y # aspect ratio
        # height * ar = width
        # width / ar = height
        # see which is bigger

        targetHeight = round(iw / ar)
        targetWidth = round(ih * ar)

        self.baseImage = None

        if (targetHeight < ih):
            # keep height constant, control width
            if (targetWidth >= iw):
                wDiff1 = targetWidth - iw
                wDiff2 = wDiff1//2
                wDiff1 -= wDiff1
                self.baseImage = cv2.copyMakeBorder(image, 0, 0, wDiff1, wDiff2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        elif (targetWidth < iw):
            # keep width constant, control height
            if (targetHeight >= ih):
                wHeight1 = targetHeight - ih
                wHeight2 = wHeight1//2
                wHeight1 -= wHeight2
                self.baseImage = cv2.copyMakeBorder(image, wHeight1, wHeight2, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        if (self.baseImage is None):
            self.baseImage = image.copy()
        
        self.h, self.w = self.baseImage.shape[:2]
        h, w = self.baseImage.shape[:2]
        d = np.sqrt(h**2 + w**2)
        # FOV = 2 arctan (x / (2 f))
        # FOV is known, though, so
        # tan(FOV / 2) = d / (2f)
        # fov must be < 180 degrees or else problems
        # f = (d/2)/(tan(FOV/2))
        f = (d/2)/(np.tan(np.deg2rad(self.fov['diagonal']) / 2))
        self.f = f
        # that must also be initial z dist, but that doesn't really matter.
        # ... I think.  Anyway, now we can do other calcs.

        rotXDeg = (-90 + self.angle['pitch'])
        self.rotXDeg = rotXDeg
        rotYDeg = (self.angle['roll'])
        self.rotYDeg = rotYDeg
        rotZDeg = (90 - self.angle['yaw'])
        rotX = np.deg2rad(rotXDeg)
        rotY = np.deg2rad(rotYDeg)
        rotZ = np.deg2rad(rotZDeg)

        # Projection 2D -> 3D matrix
        A1= np.matrix([[1, 0, -w/2],
                       [0, 1, -h/2],
                       [0, 0, 0   ],
                       [0, 0, 1   ]])

        # # Rotation matrices around the X,Y,Z axis
        # RX = np.matrix([[1,            0,             0, 0],
        #                 [0, np.cos(rotX), -np.sin(rotX), 0],
        #                 [0, np.sin(rotX),  np.cos(rotX), 0],
        #                 [0,            0,             0, 1]])

        # RY = np.matrix([[  np.cos(rotY), 0, np.sin(rotY), 0],
        #                 [             0, 1,            0, 0],
        #                 [ -np.sin(rotY), 0, np.cos(rotY), 0],
        #                 [             0, 0,            0, 1]])

        # RZ = np.matrix([[ np.cos(rotZ), -np.sin(rotZ), 0, 0],
        #                 [ np.sin(rotZ),  np.cos(rotZ), 0, 0],
        #                 [            0,             0, 1, 0],
        #                 [            0,             0, 0, 1]])

        # # Composed rotation matrix with (RX,RY,RZ)
        # R = RX * RY * RZ

        R = self.homogenousRot(Rotation.from_euler("zyx", [rotZ, rotY, rotX], degrees=False).as_matrix())

        # Translation matrix on the Z axis change dist will change the height
        T = np.matrix([[1,0,0,0],
                       [0,1,0,0],
                       [0,0,1,-height],
                       [0,0,0,1]])

        # extractT = T[:3, 3:4]
        # solveT = -R[:3, :3]@extractT
        # T[:3, 3:4] = solveT

        # Camera Intrisecs matrix 3D -> 2D
        A2 = np.matrix([[f, 0, w/2, 0],
                        [0, f, h/2, 0],
                        [0, 0,   1, 0]])

        self.A1 = A1
        self.R = R
        self.T = T
        self.A2 = A2

    @staticmethod
    def translateHomogenous(tVec):
        tVec = np.squeeze([np.array(tVec)])
        return np.block([[np.eye(3), np.array([tVec]).T],[np.array([[0, 0, 0]]), np.array([1])]])

    @staticmethod
    def invertTransform(trans):
        R = trans[:3, :3]
        t = trans[:3, 3:4]
        out = np.block([[R.T, -R.T@t], [np.zeros((1, 3)), np.ones((1, 1))]])
        return out

    @staticmethod
    def homogenousRot(rot):
        R = rot
        t = np.array([[0, 0, 0]]).T
        out = np.block([[R, t], [np.zeros((1, 3)), np.ones((1, 1))]])
        return out

    def getImage(self, location=(0.0, 0.0), facing=0.0):
        """
        getImage - Calculates and returns the image seen by the camera
        @Params:
        location: tuple(float, float) representing x, y coordinates of the camera
        facing: float representing the angle the camera is facing in radians
        """
        
        extraT = self.translateHomogenous([-location.x + self.w/2, -location.y + self.h/2, 0])
        extraR = self.homogenousRot(Rotation.from_euler("z", [-facing], degrees=False).as_matrix())
        H = self.A2@((self.R @ extraR @ self.T @ extraT) @ self.A1)
        outImg = cv2.warpPerspective(self.baseImage, H, (self.w, self.h), flags=cv2.INTER_CUBIC)
        points = calculatePolygon(self.h, self.w, self.f, self.rotXDeg, self.rotYDeg)
        if (points is not None):
            cv2.fillPoly(outImg, [points], (0, 0, 0))
        return cv2.resize(outImg, self.resolution)

        
        verticalRotationVector = [0, self.angle['vertical'] - self.fov['vertical']/2, 0]
        centerVector = np.array([1, 0, 0])

        bottomLeftRot = R.from_euler('xyz', [
            verticalRotationVector,
            [0, 0, facing + self.angle['horizontal'] + self.fov['horizontal']/2]
        ], degrees=True)
        bottomRightRot = R.from_euler('xyz', [
            verticalRotationVector,
            [0, 0, facing + self.angle['horizontal'] - self.fov['horizontal']/2]
        ], degrees=True)
        
#         np.

#         r = R.from_euler('zyx', [
# ... [90, 0, 0],
# ... [0, 45, 0],
# ... [45, 60, 30]], degrees=True)
        
