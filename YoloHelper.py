# created by Drew Sumsion to use yolov7: https://github.com/WongKinYiu/yolov7
# I suggest you look at detect.py at that github address to see how they implement it.

import glob
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, TracedModel


class YoloHelper:
    def __init__(self, img_size = 160, webcam=False):
        # change any of these values to change how you use YOLO
        weights = 'weights/yolov7.pt'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trace = True
        self.augment = False
        self.conf_thres = 0.25
        self.iou_thres = 0.45 # 'object confidence threshold'
        self.classes = None # 'filter by class: --class 0, or --class 0 2 3'
        self.agnostic_nms = False # class-agnostic NMS

        # load in information to use throughout the class. Much of rest of init comes from detect.py from YOLOv7
        imgsz = img_size
        self.half = device.type != 'cpu'  # half precision only supported on CUDA
        self.device = select_device(str(device))
        self.model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if trace:
            self.model = TracedModel(self.model, device, img_size)

        if self.half:
            self.model.half()  # to FP16

        if webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(self.model.parameters())))  # run once

        # Warmup the GPU
        img = np.zeros((1080,1920,3))
        for i in range(0,3):
            self.runNpImage_anySize(img)

    def runNpImage_anySize(self, frame):
        """Run YOLO on normal image.
            Input:
                frame: np.array Normal np.array representation from image in OpenCV

            Return:
                det: See runNPImage_correctSize function below for syntax.
        """

        img = letterbox(frame, (img_size, img_size))[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)


        return self.runNpImage_correctSize(img, frame)

    def runNpImage_correctSize(self, img, origImage):
        """ Run YOLO
            Input:
                img: np.array. Resized and reshaped image. See runNpImage_anySize function above.
                origImage: np.array. Normal np.array representation from image in OpenCV

            Returns:
                det: A tensor of the items and locations of found items. In form:
                            tensor([[top_x, top_y, bottom_x, bottom_y, confidence, label_number], repeat])
                            where top_x, top_y, etc. are the corners of the bounding box for the labeled item.
        """



        # Inference
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.augment)[0]
        # Apply NMS
        det = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)[0]

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], origImage.shape).round()

        return det

    def drawBBox(self, det, origImage, display=True):
        """
            Draws a bounding box on image

            inputs:
                det: The output from runNpImage_correctSize or runNpImage_anySize from above.
                origImage: the image that det corresponds to. In normal np.array representation of OpenCV image.
                display: boolean. Whether to display the image or not.
            outputs:
                bboxImage: image with bounding boxes drawn.
        """
        bboxImage = origImage.copy()
        for *xyxy, conf, cls in reversed(det):
            label = f'{self.names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, bboxImage, label=label, color=self.colors[int(cls)], line_thickness=1)

        if display:
            cv2.imshow("image", bboxImage)
            cv2.waitKey(0)

        return bboxImage


if __name__ == "__main__":
    img_size = 160
    testImagePath = "inference/images/horses.jpg"
    cam = cv2.VideoCapture(testImagePath)
    ret,frame = cam.read()

    dataFiles = "../archive/images/" #TODO change this to the path of the dataset desired.
    allFiles = list(glob.glob(dataFiles + "**.png"))
    print("length of all files", len(allFiles))

    multipleImages = True

    helper = YoloHelper()
    time_here = []
    if multipleImages:
        totalImages = len(allFiles)
    else:
        totalImages = 1000
    analyzed = 0
    for i in range(0, totalImages):
        if multipleImages:
            cam = cv2.VideoCapture(allFiles[i])
            ret,frame = cam.read()
            if not ret:
                continue
            analyzed += 1
        else:
            analyzed += 1
        start = time.time()
        det = helper.runNpImage_anySize(frame)
        # helper.drawBBox(det,frame,display=True)
        end = time.time()
        time_here.append(end-start)
    time_here = np.array(time_here)
    print("Stats:")
    print("Median time:", np.median(time_here), "seconds. Or fps:", 1/np.median(time_here))
    print("Mean time:", np.mean(time_here), "seconds", 1/np.mean(time_here))
    print("Max time:", np.max(time_here), "seconds", 1/np.max(time_here))
    print("Min time:", np.min(time_here), "seconds", 1/np.min(time_here))
    print("Total images:", totalImages)
    print("Total images analyzed:", analyzed)