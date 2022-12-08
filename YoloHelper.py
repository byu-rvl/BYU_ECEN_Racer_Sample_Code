# created by Drew Sumsion to use yolov7: https://github.com/WongKinYiu/yolov7
# I suggest you look at detect.py at that github address to see how they implement it.

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


class YoloHelper:
    def __init__(self, img_size = 160):
        # change any of these values to change how you use YOLO
        weights = 'weights/yolov7.pt'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trace = True
        source = 'inference/images'
        webcam = False #TODO set to True
        self.augment = 'store_true'
        self.conf_thres = 0.25
        self.iou_thres = 0.25 # 'object confidence threshold'
        self.classes = '+' # 'filter by class: --class 0, or --class 0 2 3'
        self.agnostic_nms = 'store_true' # class-agnostic NMS

        print(type(device), device)

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

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(self.model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        # Warmup
        # if device.type != 'cpu' and (
        #         old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        #     old_img_b = img.shape[0]
        #     old_img_h = img.shape[2]
        #     old_img_w = img.shape[3]
        #     for i in range(3):
        #         model(img, augment=opt.augment)[0]

    def runNpImage_anySize(self, frame):
        img, ratio, (dw, dh) = letterbox(frame, (img_size, img_size))
        img = np.reshape(img, (img.shape[2], img.shape[1], img.shape[0]))
        return self.runNpImage_correctSize(img, frame)

    def runNpImage_correctSize(self, img, origImage):
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

        # Apply Classifier
        if self.classify:
            pred = apply_classifier(pred, self.modelc, img, origImage)

        print(pred)




if __name__ == "__main__":
    img_size = 160
    testImagePath = "inference/images/horses.jpg"
    cam = cv2.VideoCapture(testImagePath)
    ret,frame = cam.read()

    helper = YoloHelper()
    helper.runNpImage_anySize(frame)