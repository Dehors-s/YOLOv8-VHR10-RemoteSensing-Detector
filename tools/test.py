from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO

from ultralytics.yolo.utils import  ROOT


CFG = 'yolov8s_CBAM.yaml'
SOURCE = ROOT / 'assets/bus.jpg'



def test_model_forward():
    model = YOLO(CFG)
    model(SOURCE)