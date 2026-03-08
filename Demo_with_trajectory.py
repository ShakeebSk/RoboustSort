import cv2
from ultralytics import YOLO
from trackertrajectory import RobustBoxTracker  # Make sure this file is in the same dir
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Or your custom model

# Initialize tracker
tracker = RobustBoxTracker(max_disappeared=30, max_distance=80)
