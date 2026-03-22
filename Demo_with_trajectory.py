import cv2
from ultralytics import YOLO
from trackertrajectory import RobustBoxTracker  # Make sure this file is in the same dir
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Or your custom model

# Initialize tracker
tracker = RobustBoxTracker(max_disappeared=30, max_distance=80)

# Open video
cap = cv2.VideoCapture("input.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Optional video output
out = cv2.VideoWriter("RobustSort.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.5, iou=0.4, verbose=False)
    detections = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []

    # Convert detections to list of tuples
    rects = [tuple(map(int, box)) for box in detections]

    # Update tracker
    objects = tracker.update(rects, frame_number=frame_num)
    trajectories = tracker.get_trajectories()
  
