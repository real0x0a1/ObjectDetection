#!/usr/bin/python3

# -*- Author: Ali (Real0x0a1) -*-
# -*- Description: Real-time Object Detection using YOLOv3 -*-

import cv2
import numpy as np
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths for YOLO weights and configuration files
weights_path = os.path.join(script_dir, 'scripts/yolov3.weights')
cfg_path = os.path.join(script_dir, 'scripts/yolov3.cfg')
names_path = os.path.join(script_dir, 'scripts/coco.names')

# Load YOLO network
net = cv2.dnn.readNet(weights_path, cfg_path)

# Load COCO class labels
classes = []
with open(names_path, 'r') as f:
    classes = f.read().strip().split('\n')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Get the height and width of the frame
    height, width, _ = frame.shape

    # Create a blob from the frame and pass it through the YOLO network
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize lists for object detection
    class_ids = []
    confidences = []
    boxes = []

    # Process the outputs from YOLO
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Adjust confidence threshold as needed
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression to remove duplicate detections
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the frame
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the result
    cv2.imshow('Object Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
