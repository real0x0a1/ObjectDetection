# Real-time Object Detection using YOLOv3

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) 

## Overview

This Python script uses the YOLOv3 (You Only Look Once) model for real-time object detection using a webcam feed. It detects objects in the frame and draws bounding boxes around them with labels and confidence scores.

## Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- NumPy (`numpy`)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Real0x0a1/ObjectDetection.git
   ```

2. Install required dependencies:

   ```bash
   pip3 install opencv-python numpy
   ```

3. Download YOLOv3 weights file:

   - YOLOv3 Weights: [Download YOLOv3 Weights](https://pjreddie.com/media/files/yolov3.weights)
   - YOLOv3 Weights: `wget https://pjreddie.com/media/files/yolov3.weights`

   Place these files in the appropriate directories as mentioned in the script.

## Usage

Run the script:

```bash
python3 ObjectDetection.py
```

Press 'q' to exit the real-time object detection.

## Customization

- Adjust confidence threshold: Modify the `confidence > 0.5` condition to change the confidence threshold.
- Customize YOLO file paths: Update the paths in the script based on your folder structure.

## Author
- Ali (Real0x0a1)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
