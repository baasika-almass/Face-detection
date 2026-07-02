# 📸 Face Detection

A Python computer vision app that detects faces in 
photos and live webcam feed using OpenCV.

## Features
- Detect faces in any photo
- Real time face detection using webcam
- Draws green rectangles around detected faces
- Counts number of faces found
- Saves output image with faces highlighted
- Press Q or ESC to quit webcam mode

## Installation
```bash
pip install opencv-python
```

## How to Run
```bash
python face_detection.py
```

## How it Works
1. Load image or webcam frame
2. Convert to grayscale
3. Run Haar Cascade classifier to find faces
4. Draw rectangles around detected faces
5. Show result and save output image

## What I Learned
- OpenCV (cv2) library for computer vision
- Haar Cascade pre-trained face detection model
- Converting images to grayscale
- Real time video capture from webcam
- Drawing shapes and text on images

## Tech Used
- Python 3
- OpenCV (cv2)
- NumPy
