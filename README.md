# YOLOv5 Object Detection API

This project is an object detection microservice based on FastAPI and YOLOv5 (ONNX format). It provides a REST API endpoint for detecting objects in uploaded images.

## Features

Detects objects using YOLOv5s model in ONNX format.
REST API endpoint: `POST /detect?label=<optional>`
Returns:
  Annotated image as a base64 string.
  List of detected objects with label, coordinates, and confidence.
  Count of matched objects (with optional filtering).
Dockerized and ready for deployment.
COCO dataset class labels supported (80 classes).

## Setup & Run

1. Prerequisites
Make sure you have the following installed:

Docker

2. Build and Run with Docker

docker compose up --build
The application will start at:
http://localhost:8000/docs

Using the API (via Swagger UI)
Open the Swagger UI in your browser.

Click on POST /detect.

Click on "Try it out".

Upload an image using the file field.

Optionally enter a label (e.g. "car").

Click "Execute".

View the response: base64 image, bounding box details, label and confidence score.

### Example Queries & Responses
Below are real examples using images added to the project folder. Only partial base64 values are shown for brevity.

***Input: table.jpg***

label = "vase"
{
  "image": "/9j/4AAQSkZJRgABAQAAAQABAAD...<base64 shortened>.../250UVsB//2Q==",
  "objects": [
    {
      "label": "vase",
      "x": 38,
      "y": 42,
      "width": 60,
      "height": 42,
      "confidence": 0.9027367234230042
    }
  ],
  "count": 1
}

label = "chair"
{
  "image": "/9j/4AAQSkZJRgABAQAAAQABAAD...<base64 shortened>.../7c6KK2A//Z",
  "objects": [
    {
      "label": "chair",
      "x": 9,
      "y": 55,
      "width": 22,
      "height": 30,
      "confidence": 0.8980157375335693
    },
    {
      "label": "chair",
      "x": 136,
      "y": 68,
      "width": 23,
      "height": 20,
      "confidence": 0.8844646215438843
    },
    {
      "label": "chair",
      "x": 24,
      "y": 30,
      "width": 20,
      "height": 27,
      "confidence": 0.8821811676025391
    }
  ],
  "count": 3
}

***Input: dog and cat.jpg***

no label provided

{
  "image": "/9j/4AAQSkZJRgABAQAAAQABAAD...<base64 shortened>...dT9nnWtV1nwbbNql885CnBkOaKKAPRaKKKACiiigAooooAKKKKACiiigAooooA/9k=",
  "objects": [
    {
      "label": "cat",
      "x": 78,
      "y": 503,
      "width": 640,
      "height": 641,
      "confidence": 0.9880656599998474
    },
    {
      "label": "dog",
      "x": 793,
      "y": 376,
      "width": 736,
      "height": 522,
      "confidence": 0.8066432476043701
    }
  ],
  "count": 2
}

***Input: teddy bear.jpg***

label = "teddy bear"
{
  "image": "/9j/4AAQSkZJRgABAQAAAQABAAD...<base64 shortened>...exnBUc+9FFADYpkhdlkXPPaopbr7LIGuGwrHg+lFFAFTVG1NL1Lq1TzUxlM10dxdaTeRJPc2QhJhXz+OpoooA/9k=",
  "objects": [
    {
      "label": "teddy bear",
      "x": 117,
      "y": 81,
      "width": 1796,
      "height": 1824,
      "confidence": 0.9221614599227905
    },
    {
      "label": "teddy bear",
      "x": 1641,
      "y": 683,
      "width": 812,
      "height": 427,
      "confidence": 0.8697301745414734
    },
    {
      "label": "teddy bear",
      "x": 415,
      "y": 169,
      "width": 639,
      "height": 400,
      "confidence": 0.7643693685531616
    }
  ],
  "count": 3
}		

## Design Decisions

FastAPI is used for building the REST API due to its speed, simplicity, and async capabilities.
ONNX Runtime enables efficient inference with platform portability.
Docker ensures a reproducible and isolated deployment environment.
cv2.dnn.NMSBoxes is used to reduce overlapping detections using non-maximum suppression (NMS).
A confidence threshold of 0.7 was chosen to balance recall and precision.
Output Formatting: The returned JSON lists each detected object on a separate line instead of inline. This improves readability, especially when dealing with multiple detections with complex metadata (e.g., bounding box coordinates and confidence scores).
Tiny bounding boxes (<20px in width or height) are ignored to reduce noise.

## Docker Notes
The image is based on python:3.10-slim.

OpenCV dependencies like libgl1-mesa-glx and libglib2.0-0 are included for ONNX and cv2 compatibility.

The service is exposed on port 8000.

## Testing
The service was tested with real-world COCO-style images.

For best results, use high-resolution images with clear subjects.

Confidence and NMS thresholds can be adjusted in yolo_handler.py.

## Assumptions
Only one ONNX model (YOLOv5s) is used and resides in the models/ directory.

COCO label classes are assumed for object mapping.

The API is expected to run as a containerized service in isolated environments.

## License

This project is licensed under the MIT License.

© 2025 Selçuk Solmaz