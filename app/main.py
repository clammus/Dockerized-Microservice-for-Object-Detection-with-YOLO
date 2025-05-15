from fastapi import FastAPI, UploadFile, File
from app.utils import decode_image, draw_boxes
from app.yolo_handler import detect
import cv2
import base64
from typing import Optional

app = FastAPI()

@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    label: Optional[str] = None
):
    # Read the uploaded image
    image_bytes = await file.read()
    image = decode_image(image_bytes)

    # Run detection
    boxes = detect(image)

    # Optional label filter
    if label:
        boxes = [box for box in boxes if box["label"] == label]

 

    # Draw boxes on image
    output_img = draw_boxes(image, boxes)

    # Encode image to base64
    _, buffer = cv2.imencode(".jpg", output_img)
    base64_img = base64.b64encode(buffer).decode("utf-8")

    return {
        "image": base64_img,
        "objects": boxes,
        "count": len(boxes)
    }
