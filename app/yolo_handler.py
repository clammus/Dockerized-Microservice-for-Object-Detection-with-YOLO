import onnxruntime
import numpy as np
import cv2

# Load ONNX model (runs only once when file is imported)
session = onnxruntime.InferenceSession("models/yolov5s.onnx", providers=["CPUExecutionProvider"])

def detect(image: np.ndarray) -> list:
    """Run YOLOv5 inference on the image and return detected objects."""
    # Resize and normalize the image
    input_image = cv2.resize(image, (640, 640))
    input_image = input_image[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB, HWC → CHW
    input_image = np.ascontiguousarray(input_image, dtype=np.float32) / 255.0
    input_tensor = input_image[np.newaxis, :]

    # Run inference
    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})[0]

    # Parse results
    detections = []
    for det in outputs[0]:
        confidence = det[4]
        if confidence < 0.4:
            continue

        class_scores = det[5:]
        class_id = int(np.argmax(class_scores))
        class_conf = class_scores[class_id]

        if class_conf < 0.4:
            continue

        # Convert box format
        cx, cy, w, h = det[0:4]
        x = int((cx - w / 2) * image.shape[1] / 640)
        y = int((cy - h / 2) * image.shape[0] / 640)
        w = int(w * image.shape[1] / 640)
        h = int(h * image.shape[0] / 640)

        detections.append({
            "label": f"class_{class_id}",
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "confidence": float(confidence)
        })

    return detections
