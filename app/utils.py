import numpy as np
import cv2

def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode the uploaded image bytes into a NumPy array (OpenCV format)."""
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def draw_boxes(image: np.ndarray, boxes: list) -> np.ndarray:
    """Draw boxes, labels, and confidence scores on the image."""
    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        label = box["label"]
        confidence = box["confidence"]

        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Put label and confidence
        text = f"{label} ({confidence:.2f})"
        cv2.putText(image, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
    return image
