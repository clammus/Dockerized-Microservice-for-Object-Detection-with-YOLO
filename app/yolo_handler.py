import numpy as np
import onnxruntime as ort
import cv2

# COCO class labels (80 class)
COCO_LABELS = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "dining table", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Load ONNX model
session = ort.InferenceSession("models/yolov5s.onnx")

def preprocess(image: np.ndarray, input_shape=(640, 640)) -> np.ndarray:
    resized = cv2.resize(image, input_shape)
    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    return img

def postprocess(
    raw_output,
    image_shape,
    input_shape=(640, 640),
    conf_threshold=0.7,
    nms_threshold=0.2
    
):
    if hasattr(raw_output, "toarray"):
        output = raw_output.toarray()
    elif hasattr(raw_output, "todense"):
        output = np.asarray(raw_output.todense())
    else:
        output = np.array(raw_output)

    predictions = output  # shape: (1, num_detections, 85)
    boxes_xyxy = []
    confidences = []
    class_ids = []

    for pred in predictions[0]:
        scores = pred[5:]
        class_id = int(np.argmax(scores))
        confidence = scores[class_id]

        if confidence < conf_threshold:
            continue

        x_center, y_center, w, h = pred[0:4]
        x = int((x_center - w / 2) * image_shape[1] / input_shape[0])
        y = int((y_center - h / 2) * image_shape[0] / input_shape[1])
        w = int(w * image_shape[1] / input_shape[0])
        h = int(h * image_shape[0] / input_shape[1])
        x2 = x + w
        y2 = y + h

        # Filter out tiny boxes
        if w < 20 or h < 20:
            continue

        boxes_xyxy.append([x, y, x2, y2])
        confidences.append(float(confidence))
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes_xyxy, confidences, conf_threshold, nms_threshold)

    boxes = []

    # Flatten index list if it's a list of lists/tuples
    flat_indices = [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indices]

    for i in flat_indices:
        conf = confidences[i]

        # Final confidence check
        if conf >= conf_threshold:
            x1, y1, x2, y2 = boxes_xyxy[i]
            w = x2 - x1
            h = y2 - y1
            label = COCO_LABELS[class_ids[i]] if class_ids[i] < len(COCO_LABELS) else f"class_{class_ids[i]}"
            boxes.append({
                "label": label,
                "x": x1,
                "y": y1,
                "width": w,
                "height": h,
                "confidence": conf
            })

    return boxes


def detect(image: np.ndarray) -> list:
    input_tensor = preprocess(image)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    raw_output = outputs[0]

    return postprocess(
        raw_output,
        image.shape[:2],
        conf_threshold=0.7,
        nms_threshold=0.2
    )
