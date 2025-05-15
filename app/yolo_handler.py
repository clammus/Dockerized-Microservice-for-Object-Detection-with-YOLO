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

def postprocess(raw_output, image_shape, input_shape=(640, 640), conf_threshold=0.5, nms_threshold=0.4):
    # Convert SparseTensor or unexpected types to numpy array
    if hasattr(raw_output, "toarray"):
        output = raw_output.toarray()
    elif hasattr(raw_output, "todense"):
        output = np.asarray(raw_output.todense())
    else:
        output = np.array(raw_output)

    predictions = output  # shape: (1, num_detections, 85)
    rects = []
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

        rects.append([x, y, w, h])
        confidences.append(float(confidence))
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(rects, confidences, conf_threshold, nms_threshold)

    boxes = []
    for i in indices:
        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        x, y, w, h = rects[i]
        label = COCO_LABELS[class_ids[i]] if class_ids[i] < len(COCO_LABELS) else f"class_{class_ids[i]}"
        boxes.append({
            "label": label,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "confidence": confidences[i]
        })

    return boxes

def detect(image: np.ndarray) -> list:
    input_tensor = preprocess(image)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    # Safe access: convert SparseTensor to array if needed
    raw_output = outputs[0]
    return postprocess(raw_output, image.shape[:2])
