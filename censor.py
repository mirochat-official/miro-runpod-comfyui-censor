import ast
import io
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


_SESSION = None
_INPUT_NAME = None
_MODEL_NAMES = None


FALLBACK_NAMES = {
    0: "nipple",
    1: "vagina",
    2: "penis",
    3: "anus",
    4: "pubic hair",
    5: "female face",
    6: "male face",
}


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def _target_classes():
    raw = os.environ.get("CENSOR_CLASSES", "penis,vagina,anus")
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _parse_names_from_metadata(session):
    try:
        meta = session.get_modelmeta()
        custom = meta.custom_metadata_map or {}

        raw = custom.get("names") or custom.get("classes")
        if not raw:
            return FALLBACK_NAMES

        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            return {int(k): str(v) for k, v in parsed.items()}

        if isinstance(parsed, list):
            return {i: str(v) for i, v in enumerate(parsed)}

        return FALLBACK_NAMES
    except Exception:
        return FALLBACK_NAMES


def _get_session():
    global _SESSION
    global _INPUT_NAME
    global _MODEL_NAMES

    if _SESSION is not None:
        return _SESSION, _INPUT_NAME, _MODEL_NAMES

    model_path = os.environ.get(
        "CENSOR_MODEL_PATH",
        "/runpod-volume/models/censor/nsfw-anime-medium-x1280.onnx",
    )

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Censor model not found: {model_path}")

    available = ort.get_available_providers()
    providers = []

    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")

    providers.append("CPUExecutionProvider")

    print(f"miro-censor - loading ONNX model: {model_path}")
    print(f"miro-censor - ONNX providers: {providers}")

    _SESSION = ort.InferenceSession(model_path, providers=providers)
    _INPUT_NAME = _SESSION.get_inputs()[0].name
    _MODEL_NAMES = _parse_names_from_metadata(_SESSION)

    print(f"miro-censor - model names: {_MODEL_NAMES}")

    return _SESSION, _INPUT_NAME, _MODEL_NAMES


def _label_matches(label):
    text = str(label or "").lower().replace("_", " ").replace("-", " ").strip()

    for target in _target_classes():
        if target in text:
            return True

    return False


def _letterbox(image, size):
    height, width = image.shape[:2]

    ratio = min(size / width, size / height)

    new_width = int(round(width * ratio))
    new_height = int(round(height * ratio))

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    pad_w = size - new_width
    pad_h = size - new_height

    left = int(round(pad_w / 2 - 0.1))
    right = int(round(pad_w / 2 + 0.1))
    top = int(round(pad_h / 2 - 0.1))
    bottom = int(round(pad_h / 2 + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    return padded, ratio, left, top


def _xywh_to_xyxy(box):
    x, y, w, h = box
    return [
        x - w / 2,
        y - h / 2,
        x + w / 2,
        y + h / 2,
    ]


def _clip_box(box, width, height):
    x1, y1, x2, y2 = box

    x1 = max(0, min(width, int(round(x1))))
    y1 = max(0, min(height, int(round(y1))))
    x2 = max(0, min(width, int(round(x2))))
    y2 = max(0, min(height, int(round(y2))))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _expand_box(box, width, height, padding_ratio):
    x1, y1, x2, y2 = box

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)

    return _clip_box(
        [
            x1 - pad_x,
            y1 - pad_y,
            x2 + pad_x,
            y2 + pad_y,
        ],
        width,
        height,
    )


def _normalize_prediction_shape(pred):
    pred = np.asarray(pred)

    if pred.ndim == 3:
        pred = pred[0]

    if pred.ndim != 2:
        raise RuntimeError(f"Unexpected prediction shape: {pred.shape}")

    # Ultralytics ONNX often returns [channels, boxes].
    # We need [boxes, channels].
    if pred.shape[0] < pred.shape[1] and pred.shape[0] < 256:
        pred = pred.T

    return pred


def _detect_boxes(image):
    session, input_name, names = _get_session()

    imgsz = _env_int("CENSOR_IMGSZ", 1280)
    conf_threshold = _env_float("CENSOR_CONF", 0.20)
    iou_threshold = _env_float("CENSOR_IOU", 0.45)

    original_height, original_width = image.shape[:2]

    padded, ratio, pad_left, pad_top = _letterbox(image, imgsz)

    input_tensor = padded.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)

    outputs = session.run(None, {input_name: input_tensor})
    pred = _normalize_prediction_shape(outputs[0])

    class_count = len(names)

    if pred.shape[1] < 4 + class_count:
        raise RuntimeError(
            f"Prediction output is too small: shape={pred.shape}, classes={class_count}"
        )

    boxes_xywh = pred[:, 0:4]
    class_scores = pred[:, 4:4 + class_count]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    raw_boxes_for_nms = []
    final_boxes = []
    final_scores = []
    final_labels = []

    for i in range(pred.shape[0]):
        score = float(confidences[i])
        if score < conf_threshold:
            continue

        class_id = int(class_ids[i])
        label = names.get(class_id, str(class_id))

        if not _label_matches(label):
            continue

        x1, y1, x2, y2 = _xywh_to_xyxy(boxes_xywh[i])

        # Remove letterbox padding and scale back to original image size.
        x1 = (x1 - pad_left) / ratio
        y1 = (y1 - pad_top) / ratio
        x2 = (x2 - pad_left) / ratio
        y2 = (y2 - pad_top) / ratio

        clipped = _clip_box([x1, y1, x2, y2], original_width, original_height)
        if clipped is None:
            continue

        bx1, by1, bx2, by2 = clipped

        raw_boxes_for_nms.append([bx1, by1, bx2 - bx1, by2 - by1])
        final_boxes.append(clipped)
        final_scores.append(score)
        final_labels.append(label)

    if not final_boxes:
        return []

    indices = cv2.dnn.NMSBoxes(
        raw_boxes_for_nms,
        final_scores,
        conf_threshold,
        iou_threshold,
    )

    if len(indices) == 0:
        return []

    selected = []

    for item in np.array(indices).flatten().tolist():
        selected.append(
            {
                "box": final_boxes[item],
                "score": final_scores[item],
                "label": final_labels[item],
            }
        )

    return selected


def censor_image_bytes(image_bytes):
    if not _env_bool("CENSOR_ENABLED", True):
        return image_bytes

    if not image_bytes:
        return image_bytes

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    np_image = np.array(image)
    detections = _detect_boxes(np_image)

    if not detections:
        print("miro-censor - no target body parts detected")
        return image_bytes

    draw = ImageDraw.Draw(image)
    padding = _env_float("CENSOR_PADDING", 0.35)

    count = 0

    for det in detections:
        expanded = _expand_box(det["box"], width, height, padding)
        if expanded is None:
            continue

        x1, y1, x2, y2 = expanded

        print(
            "miro-censor - censoring "
            f"label={det['label']}, score={det['score']:.3f}, "
            f"box=({x1},{y1},{x2},{y2})"
        )

        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
        count += 1

    print(f"miro-censor - censored boxes: {count}")

    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
