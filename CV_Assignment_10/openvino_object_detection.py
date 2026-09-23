"""
OpenVINO Object Detection - Webcam / Image
-------------------------------------------
College Assignment: Object Detection using the OpenVINO Toolkit + AI training kit.

Model : YOLOX-Nano (anchor-free, 80-class COCO detector)
Runtime: OpenVINO Runtime 2026.4 (Intel's AI inference toolkit)
Pipeline:
    1. Load a pretrained detector (.onnx) and convert it to OpenVINO's
       native IR format (.xml/.bin) using openvino.convert_model / ov.save_model.
    2. Compile the IR model on the CPU (or "GPU"/"AUTO") plugin with ov.Core().
    3. Preprocess each frame (letterbox resize + pad), run inference,
       decode the anchor-free grid outputs, apply NMS.
    4. Draw bounding boxes + labels + confidence scores on the frame.

Two entry points are provided:
    - run_on_image(path)   -> works anywhere (used for this deliverable's output image)
    - run_on_webcam()      -> works on a machine with a physical camera (device 0)

Usage:
    python openvino_object_detection.py --image input.jpg --output output.jpg
    python openvino_object_detection.py --webcam
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

# --------------------------------------------------------------------------
# 1. COCO class names (80 classes, standard order used by YOLOX / YOLO models)
# --------------------------------------------------------------------------
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

# Deterministic BGR color per class, for consistent, readable boxes
np.random.seed(42)
CLASS_COLORS = np.random.randint(60, 255, size=(len(COCO_CLASSES), 3)).tolist()

INPUT_SIZE = (416, 416)   # YOLOX-Nano's trained input resolution (H, W)
SCORE_THRESHOLD = 0.30
NMS_THRESHOLD = 0.45


# --------------------------------------------------------------------------
# 2. Model loading: ONNX -> OpenVINO IR -> compiled model
# --------------------------------------------------------------------------
def load_openvino_model(onnx_path: str, ir_xml_path: str, device: str = "CPU"):
    """Convert (if needed) and compile a YOLOX-Nano model with OpenVINO Runtime."""
    core = ov.Core()
    ir_xml_path = Path(ir_xml_path)

    if ir_xml_path.exists():
        print(f"[OpenVINO] Loading existing IR model: {ir_xml_path}")
        model = core.read_model(ir_xml_path)
    else:
        print(f"[OpenVINO] Converting ONNX -> OpenVINO IR: {onnx_path}")
        model = ov.convert_model(onnx_path)
        ov.save_model(model, str(ir_xml_path))
        print(f"[OpenVINO] Saved IR to {ir_xml_path}")

    print(f"[OpenVINO] Compiling model for device: {device}")
    compiled_model = core.compile_model(model, device)
    return compiled_model


# --------------------------------------------------------------------------
# 3. Pre / post-processing (standard YOLOX pipeline)
# --------------------------------------------------------------------------
def preprocess(img, input_size=INPUT_SIZE):
    """Letterbox-resize + pad the frame to the model's input size (no normalization)."""
    padded = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    ratio = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized = cv2.resize(
        img,
        (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded[: resized.shape[0], : resized.shape[1]] = resized
    chw = padded.transpose(2, 0, 1)[None].astype(np.float32)  # NCHW, BGR, 0-255
    return chw, ratio


def decode_predictions(outputs, input_size=INPUT_SIZE):
    """Convert the anchor-free grid outputs into (cx, cy, w, h) in input-image space."""
    grids, strides_list = [], []
    for stride in (8, 16, 32):
        h, w = input_size[0] // stride, input_size[1] // stride
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        grids.append(np.stack((xv, yv), 2).reshape(1, -1, 2))
        strides_list.append(np.full((1, h * w, 1), stride))
    grids = np.concatenate(grids, 1)
    strides = np.concatenate(strides_list, 1)

    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * strides
    return outputs


def nms(boxes, scores, thr):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thr)[0] + 1]
    return keep


def postprocess(pred, ratio, score_thr=SCORE_THRESHOLD, nms_thr=NMS_THRESHOLD):
    """Turn raw (1, N, 85) network output into a list of final detections."""
    pred = decode_predictions(pred.copy())[0]  # (N, 85)

    boxes = pred[:, :4]
    obj_conf = pred[:, 4:5]
    cls_conf = pred[:, 5:]
    scores = obj_conf * cls_conf  # (N, 80)

    # cxcywh -> xyxy, back to original image scale
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    xyxy /= ratio

    cls_ids = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_ids)), cls_ids]
    mask = cls_scores > score_thr
    if not mask.any():
        return []

    boxes_f, scores_f, ids_f = xyxy[mask], cls_scores[mask], cls_ids[mask]
    detections = []
    for c in np.unique(ids_f):
        c_mask = ids_f == c
        keep = nms(boxes_f[c_mask], scores_f[c_mask], nms_thr)
        for k in keep:
            b = boxes_f[c_mask][k]
            detections.append((int(c), float(scores_f[c_mask][k]), b))
    return detections


def draw_detections(img, detections):
    for cls_id, score, box in detections:
        x1, y1, x2, y2 = [int(v) for v in box]
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        label = f"{COCO_CLASSES[cls_id]} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def run_inference(compiled_model, frame):
    """Run one frame through OpenVINO and return the list of detections."""
    blob, ratio = preprocess(frame)
    output = compiled_model([blob])[compiled_model.output(0)]
    return postprocess(output, ratio)


# --------------------------------------------------------------------------
# 4a. Static image mode (used for this assignment's "random image" output)
# --------------------------------------------------------------------------
def run_on_image(compiled_model, image_path, output_path):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    t0 = time.time()
    detections = run_inference(compiled_model, frame)
    infer_ms = (time.time() - t0) * 1000

    print(f"[Result] {len(detections)} object(s) detected in {infer_ms:.1f} ms")
    for cls_id, score, box in detections:
        print(f"  - {COCO_CLASSES[cls_id]:<15s} conf={score:.2f}  box={[round(v) for v in box]}")

    out_frame = draw_detections(frame, detections)
    cv2.imwrite(output_path, out_frame)
    print(f"[Saved] {output_path}")
    return detections


# --------------------------------------------------------------------------
# 4b. Webcam mode (run this on your own machine with a camera attached)
# --------------------------------------------------------------------------
def run_on_webcam(compiled_model, camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check the camera index / permissions.")

    print("[Webcam] Press 'q' to quit.")
    prev_time = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = run_inference(compiled_model, frame)
        frame = draw_detections(frame, detections)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("OpenVINO Object Detection - Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------
# 5. CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenVINO Object Detection (YOLOX-Nano)")
    parser.add_argument("--onnx", default="models/yolox_nano.onnx", help="Path to source ONNX model")
    parser.add_argument("--ir", default="models/yolox_nano.xml", help="Path to OpenVINO IR (.xml)")
    parser.add_argument("--device", default="CPU", help="OpenVINO device: CPU, GPU, AUTO")
    parser.add_argument("--image", default=None, help="Run detection on a single image")
    parser.add_argument("--output", default="output.jpg", help="Where to save the annotated image")
    parser.add_argument("--webcam", action="store_true", help="Run detection on a live webcam feed")
    args = parser.parse_args()

    model = load_openvino_model(args.onnx, args.ir, args.device)

    if args.webcam:
        run_on_webcam(model)
    elif args.image:
        run_on_image(model, args.image, args.output)
    else:
        parser.print_help()
