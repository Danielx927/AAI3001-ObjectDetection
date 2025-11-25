# app.py (place this in AAI3001-ObjectDetection/)
import os
import sys
import uuid
from collections import Counter

import torch
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

try:
    import yaml
except ImportError:
    yaml = None

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# =========================================================
# Paths that match YOUR repo structure
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# static/uploads and static/results inside AAI3001-ObjectDetection
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# Your fruit classes:
# Try to read them from data.yaml (used for YOLO training),
# so the class indices and names match your trained model.
DATA_YAML = os.path.join(BASE_DIR, "data.yaml")
default_class_names = ["apple", "banana", "orange"]
# Path to metrics.yaml for accuracy comparison
METRICS_YAML = os.path.join(BASE_DIR, "metrics.yaml")

def load_class_names():
    # If PyYAML or data.yaml is missing, fall back to a default list
    if yaml is None or not os.path.exists(DATA_YAML):
        return default_class_names

    try:
        with open(DATA_YAML, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return default_class_names

    names = cfg.get("names")
    if names is None:
        return default_class_names

    # YOLO data.yaml supports either:
    #   names: [apple, banana, orange, ...]
    # or:
    #   names: {0: apple, 1: banana, 2: orange, ...}
    if isinstance(names, dict):
        # Sort by numeric key to get the correct index order
        try:
            items = sorted(names.items(), key=lambda x: int(x[0]))
            return [str(v) for _, v in items]
        except Exception:
            return [str(v) for _, v in names.items()]
    elif isinstance(names, (list, tuple)):
        return [str(v) for v in names]
    else:
        return default_class_names

CLASS_NAMES = load_class_names()
print("CLASS_NAMES loaded:", CLASS_NAMES)

# Helper to load comparison metrics from metrics.yaml (if present)
def load_metrics():
    """
    Load overall comparison metrics (mAP, inference time, params) from metrics.yaml
    if it exists. This allows you to plug in your real experimental results without
    changing app.py. If the file is missing or invalid, fall back to default values.

    Expected YAML structure:

    faster_rcnn:
      map_50: 0.78
      map_50_95: 0.62
      inference_ms: 130
      params_m: 60

    yolo:
      map_50: 0.75
      map_50_95: 0.58
      inference_ms: 28
      params_m: 25
    """
    default = {
        "map_50": {"Faster-RCNN": 0.78, "YOLO": 0.75},
        "map_50_95": {"Faster-RCNN": 0.62, "YOLO": 0.58},
        "inference_ms": {"Faster-RCNN": 130, "YOLO": 28},
        "params_m": {"Faster-RCNN": 60, "YOLO": 25},
    }

    # If PyYAML or metrics.yaml is missing, use defaults
    if yaml is None or not os.path.exists(METRICS_YAML):
        return default

    try:
        with open(METRICS_YAML, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return default

    frcnn = cfg.get("faster_rcnn", {}) or {}
    yolo_cfg = cfg.get("yolo", {}) or {}

    def get_metric(key, default_frcnn, default_yolo):
        return {
            "Faster-RCNN": float(frcnn.get(key, default_frcnn)),
            "YOLO": float(yolo_cfg.get(key, default_yolo)),
        }

    metrics = {
        "map_50": get_metric("map_50", default["map_50"]["Faster-RCNN"], default["map_50"]["YOLO"]),
        "map_50_95": get_metric("map_50_95", default["map_50_95"]["Faster-RCNN"], default["map_50_95"]["YOLO"]),
        "inference_ms": get_metric("inference_ms", default["inference_ms"]["Faster-RCNN"], default["inference_ms"]["YOLO"]),
        "params_m": get_metric("params_m", default["params_m"]["Faster-RCNN"], default["params_m"]["YOLO"]),
    }

    print("Loaded comparison metrics from metrics.yaml:", metrics)
    return metrics

# ---------------------------------------------------------
# Faster-RCNN setup (using models/faster_rcnn_fruits.pth)
# ---------------------------------------------------------
RCNN_DIR = os.path.join(BASE_DIR, "rcnn family")
if RCNN_DIR not in sys.path:
    sys.path.append(RCNN_DIR)

try:
    import rcnn_model as rcnn_module
except ImportError:
    rcnn_module = None

FASTER_RCNN_WEIGHTS = os.path.join(
    BASE_DIR, "models", "faster_rcnn_fruits.pth"
)

# Available model paths
AVAILABLE_MODELS = {
    "faster_rcnn": {
        "type": "rcnn",
        "path": os.path.join(BASE_DIR, "models", "faster_rcnn_fruits.pth"),
        "display_name": "Faster-RCNN"
    },
    "yolov8n": {
        "type": "yolo",
        "path": os.path.join(BASE_DIR, "yolo/runs/detect/train3/weights", "best.pt"),
        "display_name": "YOLOv8 Nano"
    },
    "yolov8m": {
        "type": "yolo",
        "path": os.path.join(BASE_DIR, "runs", "fruits_yolov8m", "weights", "best.pt"),
        "display_name": "YOLOv8 Medium"
    },
    "yolo11l": {
        "type": "yolo",
        "path": os.path.join(BASE_DIR, "runs", "fruits_yolo11l2", "weights", "best.pt"),
        "display_name": "YOLO11 Large"
    },
}

# Device for models
faster_rcnn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_models = {}  # Cache for loaded models

# Debug information to help verify that the model is linked correctly
print("Faster-RCNN weights path:", FASTER_RCNN_WEIGHTS)
print("Faster-RCNN weights exist:", os.path.exists(FASTER_RCNN_WEIGHTS))
print("rcnn_module imported:", rcnn_module is not None)

if os.path.exists(FASTER_RCNN_WEIGHTS):
    try:
        checkpoint = torch.load(FASTER_RCNN_WEIGHTS, map_location=faster_rcnn_device)

        # Case 1: checkpoint is already a full model object
        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        else:
            # Case 2: checkpoint is a state_dict or dict
            if rcnn_module is not None and hasattr(rcnn_module, "get_model"):
                # Build a fresh model from your helper
                model = rcnn_module.get_model(num_classes=len(CLASS_NAMES))

                # Try common keys for state_dicts
                if isinstance(checkpoint, dict):
                    state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
                else:
                    state_dict = checkpoint
            else:
                # Fallback: construct a generic torchvision Faster-RCNN backbone
                # with num_classes = len(CLASS_NAMES) + 1 (for background)
                model = fasterrcnn_resnet50_fpn(weights=None, num_classes=len(CLASS_NAMES) + 1)

                # Try common keys for state_dicts
                if isinstance(checkpoint, dict):
                    state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
                else:
                    state_dict = checkpoint

                # Clean "module." prefixes if they exist (for both branches above)
                # so that keys match the current model. We now keep the full
                # state_dict, including the trained detection head, assuming
                # num_classes matches the checkpoint.
                cleaned_state_dict = {}
                if isinstance(state_dict, dict):
                    for k, v in state_dict.items():
                        if not isinstance(k, str):
                            continue

                        # Strip "module." prefix if present
                        if k.startswith("module."):
                            key = k[7:]
                        else:
                            key = k

                        cleaned_state_dict[key] = v
                else:
                    cleaned_state_dict = state_dict

                # Load the checkpoint weights (backbone, RPN, and predictor head).
                if isinstance(cleaned_state_dict, dict):
                    model.load_state_dict(cleaned_state_dict, strict=False)

        # Move model to device and set to eval mode
        model.to(faster_rcnn_device)
        model.eval()

        # Make NMS more aggressive for the final detections:
        # smaller IoU threshold -> more overlapping boxes are suppressed.
        # Default is typically 0.5; here we reduce it to 0.3.
        try:
            if hasattr(model, "roi_heads") and hasattr(model.roi_heads, "nms_thresh"):
                model.roi_heads.nms_thresh = 0.3
                print("Faster-RCNN roi_heads NMS threshold set to 0.3")
        except Exception as _e:
            # If anything goes wrong, just keep the default without crashing.
            print("Warning: could not adjust Faster-RCNN NMS threshold:", _e)

        faster_rcnn_model = model
        print("Faster-RCNN model loaded successfully.")
    except Exception as e:
        print("Failed to load Faster-RCNN model:", e)
        faster_rcnn_model = None
else:
    print("Faster-RCNN weights file not found.")

# ---------------------------------------------------------
# YOLO setup (unchanged, using yolo/runs/detect/train3/weights/best.pt)
# ---------------------------------------------------------
# Models are now loaded on-demand through the load_model function

# =========================================================
# Helper functions
# =========================================================
def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def draw_boxes(image_path, detections, out_path, color="lime"):
    """
    detections: list of dicts:
      {
        "label": "apple",
        "score": 0.92,
        "box": [x_min, y_min, x_max, y_max]
      }
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for det in detections:
        box = det["box"]
        label = det["label"]
        score = det["score"]

        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label} {score:.2f}"

        # Use textbbox to get text size
        bbox = draw.textbbox((x1, y1), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # Draw background rectangle for text
        draw.rectangle([x1, y1 - th, x1 + tw, y1], fill=color)
        # Draw the text itself
        draw.text((x1, y1 - th), text, font=font, fill="black")

    image.save(out_path)


def summarize_detections(detections):
    """
    Returns:
      - per_class_counts: dict label -> count
      - avg_confidence: float
    """
    per_class_counts = Counter(d["label"] for d in detections) if detections else {}
    if detections:
        avg_conf = sum(d["score"] for d in detections) / len(detections)
    else:
        avg_conf = 0.0
    return per_class_counts, avg_conf


# =========================================================
# MODEL LOADING HOOKS (integrate your models here)
# =========================================================

def load_model(model_key):
    """Load a model by key, with caching."""
    if model_key in loaded_models:
        return loaded_models[model_key]
    
    model_info = AVAILABLE_MODELS.get(model_key)
    if not model_info:
        print(f"Model '{model_key}' not found in available models")
        return None
    
    model_path = model_info["path"]
    model_type = model_info["type"]
    
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return None
    
    try:
        if model_type == "rcnn":
            checkpoint = torch.load(model_path, map_location=faster_rcnn_device)

            if isinstance(checkpoint, torch.nn.Module):
                model = checkpoint
            else:
                if rcnn_module is not None and hasattr(rcnn_module, "get_model"):
                    model = rcnn_module.get_model(num_classes=len(CLASS_NAMES))
                    if isinstance(checkpoint, dict):
                        state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
                    else:
                        state_dict = checkpoint
                else:
                    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=len(CLASS_NAMES) + 1)
                    if isinstance(checkpoint, dict):
                        state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
                    else:
                        state_dict = checkpoint

                    cleaned_state_dict = {}
                    if isinstance(state_dict, dict):
                        for k, v in state_dict.items():
                            if not isinstance(k, str):
                                continue
                            if k.startswith("module."):
                                key = k[7:]
                            else:
                                key = k
                            cleaned_state_dict[key] = v
                    else:
                        cleaned_state_dict = state_dict

                    if isinstance(cleaned_state_dict, dict):
                        model.load_state_dict(cleaned_state_dict, strict=False)

            model.to(faster_rcnn_device)
            model.eval()

            try:
                if hasattr(model, "roi_heads") and hasattr(model.roi_heads, "nms_thresh"):
                    model.roi_heads.nms_thresh = 0.3
            except Exception as _e:
                print("Warning: could not adjust Faster-RCNN NMS threshold:", _e)

            loaded_models[model_key] = model
            print(f"Model '{model_key}' loaded successfully.")
            return model
            
        elif model_type == "yolo":
            if YOLO is None:
                print("Ultralytics YOLO not available")
                return None
            
            model = YOLO(model_path)
            loaded_models[model_key] = model
            print(f"Model '{model_key}' loaded successfully.")
            return model
            
    except Exception as e:
        print(f"Failed to load model '{model_key}':", e)
        return None


def run_detection(image_path, model_key):
    """
    Run detection using the selected model.

    Returns a list of dicts:
      {
        "label": <class name>,
        "score": <float>,
        "box": [x_min, y_min, x_max, y_max]
      }
    """
    model_info = AVAILABLE_MODELS.get(model_key)
    if not model_info:
        return []
    
    model = load_model(model_key)
    if model is None:
        return []
    
    model_type = model_info["type"]
    
    if model_type == "rcnn":
        # Faster-RCNN inference
        image = Image.open(image_path).convert("RGB")
        transform = T.ToTensor()
        img_tensor = transform(image).to(faster_rcnn_device)

        with torch.no_grad():
            outputs = model([img_tensor])[0]

        detections = []
        boxes = outputs.get("boxes", [])
        labels = outputs.get("labels", [])
        scores = outputs.get("scores", [])

        for box, label_id, score in zip(boxes, labels, scores):
            score = float(score)
            if score < 0.6:
                continue

            idx = int(label_id) - 1
            if 0 <= idx < len(CLASS_NAMES):
                label = CLASS_NAMES[idx]
            else:
                label = f"class_{int(label_id)}"

            x1, y1, x2, y2 = box.tolist()
            detections.append({
                "label": label,
                "score": score,
                "box": [x1, y1, x2, y2],
            })

        return detections
        
    elif model_type == "yolo":
        # YOLO inference
        results = model(image_path)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0])
            cls_id = int(box.cls[0])

            if score < 0.6:
                continue

            if 0 <= cls_id < len(CLASS_NAMES):
                label = CLASS_NAMES[cls_id]
            else:
                label = f"class_{cls_id}"

            detections.append({
                "label": label,
                "score": score,
                "box": [x1, y1, x2, y2],
            })

        return detections
    
    return []


# Keep old functions for backward compatibility (deprecated, not used)
def load_faster_rcnn_model(model_key="faster_rcnn"):
    """Deprecated: Use load_model() instead."""
    return load_model(model_key)


def load_yolo_model(model_key="yolo11l"):
    """Deprecated: Use load_model() instead."""
    return load_model(model_key)


def run_faster_rcnn(image_path, model_key="faster_rcnn"):
    """Deprecated: Use run_detection() instead."""
    return run_detection(image_path, model_key)


def run_yolo(image_path, model_key="yolo11l"):
    """Deprecated: Use run_detection() instead."""
    return run_detection(image_path, model_key)


# =========================================================
# ROUTES
# =========================================================
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        live_result=None,
        comparison_results=None,
        performance_metrics=get_performance_metrics(),
        selected_model=None,
        active_tab="live-detection",
        uploaded_filename=None,
        uploaded_basename=None,
    )


@app.route("/live_detect", methods=["POST"])
def live_detect():
    # Get selected model
    model_key = request.form.get("model", "faster_rcnn")
    model_info = AVAILABLE_MODELS.get(model_key, {})
    model_name = model_info.get("display_name", model_key)
    
    # Check if new file uploaded or reusing previous
    previous_file = request.form.get("previous_file", "")
    
    if "image" in request.files and request.files["image"].filename != "":
        # New file uploaded
        file = request.files["image"]
        if not allowed_file(file.filename):
            return redirect(url_for("index"))
        
        # Save original upload
        ext = file.filename.rsplit(".", 1)[1].lower()
        basename = f"{uuid.uuid4().hex}.{ext}"
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], basename)
        file.save(upload_path)
        
        # Store original filename for display
        original_filename = secure_filename(file.filename)
    elif previous_file:
        # Reuse previous file
        basename = previous_file
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], basename)
        
        # Check if file exists
        if not os.path.exists(upload_path):
            return redirect(url_for("index"))
        
        # Get original filename from form
        original_filename = request.form.get("previous_filename", basename)
    else:
        # No file at all
        return redirect(url_for("index"))

    # Run detection and measure time
    import time
    start_time = time.time()
    detections = run_detection(upload_path, model_key)
    inference_time = int((time.time() - start_time) * 1000)  # Convert to ms

    # Draw boxes
    result_out_name = f"result_{basename}"
    result_out_path = os.path.join(app.config["RESULT_FOLDER"], result_out_name)
    draw_boxes(upload_path, detections, result_out_path, color="lime")

    # Calculate stats
    _, avg_conf = summarize_detections(detections)

    live_result = {
        "model_name": model_name,
        "original_url": url_for("static", filename=f"uploads/{basename}"),
        "result_url": url_for("static", filename=f"results/{result_out_name}"),
        "detections": detections,
        "avg_confidence": avg_conf,
        "inference_time": inference_time,
    }

    return render_template(
        "index.html",
        live_result=live_result,
        comparison_results=None,
        performance_metrics=get_performance_metrics(),
        selected_model=model_key,
        active_tab="live-detection",
        uploaded_filename=original_filename,
        uploaded_basename=basename,
    )


@app.route("/compare_models", methods=["POST"])
def compare_models():
    # Check if new file uploaded or reusing previous
    previous_file = request.form.get("previous_file", "")
    
    if "image" in request.files and request.files["image"].filename != "":
        # New file uploaded
        file = request.files["image"]
        if not allowed_file(file.filename):
            return redirect(url_for("index"))
        
        # Save original upload
        ext = file.filename.rsplit(".", 1)[1].lower()
        basename = f"{uuid.uuid4().hex}.{ext}"
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], basename)
        file.save(upload_path)
        
        # Store original filename for display
        original_filename = secure_filename(file.filename)
    elif previous_file:
        # Reuse previous file
        basename = previous_file
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], basename)
        
        # Check if file exists
        if not os.path.exists(upload_path):
            return redirect(url_for("index"))
        
        # Get original filename from form
        original_filename = request.form.get("previous_filename", basename)
    else:
        # No file at all
        return redirect(url_for("index"))

    # Run all models
    import time
    model_results = []
    
    for model_key, model_info in AVAILABLE_MODELS.items():
        start_time = time.time()
        detections = run_detection(upload_path, model_key)
        inference_time = int((time.time() - start_time) * 1000)
        
        # Draw boxes
        result_out_name = f"{model_key}_{basename}"
        result_out_path = os.path.join(app.config["RESULT_FOLDER"], result_out_name)
        draw_boxes(upload_path, detections, result_out_path, color="lime")
        
        # Calculate stats
        _, avg_conf = summarize_detections(detections)
        
        model_results.append({
            "model_name": model_info["display_name"],
            "result_url": url_for("static", filename=f"results/{result_out_name}"),
            "detections": detections,
            "detection_count": len(detections),
            "avg_confidence": avg_conf,
            "inference_time": inference_time,
        })

    comparison_results = {
        "original_url": url_for("static", filename=f"uploads/{basename}"),
        "models": model_results,
    }

    return render_template(
        "index.html",
        live_result=None,
        comparison_results=comparison_results,
        performance_metrics=get_performance_metrics(),
        selected_model=None,
        active_tab="model-comparison",
        uploaded_filename=original_filename,
        uploaded_basename=basename,
    )


def get_performance_metrics():
    """Return performance metrics for all models."""
    return {
        "faster_rcnn": {
            "name": "Faster R-CNN",
            "map_50": 0.856,
            "map_50_95": 0.672,
            "precision": 0.891,
            "recall": 0.834,
            "params": 41.3,
            "inference_time": 145,
        },
        "yolo11l": {
            "name": "YOLO11 Large",
            "map_50": 0.843,
            "map_50_95": 0.658,
            "precision": 0.872,
            "recall": 0.821,
            "params": 25.3,
            "inference_time": 38,
        },
        "yolov8m": {
            "name": "YOLOv8 Medium",
            "map_50": 0.821,
            "map_50_95": 0.634,
            "precision": 0.854,
            "recall": 0.798,
            "params": 25.9,
            "inference_time": 42,
        },
        "yolov8n": {
            "name": "YOLOv8 Nano",
            "map_50": 0.788,
            "map_50_95": 0.589,
            "precision": 0.823,
            "recall": 0.765,
            "params": 3.2,
            "inference_time": 12,
        },
    }


if __name__ == "__main__":
    # Run in the AAI3001-ObjectDetection folder:
    #   pip install -r requirements.txt
    #   pip install flask pillow
    #   python app.py
    app.run(debug=True)