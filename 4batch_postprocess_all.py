#!/usr/bin/env python3
"""
Universal Batch Post-processing Script for 4-batch CNN Tool

This script automatically processes all model outputs in the 4batch-example directory:
1. Scans all model directories
2. Reads parameters.json for each model
3. Visualizes all output tensors with original images
4. Generates detection/segmentation/pose visualizations
5. Generates summary report

Usage:
    python 4batch_postprocess_all.py --base-dir . --image-dir ../4batch_input/image --heads-dir ./heads

    If --base-dir is not specified, it defaults to current directory.
    Each model's output will be saved to: <model_dir>/postprocess/
    e.g., atss_r50 results -> ./atss_r50/postprocess/

Parameters:
    --base-dir       Base directory containing model output folders (default: current directory)
    --image-dir      Directory containing original input images (required)
    --heads-dir      Directory containing head.onnx files (required)
    --batch-size     Batch size for visualization (default: 4)
    --models         Specific model names to process (default: all models)
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
from collections import defaultdict
import traceback

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not installed. Detection head processing will be skipped.")
    print("Install with: pip install onnxruntime")


# ==========================================
# Configuration
# ==========================================

# Model type patterns for classification
MODEL_PATTERNS = {
    "atss": ("ATSS", "detection", "atss_r50-head.onnx"),
    "paa": ("PAA", "detection", "paa_r50-head.onnx"),
    "autoassign": ("AutoAssign", "detection", "autoassign_r50-head.onnx"),
    "centernet": ("CenterNet", "detection", "centernet_update_r50-head.onnx"),
    "lad": ("LAD", "detection", "lad_r50-head.onnx"),
    "retinanet": ("RetinaNet", "detection", None),
    "yolo": ("YOLO", "yolo", "yolov8n-head.onnx"),
    "yolov5": ("YOLOv5", "yolo", "yolov5n-seg-head.onnx"),
    "ffnet": ("FFNet", "segmentation", None),
    "solo": ("SOLO", "instance_segmentation", "decoupled_solo_r50-head.onnx"),
    "mediapipe": ("MediaPipe", "mediapipe", None),
    "pose": ("PoseResNet", "pose", None),
    "resnet": ("ResNet", "backbone", None),
    "mobilenet": ("MobileNet", "backbone", None),
}

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Colors for visualization (BGR format)
COLORS = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 255, 0), (0, 128, 255), (255, 128, 0), (255, 0, 128)
]

# Segmentation colormap (20 classes for Cityscapes-style)
SEGMENTATION_COLORS = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 0)
]


# ==========================================
# Utility Functions
# ==========================================

def detect_model_type(directory_name):
    """Detect model type from directory name."""
    dir_lower = directory_name.lower()
    for pattern, (name, model_type, head_file) in MODEL_PATTERNS.items():
        if pattern in dir_lower:
            return pattern, name, model_type, head_file
    return "unknown", "Unknown", "backbone", None


def load_json_params(json_path):
    """Load parameters from JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def load_binary_file(file_path, dtype=np.int32, shape=None, fracbits=None):
    """Load binary file and convert to numpy array."""
    try:
        data = np.fromfile(file_path, dtype=dtype)

        if shape:
            expected_size = np.prod(shape)
            if data.size != expected_size:
                if data.size > expected_size:
                    data = data[:expected_size]
                else:
                    data = np.pad(data, (0, expected_size - data.size), mode='constant')
            data = data.reshape(shape)

        if fracbits is not None:
            data = data.astype(np.float32) / (2 ** fracbits)

        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def find_parameter_json(model_dir):
    """Find the parameters JSON file in model directory."""
    json_files = list(Path(model_dir).glob("*parameters.json"))
    if json_files:
        return str(json_files[0])
    return None


def get_output_info(params_data):
    """Extract output tensor information from parameters JSON."""
    outputs = []
    params = params_data.get("parameters", [])

    for param in params:
        if param.get("io_type") == "output":
            outputs.append({
                "name": param.get("parameter", ["unknown"])[0],
                "shape": param.get("shape", []),
                "dtype": param.get("dtype", "int32"),
                "fracbits": param.get("fracbits"),
                "value": param.get("value"),
            })

    return outputs


def load_original_images(image_dir):
    """Load original images from directory."""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"Warning: Image directory not found: {image_dir}")
        return []

    # Load all jpg/png files, sorted
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))

    images = []
    for img_path in image_files[:4]:  # Max 4 images
        try:
            img = Image.open(img_path).convert("RGB")
            images.append((img_path.name, img))
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")

    return images


def get_head_onnx_path(model_name, model_type, heads_dir):
    """Get the path to the head.onnx file for this model."""
    if heads_dir is None:
        return None

    heads_dir = Path(heads_dir)
    if not heads_dir.exists():
        return None

    # Try to find matching head file
    pattern, full_name, _, head_file = detect_model_type(model_name)

    if head_file:
        head_path = heads_dir / head_file
        if head_path.exists():
            return str(head_path)

    # Try pattern matching
    for head_file in heads_dir.glob("*.onnx"):
        if pattern in head_file.stem.lower():
            return str(head_file)

    return None


# ==========================================
# Visualization Functions
# ==========================================

def visualize_tensor(data, output_path, title=""):
    """Visualize a tensor as an image."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    if data is None:
        print(f"Warning: Cannot visualize None data for {output_path}")
        return

    # Remove batch dimension if present
    if data.ndim == 4:
        data = data[0]  # First batch

    # Handle different tensor shapes
    if data.ndim == 3:  # [C, H, W]
        C, H, W = data.shape

        if C == 1:
            # Single channel - grayscale
            img_data = data[0]
        elif C == 3:
            # RGB - convert to image
            img_data = np.transpose(data, (1, 2, 0))  # [H, W, C]
        else:
            # Multiple channels - take mean
            img_data = np.mean(data, axis=0)

        # Normalize to 0-255
        min_val = img_data.min()
        max_val = img_data.max()
        if max_val > min_val:
            img_data = (img_data - min_val) / (max_val - min_val) * 255
        else:
            img_data = np.zeros_like(img_data)

        img_data = img_data.astype(np.uint8)

        # Convert to PIL Image
        if C == 3:
            img = Image.fromarray(img_data, mode='RGB')
        else:
            img = Image.fromarray(img_data, mode='L')

        img.save(output_path)
        print(f"Saved: {output_path}")

    elif data.ndim == 2:  # [H, W]
        img_data = data
        min_val = img_data.min()
        max_val = img_data.max()
        if max_val > min_val:
            img_data = (img_data - min_val) / (max_val - min_val) * 255
        else:
            img_data = np.zeros_like(img_data)

        img = Image.fromarray(img_data.astype(np.uint8), mode='L')
        img.save(output_path)
        print(f"Saved: {output_path}")

    elif data.ndim == 1:  # [N] - vector
        # Plot as histogram
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data[:1000])  # Show first 1000 elements
        ax.set_title(f"{title} (first 1000 elements)")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
        print(f"Saved: {output_path}")


def visualize_tensor_grid(data, output_path, title="", grid_size=(4, 4)):
    """Visualize tensor channels in a grid."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    if data is None:
        return

    if data.ndim == 4:
        data = data[0]

    if data.ndim != 3:
        return

    C, H, W = data.shape
    rows, cols = grid_size
    num_channels = min(C, rows * cols)

    # Create grid image
    cell_w, cell_h = 80, 80
    grid_img = Image.new('RGB', (cols * cell_w, rows * cell_h), color=(50, 50, 50))
    draw = ImageDraw.Draw(grid_img)

    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except:
        font = ImageFont.load_default()

    for idx in range(num_channels):
        row = idx // cols
        col = idx % cols

        channel = data[idx]

        # Normalize
        min_val = channel.min()
        max_val = channel.max()
        if max_val > min_val:
            channel = (channel - min_val) / (max_val - min_val)
        else:
            channel = np.zeros_like(channel)

        # Resize to cell size
        channel_img = Image.fromarray((channel * 255).astype(np.uint8), mode='L')
        channel_img = channel_img.resize((cell_w - 2, cell_h - 20), Image.Resampling.NEAREST)

        # Convert to RGB
        channel_rgb = Image.merge('RGB', [channel_img] * 3)

        # Paste into grid
        x, y = col * cell_w + 1, row * cell_h + 1
        grid_img.paste(channel_rgb, (x, y))

        # Add channel number
        draw.text((x + 2, y + 2), f"Ch{idx}", fill=(255, 255, 0), font=font)

    grid_img.save(output_path)
    print(f"Saved grid: {output_path}")


def overlay_segmentation(original_img, segmentation, output_path, alpha=0.5):
    """Overlay segmentation mask on original image."""
    # Resize segmentation to match original image
    orig_w, orig_h = original_img.size
    seg_h, seg_w = segmentation.shape[-2:]

    # Get segmentation class map (argmax across channels if C > 1)
    if segmentation.ndim == 3:
        seg_map = np.argmax(segmentation, axis=0)
    else:
        seg_map = segmentation

    # Create colored mask
    colored_mask = np.zeros((seg_h, seg_w, 3), dtype=np.uint8)
    for class_id in np.unique(seg_map):
        if class_id < len(SEGMENTATION_COLORS):
            colored_mask[seg_map == class_id] = SEGMENTATION_COLORS[class_id]

    # Resize to original size
    mask_resized = cv2.resize(colored_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Convert original to numpy
    orig_np = np.array(original_img)

    # Blend
    blended = cv2.addWeighted(orig_np, 1 - alpha, mask_resized, alpha, 0)

    # Save
    result = Image.fromarray(blended)
    result.save(output_path)
    print(f"Saved segmentation overlay: {output_path}")


def create_side_by_side_comparison(original_img, tensor_img, output_path, labels=("Original", "Output")):
    """Create side-by-side comparison image."""
    # Resize tensor image to match original height
    orig_w, orig_h = original_img.size

    # Convert tensor to RGB if needed
    if isinstance(tensor_img, np.ndarray):
        if tensor_img.ndim == 2:
            tensor_img = np.stack([tensor_img] * 3, axis=-1)
        elif tensor_img.ndim == 3 and tensor_img.shape[0] == 3:
            tensor_img = np.transpose(tensor_img, (1, 2, 0))

        # Normalize
        tensor_img = (tensor_img - tensor_img.min()) / (tensor_img.max() - tensor_img.min() + 1e-8) * 255
        tensor_img = tensor_img.astype(np.uint8)
        tensor_pil = Image.fromarray(tensor_img)
    else:
        tensor_pil = tensor_img

    # Resize
    tensor_resized = tensor_pil.resize((orig_w, orig_h))

    # Create side-by-side image
    combined = Image.new('RGB', (orig_w * 2, orig_h))
    combined.paste(original_img, (0, 0))
    combined.paste(tensor_resized, (orig_w, 0))

    # Add labels
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    draw.text((10, 10), labels[0], fill=(255, 255, 255), font=font)
    draw.text((orig_w + 10, 10), labels[1], fill=(255, 255, 255), font=font)

    combined.save(output_path)
    print(f"Saved comparison: {output_path}")


# ==========================================
# Detection Processing with ONNX Head
# ==========================================

def run_onnx_detection(onnx_path, feature_maps, conf_threshold=0.3):
    """Run ONNX detection head on FPN feature maps."""
    if not ONNX_AVAILABLE:
        print("Warning: onnxruntime not available, skipping ONNX detection")
        return None

    try:
        import onnxruntime as ort
    except ImportError:
        print("Warning: onnxruntime not installed, skipping ONNX detection")
        return None

    print(f"Loading ONNX model: {onnx_path}")

    try:
        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        # Get input names
        input_names = [i.name for i in session.get_inputs()]
        print(f"ONNX inputs: {input_names}")

        # Prepare inputs - try different ordering strategies
        onnx_inputs = {}

        # Strategy 1: Direct name match (for YOLO and other models)
        for input_name in input_names:
            if input_name in feature_maps:
                onnx_inputs[input_name] = feature_maps[input_name]
                print(f"  Input '{input_name}': {onnx_inputs[input_name].shape}")

        # Strategy 2: Match by level name pattern (for FPN models like ATSS)
        if len(onnx_inputs) < len(input_names):
            level_map = {'p2': 2, 'p3': 3, 'p4': 4, 'p5': 5, 'p6': 6}
            for input_name in input_names:
                if input_name in onnx_inputs:
                    continue  # Already matched
                matched_level = None
                for level in level_map:
                    if level in input_name.lower():
                        matched_level = level
                        break

                if matched_level and matched_level in feature_maps:
                    onnx_inputs[input_name] = feature_maps[matched_level].astype(np.float32)
                    print(f"  Input '{input_name}' -> {matched_level}: {onnx_inputs[input_name].shape}")

        # Strategy 3: Positional ordering (fallback)
        if len(onnx_inputs) < len(input_names):
            feature_map_list = list(feature_maps.values())
            for i, name in enumerate(input_names):
                if name not in onnx_inputs and i < len(feature_map_list):
                    onnx_inputs[name] = feature_map_list[i].astype(np.float32)
                    print(f"  Input '{name}': {onnx_inputs[name].shape} (positional)")

        if len(onnx_inputs) != len(input_names):
            print(f"Warning: Only provided {len(onnx_inputs)}/{len(input_names)} inputs")

        # Run inference
        print("Running ONNX inference...")
        outputs = session.run(None, onnx_inputs)

        print(f"ONNX outputs: {[o.shape for o in outputs]}")

        # Parse outputs - common formats:
        # Format 1: [dets, labels] where dets=[N,5], labels=[N]
        # Format 2: single output [N,6] with [x1,y1,x2,y2,score,class]
        # Format 3 (YOLO): single output [1, 84, 8400] = [1, 4+80, 8400]

        detections = []

        # Check for YOLO format: [1, num_classes+4, num_anchors]
        if len(outputs) == 1 and outputs[0].ndim == 3 and outputs[0].shape[1] >= 4:
            data = outputs[0]  # [1, 84, 8400]
            data = data[0]  # [84, 8400]
            
            num_classes = data.shape[0] - 4
            num_anchors = data.shape[1]
            
            # Transpose to [8400, 84]
            data = data.T  # [8400, 84]
            
            print(f"  Parsing YOLO output: {num_anchors} anchors, {num_classes} classes")
            
            # Extract boxes and scores
            boxes = data[:, :4]  # [8400, 4] - xyxy or xywh
            scores = data[:, 4:]  # [8400, 80] - class scores
            
            # Get max score and class for each anchor
            max_scores = np.max(scores, axis=1)  # [8400]
            class_ids = np.argmax(scores, axis=1)  # [8400]
            
            # Filter by confidence threshold
            for i in range(num_anchors):
                score = float(max_scores[i])
                if score >= conf_threshold:
                    # YOLO typically outputs [x_center, y_center, width, height]
                    # Convert to [x1, y1, x2, y2]
                    x_c, y_c, w, h = boxes[i]
                    x1 = float(x_c - w / 2)
                    y1 = float(y_c - h / 2)
                    x2 = float(x_c + w / 2)
                    y2 = float(y_c + h / 2)
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'score': score,
                        'label': int(class_ids[i])
                    })
            
            print(f"  Got {len(detections)} detections with score >= {conf_threshold}")
            
        elif len(outputs) >= 2:
            # Format 1: separate boxes and labels
            boxes_data = outputs[0]
            labels_data = outputs[1]

            # Remove batch dimension if present
            if boxes_data.ndim == 3:
                boxes_data = boxes_data[0]
            if labels_data.ndim == 2 or labels_data.ndim == 3:
                labels_data = labels_data.flatten()

            print(f"Parsing {len(boxes_data)} boxes...")

            for i in range(len(boxes_data)):
                if boxes_data.ndim == 2:
                    box = boxes_data[i]
                    score = float(box[4]) if box.shape[0] >= 5 else 0.5
                else:
                    continue

                if score >= conf_threshold:
                    detections.append({
                        'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        'score': score,
                        'label': int(labels_data[i]) if i < len(labels_data) else 0
                    })

        elif len(outputs) == 1:
            # Format 2: single output
            data = outputs[0]
            if data.ndim == 2:
                for row in data:
                    if len(row) >= 6:
                        score = float(row[4])
                        if score >= conf_threshold:
                            detections.append({
                                'bbox': [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
                                'score': score,
                                'label': int(row[5])
                            })

        print(f"Got {len(detections)} detections with score >= {conf_threshold}")
        return detections

    except Exception as e:
        print(f"Error running ONNX detection: {e}")
        import traceback
        traceback.print_exc()
        return None


def draw_detections_on_image(img_pil, detections, conf_threshold=0.3):
    """Draw detection boxes on PIL image."""
    import cv2

    # Convert PIL to CV2
    img_np = np.array(img_pil)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    height, width = img_np.shape[:2]

    for det in detections:
        bbox = det.get('bbox', [])
        score = det.get('score', 0)
        label_id = det.get('label', 0)

        if score < conf_threshold:
            continue

        x1, y1, x2, y2 = [int(x) for x in bbox]

        # Clip to image bounds
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue

        # Get color and label
        color = COLORS[label_id % len(COLORS)]
        class_name = COCO_CLASSES[label_id] if label_id < len(COCO_CLASSES) else f"class_{label_id}"
        label_text = f"{class_name}: {score:.2f}"

        # Draw bounding box
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_np, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)

        # Draw label text
        cv2.putText(img_np, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Convert back to PIL
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_np)


def load_fpn_feature_maps(model_dir, batch_idx=0):
    """Load FPN feature maps for detection models."""
    # Find parameter JSON to get tensor info
    json_path = find_parameter_json(model_dir)
    if not json_path:
        return {}

    params_data = load_json_params(json_path)
    if not params_data:
        return {}

    # Get output info
    outputs = get_output_info(params_data)

    # Map output shapes to FPN levels based on spatial size
    feature_maps = {}

    for output_info in outputs:
        shape = output_info.get("shape", [])
        value_file = output_info.get("value", "")
        fracbits = output_info.get("fracbits", 29)

        if not shape or len(shape) != 4:
            continue

        # Shape is [B, C, H, W]
        batch_size, channels, h, w = shape

        # Determine FPN level based on spatial size
        level = None
        if w == 100 and h == 100:
            level = 'p2'
        elif w == 25 and h == 25:
            level = 'p3'
        elif w == 50 and h == 50:
            level = 'p4'
        elif w == 7 and h == 7:
            level = 'p5'
        elif w == 13 and h == 13:
            level = 'p6'

        if not level:
            continue

        # Find the tensor file
        tensor_file = None
        if value_file and os.path.exists(os.path.join(model_dir, value_file)):
            tensor_file = os.path.join(model_dir, value_file)
        else:
            # Try to find matching tensor file by looking for files with the right size
            expected_size = batch_size * channels * h * w * 4  # int32 = 4 bytes
            for tensor_path in Path(model_dir).glob("*.bin"):
                if 'tensor' in tensor_path.name.lower():
                    file_size = tensor_path.stat().st_size
                    if abs(file_size - expected_size) < 1000:
                        tensor_file = str(tensor_path)
                        break

        if not tensor_file:
            continue

        # Load tensor
        try:
            data = load_binary_file(tensor_file, dtype=np.int32, shape=tuple(shape), fracbits=fracbits)
            if data is not None and data.ndim == 4:
                # Extract specific batch and add batch dimension for ONNX [1, C, H, W]
                batch_data = data[batch_idx:batch_idx+1]
                feature_maps[level] = batch_data
                print(f"  Loaded {level}: {batch_data.shape} from {os.path.basename(tensor_file)}")
        except Exception as e:
            print(f"  Error loading {level}: {e}")

    return feature_maps


def load_yolo_feature_maps(model_dir, batch_idx=0):
    """Load YOLO feature maps for detection models."""
    feature_maps = {}
    
    # YOLOv8 outputs feature maps with specific naming
    yolo_outputs = [
        ("_model_22_cv2_0_cv2_0_1_act_Mul_output_0.bin", [4, 64, 80, 80], 25, "/model.22/cv2.0/cv2.0.1/act/Mul_output_0"),
        ("_model_22_cv3_0_cv3_0_1_act_Mul_output_0.bin", [4, 80, 80, 80], 24, "/model.22/cv3.0/cv3.0.1/act/Mul_output_0"),
        ("_model_22_cv2_1_cv2_1_1_act_Mul_output_0.bin", [4, 64, 40, 40], 25, "/model.22/cv2.1/cv2.1.1/act/Mul_output_0"),
        ("_model_22_cv3_1_cv3_1_1_act_Mul_output_0.bin", [4, 80, 40, 40], 23, "/model.22/cv3.1/cv3.1.1/act/Mul_output_0"),
        ("_model_22_cv2_2_cv2_2_1_act_Mul_output_0.bin", [4, 64, 20, 20], 25, "/model.22/cv2.2/cv2.2.1/act/Mul_output_0"),
        ("_model_22_cv3_2_cv3_2_1_act_Mul_output_0.bin", [4, 80, 20, 20], 24, "/model.22/cv3.2/cv3.2.1/act/Mul_output_0"),
    ]
    
    for file_name, shape, fracbits, onnx_name in yolo_outputs:
        file_path = os.path.join(model_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"  Warning: {file_name} not found")
            continue
        
        try:
            data = load_binary_file(file_path, dtype=np.int32, shape=shape, fracbits=fracbits)
            if data is not None and data.ndim == 4:
                # Extract specific batch [1, C, H, W]
                batch_data = data[batch_idx:batch_idx+1]
                feature_maps[onnx_name] = batch_data.astype(np.float32)
                scale = f"{shape[2]}x{shape[3]}"
                print(f"  Loaded {file_name}: {batch_data.shape} ({scale})")
        except Exception as e:
            print(f"  Error loading {file_name}: {e}")
    
    return feature_maps

def process_detection_with_boxes(model_dir, head_onnx_path, original_images, output_dir, conf_threshold=0.3, model_type="detection"):
    """Process detection model with ONNX head to draw boxes on images."""
    if not head_onnx_path or not ONNX_AVAILABLE:
        print("Skipping detection boxes (no ONNX head or onnxruntime not available)")
        return []

    print("\n" + "-"*50)
    print(f"Running ONNX detection head to generate boxes (model_type={model_type})...")
    print("-"*50)

    detection_results = []

    for batch_idx, (img_name, img_pil) in enumerate(original_images):
        print(f"\n[{batch_idx+1}/{len(original_images)}] Processing {img_name}")

        # Load feature maps based on model type
        if "yolo" in model_type.lower():
            feature_maps = load_yolo_feature_maps(model_dir, batch_idx)
        else:
            feature_maps = load_fpn_feature_maps(model_dir, batch_idx)

        if len(feature_maps) < 1:
            print(f"  Warning: Only loaded {len(feature_maps)} feature maps, skipping")
            continue

        print(f"  Loaded {len(feature_maps)} feature map scales: {list(feature_maps.keys())}")

        # Run ONNX detection
        detections = run_onnx_detection(head_onnx_path, feature_maps, conf_threshold)

        if detections is None:
            print("  ONNX detection failed, using fallback...")
            continue

        # Draw detections on image
        result_img = draw_detections_on_image(img_pil, detections, conf_threshold)

        # Save result
        output_path = os.path.join(output_dir, f"detection_boxes_{img_name}")
        result_img.save(output_path)
        print(f"  Saved: {output_path} ({len(detections)} detections)")

        detection_results.append({
            'image': img_name,
            'detections': detections,
            'output_path': output_path
        })

    return detection_results


# ==========================================
# Model Processing
# ==========================================

def process_model(model_dir, output_base_dir, image_dir, heads_dir, batch_size=4):
    """Process a single model directory."""
    model_name = os.path.basename(model_dir)
    pattern, full_name, model_type, head_file = detect_model_type(model_name)

    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"Type: {full_name} ({model_type})")
    if head_file:
        print(f"Head ONNX: {head_file}")
    print(f"{'='*60}")

    # Find parameters JSON
    json_path = find_parameter_json(model_dir)
    if not json_path:
        print(f"Warning: No parameters.json found in {model_name}")
        return None

    # Load parameters
    params_data = load_json_params(json_path)
    if not params_data:
        return None

    # Get output information
    outputs = get_output_info(params_data)
    if not outputs:
        print(f"Warning: No outputs found in parameters")
        return None

    print(f"Found {len(outputs)} output tensors")

    # Load original images
    original_images = load_original_images(image_dir)
    print(f"Loaded {len(original_images)} original images")

    # Check for head ONNX
    head_onnx_path = get_head_onnx_path(model_name, model_type, heads_dir)
    if head_onnx_path:
        print(f"Found head ONNX: {head_onnx_path}")
    elif model_type == "detection":
        print(f"Warning: No head ONNX found for detection model")

    # Create output directory for this model
    model_output_dir = output_base_dir
    os.makedirs(model_output_dir, exist_ok=True)

    results = {
        "model_name": model_name,
        "model_type": model_type,
        "model_full_name": full_name,
        "num_outputs": len(outputs),
        "outputs": []
    }

    # Process detection models with ONNX head to draw boxes
    if (model_type in ["detection", "yolo"]) and head_onnx_path and ONNX_AVAILABLE:
        print("\n*** Running detection with ONNX head to draw boxes ***")
        detection_results = process_detection_with_boxes(
            model_dir, head_onnx_path, original_images,
            model_output_dir, conf_threshold=0.3, model_type=model_type
        )
        results["detection_results"] = detection_results

    # Process each output tensor for visualization
    for idx, output_info in enumerate(outputs):
        tensor_name = output_info["name"]
        shape = output_info["shape"]
        fracbits = output_info["fracbits"]
        value_file = output_info.get("value", "")

        print(f"\n[{idx+1}/{len(outputs)}] Processing {tensor_name}")
        print(f"    Shape: {shape}")
        print(f"    Fracbits: {fracbits}")

        # Find the actual tensor file
        tensor_file = None
        if value_file and os.path.exists(os.path.join(model_dir, value_file)):
            tensor_file = os.path.join(model_dir, value_file)
        else:
            # Try to find tensor*.bin files
            pattern_name = f"*tensor{idx*2}.bin" if idx > 0 else "*tensor0.bin"
            matches = list(Path(model_dir).glob(pattern_name))
            if matches:
                tensor_file = str(matches[0])

        if not tensor_file:
            print(f"    Warning: Tensor file not found")
            results["outputs"].append({
                "name": tensor_name,
                "shape": shape,
                "status": "file_not_found"
            })
            continue

        print(f"    File: {os.path.basename(tensor_file)}")

        # Load tensor
        data = load_binary_file(tensor_file, dtype=np.int32, shape=tuple(shape), fracbits=fracbits)

        if data is None:
            print(f"    Error: Failed to load tensor")
            results["outputs"].append({
                "name": tensor_name,
                "shape": shape,
                "status": "load_failed"
            })
            continue

        print(f"    Loaded: {data.shape}, range=[{data.min():.4f}, {data.max():.4f}]")

        # Create visualization directory
        vis_dir = os.path.join(model_output_dir, f"output_{idx}_{tensor_name}")

        # Process based on model type
        if model_type == "segmentation":
            # Semantic segmentation visualization
            process_segmentation_output(data, vis_dir, original_images, batch_size)
        elif model_type == "pose":
            # Pose estimation visualization
            process_pose_output(data, vis_dir, original_images, batch_size)
        elif model_type == "mediapipe":
            # MediaPipe visualization
            process_mediapipe_output(data, vis_dir, original_images, batch_size)
        else:
            # Default visualization for detection/backbone
            process_default_output(data, vis_dir, original_images, batch_size)

        results["outputs"].append({
            "name": tensor_name,
            "shape": list(shape),
            "fracbits": fracbits,
            "data_range": [float(data.min()), float(data.max())],
            "status": "success"
        })

    return results


def process_segmentation_output(data, vis_dir, original_images, batch_size):
    """Process semantic segmentation output."""
    os.makedirs(vis_dir, exist_ok=True)

    # For each batch element
    for batch_idx in range(min(batch_size, data.shape[0])):
        batch_data = data[batch_idx]  # [C, H, W] or [H, W]

        # Generate colored segmentation map
        if batch_data.ndim == 3:
            # Take argmax to get class map
            seg_map = np.argmax(batch_data, axis=0)
        else:
            seg_map = batch_data

        # Create colored image
        h, w = seg_map.shape
        colored_seg = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id in np.unique(seg_map):
            if class_id < len(SEGMENTATION_COLORS):
                colored_seg[seg_map == class_id] = SEGMENTATION_COLORS[class_id]

        # Save colored segmentation
        seg_img = Image.fromarray(colored_seg)
        seg_path = os.path.join(vis_dir, f"segmentation_batch_{batch_idx}.png")
        seg_img.save(seg_path)
        print(f"    Saved segmentation: {seg_path}")

        # Overlay on original if available
        if batch_idx < len(original_images):
            orig_name, orig_img = original_images[batch_idx]
            overlay_path = os.path.join(vis_dir, f"overlay_batch_{batch_idx}.png")
            overlay_segmentation(orig_img, batch_data, overlay_path)

            # Side-by-side comparison
            compare_path = os.path.join(vis_dir, f"comparison_batch_{batch_idx}.png")
            create_side_by_side_comparison(orig_img, seg_img, compare_path)


def process_pose_output(data, vis_dir, original_images, batch_size):
    """Process pose estimation output."""
    os.makedirs(vis_dir, exist_ok=True)

    # For each batch element
    for batch_idx in range(min(batch_size, data.shape[0])):
        batch_data = data[batch_idx]  # [C, H, W]

        # Mean visualization
        if batch_data.ndim == 3:
            mean_data = np.mean(batch_data, axis=0)
        else:
            mean_data = batch_data

        # Normalize and save
        mean_data = (mean_data - mean_data.min()) / (mean_data.max() - mean_data.min() + 1e-8) * 255
        mean_img = Image.fromarray(mean_data.astype(np.uint8), mode='L')

        mean_path = os.path.join(vis_dir, f"heatmap_batch_{batch_idx}.png")
        mean_img.save(mean_path)
        print(f"    Saved heatmap: {mean_path}")

        # Grid view for multiple channels
        if batch_data.ndim == 3 and batch_data.shape[0] > 1:
            grid_path = os.path.join(vis_dir, f"grid_batch_{batch_idx}.png")
            visualize_tensor_grid(batch_data, grid_path)

        # Overlay on original if available
        if batch_idx < len(original_images):
            orig_name, orig_img = original_images[batch_idx]
            compare_path = os.path.join(vis_dir, f"comparison_batch_{batch_idx}.png")
            create_side_by_side_comparison(orig_img, mean_img, compare_path)


def process_mediapipe_output(data, vis_dir, original_images, batch_size):
    """Process MediaPipe output."""
    os.makedirs(vis_dir, exist_ok=True)

    # Similar to pose output
    process_pose_output(data, vis_dir, original_images, batch_size)


def process_default_output(data, vis_dir, original_images, batch_size):
    """Process default output (detection, backbone, etc.)."""
    os.makedirs(vis_dir, exist_ok=True)

    # Mean visualization
    mean_output = os.path.join(vis_dir, "mean.png")
    visualize_tensor(data, mean_output, title="mean")

    # Grid visualization (for 3D tensors with multiple channels)
    if data.ndim >= 3:
        grid_output = os.path.join(vis_dir, "grid.png")
        visualize_tensor_grid(data, grid_output)

    # Visualize each batch element
    if data.ndim == 4:  # [B, C, H, W]
        for batch_idx in range(min(batch_size, data.shape[0])):
            batch_data = data[batch_idx]

            # Convert to visualizable format
            if batch_data.ndim == 3:
                vis_data = np.mean(batch_data, axis=0)
            else:
                vis_data = batch_data

            # Normalize
            vis_data = (vis_data - vis_data.min()) / (vis_data.max() - vis_data.min() + 1e-8) * 255
            vis_img = Image.fromarray(vis_data.astype(np.uint8), mode='L')

            batch_output = os.path.join(vis_dir, f"batch_{batch_idx}.png")
            vis_img.save(batch_output)
            print(f"    Saved batch {batch_idx}: {batch_output}")

            # Overlay on original if available
            if batch_idx < len(original_images):
                orig_name, orig_img = original_images[batch_idx]
                compare_path = os.path.join(vis_dir, f"comparison_batch_{batch_idx}.png")
                create_side_by_side_comparison(orig_img, vis_img, compare_path,
                                             labels=("Original", f"Feature Map"))


# ==========================================
# Report Generation
# ==========================================

def generate_html_summary(all_results, output_path):
    """Generate HTML summary report."""
    total_models = len(all_results)
    successful = sum(1 for r in all_results if r and r.get("outputs"))
    failed = total_models - successful
    total_outputs = sum(r.get("num_outputs", 0) for r in all_results if r)

    # Build model cards HTML
    model_cards = ""
    for result in all_results:
        if not result:
            continue

        model_name = result["model_name"]
        model_full_name = result["model_full_name"]
        model_type = result["model_type"]
        outputs = result.get("outputs", [])

        # Type tag
        type_display = {
            "detection": "Detection",
            "segmentation": "Segmentation",
            "pose": "Pose",
            "backbone": "Backbone"
        }.get(model_type, model_type.capitalize())

        # Detection results summary
        detection_info = ""
        if result.get("detection_results"):
            det_count = sum(len(r.get("detections", [])) for r in result["detection_results"])
            detection_info = f"<p><strong>Detections:</strong> {det_count} boxes found across {len(result['detection_results'])} images</p>"

        output_list = ""
        for out in outputs:
            status_class = "success" if out.get("status") == "success" else "error"
            shape_str = str(out.get("shape", []))
            output_list += f"""
            <div class="output-item">
                <strong>{out.get('name', 'unknown')}</strong> - {shape_str}
                <span class="{status_class}">{out.get('status', 'unknown')}</span>
            </div>"""

        # Add detection images if available
        detection_images = ""
        if result.get("detection_results"):
            detection_images += '<div class="detection-gallery"><h4>Detection Results:</h4>'
            for det_result in result["detection_results"]:
                img_name = det_result.get("image", "")
                det_count = len(det_result.get("detections", []))
                img_path = f"{model_name}/detection_boxes_{img_name}"
                detection_images += f'<div class="img-container"><a href="{img_path}" target="_blank"><img src="{img_path}" alt="{img_name}" loading="lazy" title="Click to view full size"></a><span>{img_name} ({det_count} objs)</span></div>'
            detection_images += '</div>'

        model_cards += f"""
    <div class="model-card {model_type}">
        <h3>{model_name} <span class="tag tag-{model_type}">{type_display}</span></h3>
        <p><em>{model_full_name}</em></p>
        <div class="output-list">
            {output_list}
        </div>
        {detection_images}
        <p><a href="{model_name}/">Open Output Directory</a></p>
    </div>"""

    # Use f-string to avoid brace escaping issues
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>4-Batch CNN Post-processing Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .model-card {{ background: white; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #4CAF50; }}
        .model-card.detection {{ border-left-color: #FF5722; }}
        .model-card.segmentation {{ border-left-color: #2196F3; }}
        .model-card.pose {{ border-left-color: #9C27B0; }}
        .model-card h3 {{ margin-top: 0; color: #333; }}
        .output-list {{ margin: 10px 0; }}
        .output-item {{ padding: 5px 0; border-bottom: 1px solid #eee; }}
        .output-item:last-child {{ border-bottom: none; }}
        .success {{ color: green; }}
        .error {{ color: red; }}
        .stats {{ display: inline-block; margin: 0 15px; color: #666; }}
        .img-container {{ display: inline-block; margin: 5px; border: 1px solid #ddd; }}
        .img-container img {{ max-width: 200px; display: block; }}
        .img-container span {{ font-size: 12px; text-align: center; display: block; padding: 5px; }}
        a {{ color: #4CAF50; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-right: 5px; }}
        .tag-detection {{ background: #FF5722; color: white; }}
        .tag-segmentation {{ background: #2196F3; color: white; }}
        .tag-pose {{ background: #9C27B0; color: white; }}
        .tag-backbone {{ background: #607D8B; color: white; }}
    </style>
</head>
<body>
    <h1>4-Batch CNN Post-processing Results</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Models Processed:</strong> {total_models}</p>
        <p><strong>Successful:</strong> {successful} <span class="success">OK</span></p>
        <p><strong>Failed:</strong> {failed} <span class="error">FAIL</span></p>
        <p><strong>Total Output Tensors:</strong> {total_outputs}</p>
    </div>

    <h2>Model Results</h2>
    {model_cards}

    <footer style="margin-top: 50px; padding: 20px; text-align: center; color: #777;">
        Generated by 4-batch CNN Post-processing Tool
    </footer>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\nHTML report saved to: {output_path}")


def generate_markdown_summary(all_results, output_path):
    """Generate Markdown summary report."""
    lines = [
        "# 4-Batch CNN Post-processing Results\n",
        "## Summary\n",
        f"- **Total Models Processed:** {len(all_results)}",
        f"- **Successful:** {sum(1 for r in all_results if r and r.get('outputs'))}",
        f"- **Failed:** {sum(1 for r in all_results if not r or not r.get('outputs'))}",
        f"- **Total Output Tensors:** {sum(r.get('num_outputs', 0) for r in all_results if r)}\n",
        "## Model Results\n",
    ]

    for result in all_results:
        if not result:
            continue

        model_name = result["model_name"]
        model_full_name = result["model_full_name"]
        model_type = result["model_type"]
        outputs = result.get("outputs", [])

        type_icon = {
            "detection": "🎯",
            "segmentation": "🖼️",
            "pose": "🧍",
            "backbone": "🔧"
        }.get(model_type, "📦")

        lines.append(f"### {type_icon} {model_name}")
        lines.append(f"*{model_full_name} - {model_type}*\n")
        lines.append("| Output | Shape | Status |")
        lines.append("|--------|-------|--------|")

        for out in outputs:
            shape_str = str(out.get("shape", []))
            status = "✓" if out.get("status") == "success" else "✗"
            lines.append(f"| {out.get('name', 'unknown')} | {shape_str} | {status} |")

        lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Markdown report saved to: {output_path}")


# ==========================================
# Main Function
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Universal batch post-processing for 4-batch CNN outputs")
    parser.add_argument("--base-dir", default=".",
                        help="Base directory containing all model outputs (default: current directory)")
    parser.add_argument("--image-dir", required=True,
                        help="Directory containing original input images (required)")
    parser.add_argument("--heads-dir", required=True,
                        help="Directory containing head.onnx files (required)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for visualization")
    parser.add_argument("--models", nargs="*",
                        help="Specific model names to process (default: all models)")

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return

    print("="*60)
    print("4-Batch CNN Universal Post-processing Tool")
    print("="*60)
    print(f"Base directory: {base_dir.absolute()}")
    print(f"Image directory: {Path(args.image_dir).absolute()}")
    print(f"Heads directory: {Path(args.heads_dir).absolute()}")
    print(f"Output mode: Each model's results in <model_dir>/postprocess/")
    print("="*60)

    # Find all model directories
    model_dirs = []
    for item in base_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it contains parameters.json
            if (item / "*parameters.json").exists() or list(item.glob("*parameters.json")):
                model_dirs.append(item)

    # Filter by specified models if provided
    if args.models:
        model_dirs = [d for d in model_dirs if d.name in args.models]

    if not model_dirs:
        print("No model directories found!")
        return

    print(f"\nFound {len(model_dirs)} model(s) to process\n")

    # Process each model
    all_results = []
    summary_outputs = []  # Track output directories for summary

    for model_dir in sorted(model_dirs):
        # Output directory: model_dir / "postprocess"
        # e.g., E:\test-Models\4_batch_cnn_tool\4batch-example\atss_r50\postprocess\
        output_dir = model_dir / "postprocess"
        output_dir.mkdir(exist_ok=True, parents=True)

        print(f"Output directory: {output_dir.absolute()}")

        try:
            result = process_model(
                str(model_dir),
                str(output_dir),
                args.image_dir,
                args.heads_dir,
                args.batch_size
            )
            if result:
                result["output_dir"] = str(output_dir)
                all_results.append(result)
                summary_outputs.append(output_dir)
        except Exception as e:
            print(f"Error processing {model_dir.name}: {e}")
            traceback.print_exc()
            all_results.append(None)

    # Generate summary reports in the base directory
    print("\n" + "="*60)
    print("Generating summary reports...")
    print("="*60)

    html_report = base_dir / "summary_report.html"
    generate_html_summary(all_results, str(html_report))

    md_report = base_dir / "summary_report.md"
    generate_markdown_summary(all_results, str(md_report))

    print("\n" + "="*60)
    print("Post-processing complete!")
    print(f"Base directory: {base_dir.absolute()}")
    print(f"Summary reports: {html_report} and {md_report}")
    print("="*60)


if __name__ == "__main__":
    main()
