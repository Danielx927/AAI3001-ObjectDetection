import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

@dataclass
class BboxMetrics:
    """Store metrics for bounding box evaluation"""
    map_50: float
    map_75: float
    map_50_95: float
    mean_iou: float
    recall_50: float
    precision_50: float
    false_positives: int
    false_negatives: int
    total_predictions: int
    total_ground_truths: int
    inference_time_ms: float = 0.0


class BboxEvaluator:
    """Evaluate object detection models for bounding box accuracy only"""
    
    def __init__(self, iou_thresholds: List[float] = None):
        """
        Args:
            iou_thresholds: List of IoU thresholds for mAP calculation
        """
        self.iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)
        self.results = {
            'predictions': [],
            'ground_truths': [],
            'matches': [],
            'ious': []
        }
    
    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate IoU between two boxes
        Args:
            box1, box2: [x1, y1, x2, y2] format
        Returns:
            IoU score
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def match_boxes(self, pred_boxes: np.ndarray, gt_boxes: np.ndarray, 
                    iou_threshold: float = 0.5) -> Tuple[List, List, List]:
        """
        Match predicted boxes to ground truth boxes using Hungarian algorithm
        Args:
            pred_boxes: [N, 4] predicted boxes
            gt_boxes: [M, 4] ground truth boxes
            iou_threshold: minimum IoU for a match
        Returns:
            matched_pairs: List of (pred_idx, gt_idx, iou)
            unmatched_preds: List of unmatched prediction indices
            unmatched_gts: List of unmatched ground truth indices
        """
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                iou_matrix[i, j] = self.calculate_iou(pred_box, gt_box)
        
        # Use Hungarian algorithm for optimal matching
        pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)
        
        matched_pairs = []
        unmatched_preds = list(range(len(pred_boxes)))
        unmatched_gts = list(range(len(gt_boxes)))
        
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            iou = iou_matrix[pred_idx, gt_idx]
            if iou >= iou_threshold:
                matched_pairs.append((pred_idx, gt_idx, iou))
                unmatched_preds.remove(pred_idx)
                unmatched_gts.remove(gt_idx)
        
        return matched_pairs, unmatched_preds, unmatched_gts
    
    def add_image_predictions(self, pred_boxes: np.ndarray, pred_scores: np.ndarray,
                             gt_boxes: np.ndarray, image_id: str = None):
        """
        Add predictions for a single image
        Args:
            pred_boxes: [N, 4] predicted boxes in [x1, y1, x2, y2] format
            pred_scores: [N] confidence scores
            gt_boxes: [M, 4] ground truth boxes
            image_id: optional image identifier
        """
        self.results['predictions'].append({
            'boxes': pred_boxes,
            'scores': pred_scores,
            'image_id': image_id
        })
        self.results['ground_truths'].append({
            'boxes': gt_boxes,
            'image_id': image_id
        })
    
    def calculate_metrics(self, confidence_threshold: float = 0.5) -> BboxMetrics:
        """
        Calculate all metrics for the current predictions
        Args:
            confidence_threshold: minimum confidence score for predictions
        Returns:
            BboxMetrics object with all computed metrics
        """
        all_matches = []
        all_ious = []
        total_fps = 0
        total_fns = 0
        total_preds = 0
        total_gts = 0
        
        # Process each image
        for pred_dict, gt_dict in zip(self.results['predictions'], 
                                      self.results['ground_truths']):
            pred_boxes = pred_dict['boxes']
            pred_scores = pred_dict['scores']
            gt_boxes = gt_dict['boxes']
            
            # Filter by confidence threshold
            valid_mask = pred_scores >= confidence_threshold
            filtered_pred_boxes = pred_boxes[valid_mask]
            
            total_preds += len(filtered_pred_boxes)
            total_gts += len(gt_boxes)
            
            # Match boxes at IoU 0.5
            matched_pairs, unmatched_preds, unmatched_gts = self.match_boxes(
                filtered_pred_boxes, gt_boxes, iou_threshold=0.5
            )
            
            all_matches.extend(matched_pairs)
            total_fps += len(unmatched_preds)
            total_fns += len(unmatched_gts)
            
            # Calculate IoUs for matched pairs
            for pred_idx, gt_idx, iou in matched_pairs:
                all_ious.append(iou)
        
        # Calculate precision and recall
        true_positives = len(all_matches)
        recall = true_positives / total_gts if total_gts > 0 else 0.0
        precision = true_positives / total_preds if total_preds > 0 else 0.0
        
        # Calculate mAP at different IoU thresholds
        map_values = []
        for iou_thresh in self.iou_thresholds:
            ap = self._calculate_ap_at_iou(iou_thresh, confidence_threshold)
            map_values.append(ap)
        
        map_50 = map_values[0]  # IoU 0.5
        map_75 = map_values[5] if len(map_values) > 5 else 0.0  # IoU 0.75
        map_50_95 = np.mean(map_values)
        
        # Calculate IoU statistics
        mean_iou = np.mean(all_ious) if all_ious else 0.0
        
        return BboxMetrics(
            map_50=map_50,
            map_75=map_75,
            map_50_95=map_50_95,
            mean_iou=mean_iou,
            recall_50=recall,
            precision_50=precision,
            false_positives=total_fps,
            false_negatives=total_fns,
            total_predictions=total_preds,
            total_ground_truths=total_gts
        )
    
    def _calculate_ap_at_iou(self, iou_threshold: float, 
                            confidence_threshold: float = 0.0) -> float:
        """Calculate Average Precision at a specific IoU threshold"""
        # Collect all predictions with scores
        all_predictions = []
        for pred_dict, gt_dict in zip(self.results['predictions'], 
                                      self.results['ground_truths']):
            pred_boxes = pred_dict['boxes']
            pred_scores = pred_dict['scores']
            gt_boxes = gt_dict['boxes']
            
            valid_mask = pred_scores >= confidence_threshold
            filtered_boxes = pred_boxes[valid_mask]
            filtered_scores = pred_scores[valid_mask]
            
            matched_pairs, _, _ = self.match_boxes(
                filtered_boxes, gt_boxes, iou_threshold=iou_threshold
            )
            
            for i, score in enumerate(filtered_scores):
                is_tp = any(pred_idx == i for pred_idx, _, _ in matched_pairs)
                all_predictions.append((score, is_tp))
        
        # Sort by confidence score (descending)
        all_predictions.sort(key=lambda x: x[0], reverse=True)
        
        # Calculate precision-recall curve
        tp_cumsum = 0
        precisions = []
        recalls = []
        
        total_gt = sum(len(gt['boxes']) for gt in self.results['ground_truths'])
        
        for i, (_, is_tp) in enumerate(all_predictions):
            if is_tp:
                tp_cumsum += 1
            
            precision = tp_cumsum / (i + 1)
            recall = tp_cumsum / total_gt if total_gt > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            matching_precisions = [p for r, p in zip(recalls, precisions) if r >= t]
            max_precision = max(matching_precisions) if matching_precisions else 0.0
            ap += max_precision / 11.0
        
        return ap
    
    def calculate_precision_recall_curve(self, iou_threshold: float = 0.5):
        """Calculate precision-recall curve at different confidence thresholds"""
        confidence_thresholds = np.arange(0.0, 1.0, 0.05)
        precisions = []
        recalls = []
        
        for conf_thresh in confidence_thresholds:
            metrics = self.calculate_metrics(confidence_threshold=conf_thresh)
            precisions.append(metrics.precision_50)
            recalls.append(metrics.recall_50)
        
        return recalls, precisions, confidence_thresholds
    
    def get_iou_distribution(self, confidence_threshold: float = 0.5) -> List[float]:
        """Get all IoU values for matched boxes"""
        all_ious = []
        
        for pred_dict, gt_dict in zip(self.results['predictions'], 
                                      self.results['ground_truths']):
            pred_boxes = pred_dict['boxes']
            pred_scores = pred_dict['scores']
            gt_boxes = gt_dict['boxes']
            
            valid_mask = pred_scores >= confidence_threshold
            filtered_pred_boxes = pred_boxes[valid_mask]
            
            matched_pairs, _, _ = self.match_boxes(
                filtered_pred_boxes, gt_boxes, iou_threshold=0.5
            )
            
            for pred_idx, gt_idx, iou in matched_pairs:
                all_ious.append(iou)
        
        return all_ious
    
    def plot_precision_recall_curve(self, save_path: Optional[Path] = None):
        """Plot and optionally save precision-recall curve"""
        recalls, precisions, conf_thresholds = self.calculate_precision_recall_curve()
        
        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, linewidth=2, marker='o', markersize=4)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve (IoU@0.5)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_iou_distribution(self, save_path: Optional[Path] = None):
        """Plot IoU distribution histogram"""
        ious = self.get_iou_distribution()
        
        if not ious:
            print("No matched boxes to plot IoU distribution")
            return
        
        plt.figure(figsize=(10, 6))
        plt.hist(ious, bins=20, edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(ious), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(ious):.3f}')
        plt.axvline(np.median(ious), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(ious):.3f}')
        plt.xlabel('IoU', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('IoU Distribution of Matched Boxes', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confidence_analysis(self, save_path: Optional[Path] = None):
        """Plot metrics vs confidence threshold"""
        confidence_thresholds = np.arange(0.1, 1.0, 0.05)
        precisions = []
        recalls = []
        f1_scores = []
        
        for conf_thresh in confidence_thresholds:
            metrics = self.calculate_metrics(confidence_threshold=conf_thresh)
            precisions.append(metrics.precision_50)
            recalls.append(metrics.recall_50)
            
            if metrics.precision_50 + metrics.recall_50 > 0:
                f1 = 2 * (metrics.precision_50 * metrics.recall_50) / \
                     (metrics.precision_50 + metrics.recall_50)
            else:
                f1 = 0
            f1_scores.append(f1)
        
        plt.figure(figsize=(12, 6))
        plt.plot(confidence_thresholds, precisions, label='Precision', linewidth=2, marker='o')
        plt.plot(confidence_thresholds, recalls, label='Recall', linewidth=2, marker='s')
        plt.plot(confidence_thresholds, f1_scores, label='F1-Score', linewidth=2, marker='^')
        plt.xlabel('Confidence Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Metrics vs Confidence Threshold', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, model_name: str, output_dir: Path, 
                       confidence_threshold: float = 0.5):
        """
        Generate complete evaluation report for a model
        Args:
            model_name: Name of the model being evaluated
            output_dir: Directory to save results
            confidence_threshold: Confidence threshold for metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate metrics
        metrics = self.calculate_metrics(confidence_threshold=confidence_threshold)
        
        # Save metrics as JSON
        metrics_dict = asdict(metrics)
        metrics_dict['model_name'] = model_name
        metrics_dict['confidence_threshold'] = confidence_threshold
        
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        # Generate plots
        self.plot_precision_recall_curve(output_dir / 'precision_recall_curve.png')
        self.plot_iou_distribution(output_dir / 'iou_distribution.png')
        self.plot_confidence_analysis(output_dir / 'confidence_analysis.png')
        
        # Generate text report
        report_lines = [
            f"=" * 60,
            f"BOUNDING BOX EVALUATION REPORT: {model_name}",
            f"=" * 60,
            f"\nConfidence Threshold: {confidence_threshold}",
            f"\n{'-' * 60}",
            f"PRIMARY METRICS",
            f"{'-' * 60}",
            f"mAP@0.5:          {metrics.map_50:.4f}",
            f"mAP@0.75:         {metrics.map_75:.4f}",
            f"mAP@[0.5:0.95]:   {metrics.map_50_95:.4f}",
            f"Mean IoU:         {metrics.mean_iou:.4f}",
            f"\n{'-' * 60}",
            f"DETECTION METRICS (IoU@0.5)",
            f"{'-' * 60}",
            f"Recall:           {metrics.recall_50:.4f}",
            f"Precision:        {metrics.precision_50:.4f}",
            f"F1-Score:         {2 * metrics.precision_50 * metrics.recall_50 / (metrics.precision_50 + metrics.recall_50) if (metrics.precision_50 + metrics.recall_50) > 0 else 0:.4f}",
            f"\n{'-' * 60}",
            f"ERROR ANALYSIS",
            f"{'-' * 60}",
            f"False Positives:  {metrics.false_positives}",
            f"False Negatives:  {metrics.false_negatives}",
            f"Total Predictions: {metrics.total_predictions}",
            f"Total Ground Truths: {metrics.total_ground_truths}"
        ]
        
        if metrics.inference_time_ms > 0:
            report_lines.extend([
                f"\n{'-' * 60}",
                f"PERFORMANCE",
                f"{'-' * 60}",
                f"Inference Time:   {metrics.inference_time_ms:.2f} ms",
                f"FPS:              {1000/metrics.inference_time_ms:.2f}",
            ])
        
        report_lines.append(f"\n{'=' * 60}\n")
        
        report_text = "\n".join(report_lines)
        
        with open(output_dir / 'evaluation_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nReport saved to: {output_dir}")
        
        return metrics
    
def load_faster_rcnn_model(weights_path=None, num_classes=11):
    """
    Load Faster R-CNN model
    Args:
        weights_path: Path to trained weights (optional)
        num_classes: Number of classes (10 fruits + 1 background)
    """
    # Load pretrained Faster R-CNN
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    
    # Replace the classifier head for your number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    
    # Load your trained weights if provided
    if weights_path:
        checkpoint = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def load_image(image_path):
    """Load and preprocess image for Faster R-CNN"""
    image = Image.open(image_path).convert("RGB")
    return image

def preprocess_image(image):
    """Convert PIL image to tensor"""
    image_tensor = F.to_tensor(image)
    return image_tensor

def load_yolo_labels(label_path, img_width, img_height):
    """
    Convert YOLO format labels to [x1, y1, x2, y2] format
    YOLO format: [class, x_center, y_center, width, height] (normalized 0-1)
    """
    if not label_path.exists():
        return np.array([]).reshape(0, 4)
    
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Ignore class (parts[0]), only get bbox
                x_center, y_center, width, height = map(float, parts[1:5])
                
                # Convert to pixel coordinates
                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height
                
                # Convert to x1, y1, x2, y2
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                boxes.append([x1, y1, x2, y2])
    
    return np.array(boxes)

def evaluate_faster_rcnn(
    model_weights_path,
    test_image_dir,
    test_label_dir,
    output_dir,
    confidence_threshold=0.5,
    device='cuda'
):
    """
    Complete evaluation pipeline for Faster R-CNN
    
    Args:
        model_weights_path: Path to trained model weights
        test_image_dir: Directory with test images
        test_label_dir: Directory with YOLO format labels
        output_dir: Where to save evaluation results
        confidence_threshold: Minimum confidence for predictions
        device: 'cuda' or 'cpu'
    """
    # Load model
    print("Loading Faster R-CNN model...")
    model = load_faster_rcnn_model(model_weights_path)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize evaluator
    evaluator = BboxEvaluator()
    
    # Process all test images
    test_image_dir = Path(test_image_dir)
    test_label_dir = Path(test_label_dir)
    
    image_paths = list(test_image_dir.glob('*.jpg')) + list(test_image_dir.glob('*.png'))
    print(f"Found {len(image_paths)} test images")
    
    for i, image_path in enumerate(image_paths):
        if (i + 1) % 10 == 0:
            print(f"Processing {i+1}/{len(image_paths)}...")
        
        # Load and preprocess image
        image = load_image(image_path)
        img_width, img_height = image.size
        image_tensor = preprocess_image(image).to(device)
        
        # Run inference
        with torch.no_grad():
            predictions = model([image_tensor])[0]
        
        # Extract predictions
        pred_boxes = predictions['boxes'].cpu().numpy()  # [N, 4] in x1,y1,x2,y2
        pred_scores = predictions['scores'].cpu().numpy()  # [N]
        # pred_labels = predictions['labels'].cpu().numpy()  # [N] - we ignore these
        
        # Load ground truth
        label_path = test_label_dir / f"{image_path.stem}.txt"
        gt_boxes = load_yolo_labels(label_path, img_width, img_height)
        
        # Add to evaluator
        evaluator.add_image_predictions(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            image_id=image_path.stem
        )
    
    # Generate evaluation report
    print("\nGenerating evaluation report...")
    metrics = evaluator.generate_report(
        model_name="Faster R-CNN",
        output_dir=Path(output_dir),
        confidence_threshold=confidence_threshold
    )
    
    return metrics

def load_yolo_model(weights_path):
    """
    Load YOLO model (YOLOv8, YOLOv5, etc.)
    Args:
        weights_path: Path to trained weights (.pt file)
    """
    model = YOLO(weights_path)
    return model

def evaluate_yolo(
    model_weights_path,
    test_image_dir,
    test_label_dir,
    output_dir,
    confidence_threshold=0.5,
    iou_threshold=0.45,  # NMS IoU threshold
    device='0'  # '0' for GPU, 'cpu' for CPU
):
    """
    Complete evaluation pipeline for YOLO models
    
    Args:
        model_weights_path: Path to trained YOLO weights (.pt file)
        test_image_dir: Directory with test images
        test_label_dir: Directory with YOLO format labels
        output_dir: Where to save evaluation results
        confidence_threshold: Minimum confidence for predictions
        iou_threshold: IoU threshold for NMS
        device: '0' for GPU, 'cpu' for CPU
    """
    # Load model
    print(f"Loading YOLO model from {model_weights_path}...")
    model = load_yolo_model(model_weights_path)
    
    # Initialize evaluator
    evaluator = BboxEvaluator()
    
    # Process all test images
    test_image_dir = Path(test_image_dir)
    test_label_dir = Path(test_label_dir)
    
    image_paths = list(test_image_dir.glob('*.jpg')) + list(test_image_dir.glob('*.png'))
    print(f"Found {len(image_paths)} test images")
    
    for i, image_path in enumerate(image_paths):
        if (i + 1) % 10 == 0:
            print(f"Processing {i+1}/{len(image_paths)}...")
        
        # Run YOLO inference
        results = model(
            str(image_path),
            conf=0.001,  # Use very low threshold, we'll filter later in evaluator
            iou=iou_threshold,
            device=device,
            verbose=False
        )[0]
        
        # Get image dimensions
        img_height, img_width = results.orig_shape
        
        # Extract predictions
        if len(results.boxes) > 0:
            pred_boxes = results.boxes.xyxy.cpu().numpy()  # [N, 4] in x1,y1,x2,y2
            pred_scores = results.boxes.conf.cpu().numpy()  # [N]
            # pred_classes = results.boxes.cls.cpu().numpy()  # [N] - we ignore these
        else:
            pred_boxes = np.array([]).reshape(0, 4)
            pred_scores = np.array([])
        
        # Load ground truth
        label_path = test_label_dir / f"{image_path.stem}.txt"
        gt_boxes = load_yolo_labels(label_path, img_width, img_height)
        
        # Add to evaluator
        evaluator.add_image_predictions(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            image_id=image_path.stem
        )
    
    # Generate evaluation report
    print("\nGenerating evaluation report...")
    metrics = evaluator.generate_report(
        model_name=f"YOLO ({Path(model_weights_path).stem})",
        output_dir=Path(output_dir),
        confidence_threshold=confidence_threshold
    )
    
    return metrics