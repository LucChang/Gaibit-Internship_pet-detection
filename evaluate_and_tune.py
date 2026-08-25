#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Evaluation and Threshold Tuning Script
"""

import os
import sys
import yaml
from ultralytics import YOLO

# Dataset data.yaml path
DATA_YAML = os.path.abspath(os.path.join("Cat and Dog cctv.yolo26", "data.yaml"))

MODEL_PATH = os.path.join("runs", "detect", "runs", "detect", "train_yolo26_small-3", "weights", "best.pt")

if not os.path.exists(MODEL_PATH):
    # Fallback to any best.pt found
    for root, _, files in os.walk("runs"):
        if "best.pt" in files:
            MODEL_PATH = os.path.join(root, "best.pt")
            break

print(f"Loading model for evaluation: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

conf_candidates = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
results_summary = []

best_f1 = -1.0
best_conf = 0.25
best_metrics = None

print("\n================ EVALUATION & THRESHOLD SCAN ================")
print(f"{'Conf':<8}{'Precision':<12}{'Recall':<10}{'mAP50':<10}{'mAP50-95':<12}{'F1 Score':<10}")
print("-" * 62)

for conf_val in conf_candidates:
    metrics = model.val(
        data=DATA_YAML,
        split='val',
        imgsz=640,
        conf=conf_val,
        iou=0.45,
        device=0,
        verbose=False
    )
    
    p = metrics.results_dict.get('metrics/precision(B)', 0.0)
    r = metrics.results_dict.get('metrics/recall(B)', 0.0)
    map50 = metrics.results_dict.get('metrics/mAP50(B)', 0.0)
    map50_95 = metrics.results_dict.get('metrics/mAP50-95(B)', 0.0)
    
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    print(f"{conf_val:<8.2f}{p:<12.4f}{r:<10.4f}{map50:<10.4f}{map50_95:<12.4f}{f1:<10.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_conf = conf_val
        best_metrics = {
            'precision': p,
            'recall': r,
            'map50': map50,
            'map50_95': map50_95,
            'f1': f1
        }

print("\n================ OPTIMAL THRESHOLD RESULT ================")
print(f"Optimal Confidence Threshold: {best_conf}")
print(f"  - Precision : {best_metrics['precision']:.4f}")
print(f"  - Recall    : {best_metrics['recall']:.4f}")
print(f"  - mAP50     : {best_metrics['map50']:.4f}")
print(f"  - mAP50-95  : {best_metrics['map50_95']:.4f}")
print(f"  - F1 Score  : {best_metrics['f1']:.4f}")

# Update bytetrack_tuned.yaml with optimal thresholds
bytetrack_config = {
    'tracker_type': 'bytetrack',
    'track_high_thresh': round(best_conf, 2),
    'track_low_thresh': round(max(0.05, best_conf - 0.15), 2),
    'new_track_thresh': round(best_conf + 0.05, 2),
    'track_buffer': 60,
    'match_thresh': 0.5,
    'fuse_score': True
}

bytetrack_path = "bytetrack_tuned.yaml"
with open(bytetrack_path, 'w', encoding='utf-8') as f:
    yaml.dump(bytetrack_config, f, default_flow_style=False)

print(f"\nUpdated {bytetrack_path} with tuned parameters:")
print(yaml.dump(bytetrack_config, default_flow_style=False))
