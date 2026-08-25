#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copy-Paste Augmentation for MP4 Training Images in YOLO Dataset
"""

import os
import glob
import random
import cv2
import numpy as np

# Set random seed
random.seed(42)
np.random.seed(42)

BASE_DIR = r"c:\Users\USER\Desktop\GAIBIT_INTERNSHIP\Objection_localization_roi_detect\Cat and Dog cctv.yolo26"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train", "images")
TRAIN_LBL_DIR = os.path.join(BASE_DIR, "train", "labels")

# Class mapping: 0: bowl, 1: cat, 2: litter_box

def extract_object_patches():
    """Extract object patches from all training images for Copy-Paste bank"""
    patches = [] # [{ 'class': int, 'patch': np.ndarray, 'w_ratio': float, 'h_ratio': float }]
    
    img_files = glob.glob(os.path.join(TRAIN_IMG_DIR, "*.*"))
    for img_path in img_files:
        ext = os.path.splitext(img_path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            continue
            
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(TRAIN_LBL_DIR, stem + ".txt")
        
        if not os.path.exists(lbl_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h_img, w_img, _ = img.shape
        
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, w_norm, h_norm = map(float, parts[1:5])
                    
                    x1 = max(0, int((xc - w_norm / 2) * w_img))
                    y1 = max(0, int((yc - h_norm / 2) * h_img))
                    x2 = min(w_img, int((xc + w_norm / 2) * w_img))
                    y2 = min(h_img, int((yc + h_norm / 2) * h_img))
                    
                    if (x2 - x1) > 10 and (y2 - y1) > 10:
                        patch = img[y1:y2, x1:x2].copy()
                        patches.append({
                            'class': cls_id,
                            'patch': patch,
                            'w_ratio': w_norm,
                            'h_ratio': h_norm
                        })
                        
    print(f"Extracted {len(patches)} object patches across dataset.")
    return patches


def blend_patch(bg_img, patch, px, py):
    """Paste patch onto bg_img with smooth edge blending"""
    ph, pw, _ = patch.shape
    bh, bw, _ = bg_img.shape
    
    px = max(0, min(bw - pw, px))
    py = max(0, min(bh - ph, py))
    
    # Create mask with feathered edges
    mask = np.ones((ph, pw), dtype=np.float32)
    blur_kernel = max(3, min(ph, pw) // 10)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
    mask = np.stack([mask] * 3, axis=-1)
    
    roi = bg_img[py:py+ph, px:px+pw].astype(np.float32)
    blended = (patch.astype(np.float32) * mask + roi * (1.0 - mask)).astype(np.uint8)
    bg_img[py:py+ph, px:px+pw] = blended
    return bg_img, (px, py, pw, ph)


def apply_copy_paste(num_copies_per_mp4=6):
    patches = extract_object_patches()
    if not patches:
        print("Error: No object patches extracted.")
        return
        
    img_files = glob.glob(os.path.join(TRAIN_IMG_DIR, "*.*"))
    mp4_files = [f for f in img_files if 'mp4' in os.path.basename(f).lower()]
    
    print(f"Found {len(mp4_files)} MP4 images in train split. Generating {num_copies_per_mp4} Copy-Paste augmentations per file...")
    
    created_count = 0
    
    for img_path in mp4_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]
        lbl_path = os.path.join(TRAIN_LBL_DIR, stem + ".txt")
        
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            continue
            
        orig_labels = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        orig_labels.append(line.strip())
                        
        bh, bw, _ = orig_img.shape
        
        for copy_idx in range(1, num_copies_per_mp4 + 1):
            aug_img = orig_img.copy()
            aug_labels = orig_labels.copy()
            
            # Pick 2-4 patches to paste
            num_pastes = random.randint(2, 4)
            selected_patches = random.choices(patches, k=num_pastes)
            
            for item in selected_patches:
                patch = item['patch'].copy()
                cls_id = item['class']
                
                # Random scale
                scale = random.uniform(0.7, 1.3)
                nh = max(10, int(patch.shape[0] * scale))
                nw = max(10, int(patch.shape[1] * scale))
                if nh >= bh or nw >= bw:
                    continue
                patch = cv2.resize(patch, (nw, nh))
                
                # Random horizontal flip
                if random.random() > 0.5:
                    patch = cv2.flip(patch, 1)
                    
                # Random brightness jitter
                brightness = random.uniform(0.85, 1.15)
                patch = np.clip(patch.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
                
                # Random location
                px = random.randint(0, max(1, bw - nw))
                py = random.randint(0, max(1, bh - nh))
                
                aug_img, (real_x, real_y, real_w, real_h) = blend_patch(aug_img, patch, px, py)
                
                # Calculate normalized bbox
                xc = (real_x + real_w / 2.0) / bw
                yc = (real_y + real_h / 2.0) / bh
                wn = real_w / bw
                hn = real_h / bh
                
                aug_labels.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
                
            # Save augmented image and label
            out_img_name = f"{stem}_cpaug{copy_idx}{ext}"
            out_lbl_name = f"{stem}_cpaug{copy_idx}.txt"
            
            out_img_path = os.path.join(TRAIN_IMG_DIR, out_img_name)
            out_lbl_path = os.path.join(TRAIN_LBL_DIR, out_lbl_name)
            
            cv2.imwrite(out_img_path, aug_img)
            with open(out_lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(aug_labels) + "\n")
                
            created_count += 1
            
    print(f"\n==========================================")
    print(f"Copy-Paste Augmentation Complete!")
    print(f"Created {created_count} augmented MP4 training images.")
    print(f"Total training images now: {len(os.listdir(TRAIN_IMG_DIR))}")
    print(f"==========================================")

if __name__ == "__main__":
    apply_copy_paste(num_copies_per_mp4=6)
