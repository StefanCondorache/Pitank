import os
import cv2 # type: ignore
import numpy as np
import random

def load_plain_circles(directory, img_size=(64, 64)):
    print(f"Loading base circles from {directory}...")
    circles = []
    if os.path.exists(directory):
        for f in os.listdir(directory):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(directory, f)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    circles.append(img)
    else:
        print(f"Warning: Directory {directory} not found.")
    return circles

def apply_occlusion(img):
    """Covers random image region with noise for augmentation."""
    h, w = img.shape
    occ_img = img.copy()
    type_idx = random.randint(0, 5)
    if type_idx == 0: return occ_img

    block_h = h // 3
    block_w = w // 3
    noise = np.random.randint(0, 255, (h, w), dtype='uint8')
    if type_idx == 1:  # Right
        occ_img[:, w-block_w:] = noise[:, w-block_w:]
    elif type_idx == 2:  # Left
        occ_img[:, :block_w] = noise[:, :block_w]
    elif type_idx == 3:  # Top
        occ_img[:block_h, :] = noise[:block_h, :]
    elif type_idx == 4:  # Bottom
        occ_img[h-block_h:, :] = noise[h-block_h:, :]
    elif type_idx == 5:  # Random Corner
        cx = random.choice([0, w-block_w])
        cy = random.choice([0, h-block_h])
        occ_img[cy:cy+block_h, cx:cx+block_w] = noise[cy:cy+block_h, cx:cx+block_w]
    return occ_img

def create_synthetic_data(base_circles, num_samples=1000, img_size=(64, 64)):
    print("Generating Synthetic Data with Custom Filters...")
    X, y = [], []
    if not base_circles:
        print("No base circles provided!")
        return np.array(X), np.array(y)

    for _ in range(num_samples):
        base = random.choice(base_circles).copy()
        label = random.randint(0, 2)
        # Draw symbol (left, right, turnaround)
        if label == 0: # Left
            cv2.arrowedLine(base, (54, 32), (10, 32), (255), 4, tipLength=0.3)
        elif label == 1: # Right
            cv2.arrowedLine(base, (10, 32), (54, 32), (255), 4, tipLength=0.3)
        elif label == 2: # U-Turn
            cv2.arrowedLine(base, (24, 28), (24, 52), (255), 4, tipLength=0.3)
            cv2.line(base, (40, 52), (40, 28), (255), 4)
            cv2.ellipse(base, (32, 28), (8, 8), 0, 180, 360, (255), 4)
        # Add contrast, then occlusion
        _, contrast_img = cv2.threshold(base, 127, 255, cv2.THRESH_BINARY)
        final_img = apply_occlusion(contrast_img)
        X.append(final_img)
        y.append(label)
    return np.array(X), np.array(y)