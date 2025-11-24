import os
import cv2 # type: ignore
import numpy as np
import random

from image_processing import apply_custom_filter

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

def load_plain_circles(directory, img_size):
    """
    Loads base circle templates. 
    Tries to load in COLOR to preserve the blue if possible, 
    otherwise converts to BGR.
    """
    print(f"Loading base circles from {directory}...")
    circles = []
    if os.path.exists(directory):
        for f in os.listdir(directory):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(directory, f)
                # Load in Color (BGR) so we can have Blue background
                img = cv2.imread(path, cv2.IMREAD_COLOR) 
                if img is not None:
                    img = cv2.resize(img, img_size)
                    circles.append(img)
    return circles

def draw_thick_arrow(img, pt1, pt2, color=(255, 255, 255), thickness=4, tip_size=10):
    """
    Draws a 'perfect' traffic sign arrow with a distinct head.
    """
    # Draw the shaft
    cv2.line(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
    
    # Calculate angle for the arrowhead
    angle = np.arctan2(pt1[1] - pt2[1], pt1[0] - pt2[0])
    
    # Arrowhead points
    p1 = (int(pt2[0] + tip_size * np.cos(angle + np.pi / 6)),
          int(pt2[1] + tip_size * np.sin(angle + np.pi / 6)))
    p2 = (int(pt2[0] + tip_size * np.cos(angle - np.pi / 6)),
          int(pt2[1] + tip_size * np.sin(angle - np.pi / 6)))
    
    # Draw filled triangle for head
    triangle_cnt = np.array([pt2, p1, p2])
    cv2.fillPoly(img, [triangle_cnt], color)

def create_synthetic_data(base_circles, save_dir='dataset/synthetic_created', num_samples=1000, img_size=(64,64)):
    """
    Generates 'Perfect' synthetic signs (Blue & White) and saves them.
    Returns Grayscale arrays for training.
    """
    print(f"Generating {num_samples} Perfect Signs...")
    
    # 1. Setup Save Directories
    classes = ['0_left', '1_right', '2_turn_around']
    
    for cls in classes:
        os.makedirs(os.path.join(save_dir, cls), exist_ok=True)
        
    X = []
    y = []
    
    if not base_circles:
        print("Error: No base circles provided!")
        return np.array(X), np.array(y)

    for i in range(num_samples):
        # Pick a base circle
        base = random.choice(base_circles).copy()
        
        # Ensure base is BGR (Blue-Green-Red)
        if len(base.shape) == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            
        mask = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        base[mask > 0] = (255, 0, 0) # OpenCV Blue
        
        label = random.randint(0, 2)
        save_folder = classes[label]
        
        # --- DRAWING PERFECT SIGNS ---
        WHITE_COLOR = (255, 255, 255)
        THICK = 6 
        
        if label == 0: # LEFT
            # Arrow pointing Left
            start = (48, 32)
            end = (16, 32)
            draw_thick_arrow(base, start, end, WHITE_COLOR, thickness=THICK, tip_size=14)
            
        elif label == 1: # RIGHT
            # Arrow pointing Right
            start = (16, 32)
            end = (48, 32)
            draw_thick_arrow(base, start, end, WHITE_COLOR, thickness=THICK, tip_size=14)
            
        elif label == 2: # TURN AROUND (U-Turn)
            # Inverted U shape
            # Left Leg (Down with Arrow)
            draw_thick_arrow(base, (24, 30), (24, 50), WHITE_COLOR, thickness=THICK, tip_size=12)
            # Right Leg (Straight Up)
            cv2.line(base, (40, 50), (40, 30), WHITE_COLOR, THICK, lineType=cv2.LINE_AA)
            # Top Arc
            # Center(32, 30), Axes(8, 8), Angle 0, Start 180, End 360
            cv2.ellipse(base, (32, 30), (8, 8), 0, 180, 360, WHITE_COLOR, THICK, lineType=cv2.LINE_AA)

        # --- SAVE TO DISK (COLOR) ---
        filename = f"{save_dir}/{save_folder}/synth_{i}.png"
        cv2.imwrite(filename, base)

        # --- PREPARE FOR MODEL (GRAYSCALE) ---
        # The model expects Grayscale inputs
        gray_version = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        
        # Apply the Matrix Filter (The "Useful Part")
        filtered = apply_custom_filter(gray_version)
        
        X.append(filtered)
        y.append(label)
        
    print(f"Saved generated images to '{save_dir}/'")
    return np.array(X), np.array(y)