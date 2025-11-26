import os
from typing import final
import numpy as np
import cv2                                          #type: ignore
import random

def draw_thick_arrow(img, pt1, pt2, color=(255, 255, 255), thickness=8, tip_size=20, sign_type='U_turn'):
    """ 
    Draws a very thick arrow with a wide, bold tip to match the reference. 
    """
    # Draw the shaft
    cv2.line(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
    
    # Calculate angle of the line
    angle = np.arctan2(pt1[1] - pt2[1], pt1[0] - pt2[0])
    arrow_angle = np.pi / 3.5  
    
    p1 = (int(pt2[0] + tip_size * np.cos(angle + arrow_angle)),
          int(pt2[1] + tip_size * np.sin(angle + arrow_angle)))
    p2 = (int(pt2[0] + tip_size * np.cos(angle - arrow_angle)),
          int(pt2[1] + tip_size * np.sin(angle - arrow_angle)))
    if sign_type == 'right':
        triangle_cnt = np.array([(pt2[0]+7, pt2[1]), p1, p2])
    elif sign_type == 'left':
        triangle_cnt = np.array([(pt2[0]-7, pt2[1]), p1, p2])
    else:
        triangle_cnt = np.array([(pt2[0], pt2[1]+7), p1, p2])
    cv2.fillPoly(img, [triangle_cnt], color, lineType=cv2.LINE_AA)

def create_base_sign(label, img_size=(64, 64)):
    img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    
    center = (img_size[0]//2, img_size[1]//2)
    radius = (img_size[0]//2) - 3
    
    cv2.circle(img, center, radius, (255, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, center, radius, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    
    white = (255, 255, 255)

    thick = 8          
    arrow_tip = 15     
    
    if label == 0: # Turn Left
        draw_thick_arrow(img, (48, 32), (16, 32), white, thickness=thick, tip_size=arrow_tip, sign_type='left')
        
    elif label == 1: # Turn Right
        draw_thick_arrow(img, (16, 32), (48, 32), white, thickness=thick, tip_size=arrow_tip, sign_type='right')
        
    elif label == 2: # U-Turn
        # 1. Right straight line (going up)
        cv2.line(img, (44, 40), (44, 28), white, thick, lineType=cv2.LINE_AA)
        
        # 2. Top semi-circle arc
        cv2.ellipse(img, (32, 28), (12, 12), 0, 180, 360, white, thick, lineType=cv2.LINE_AA)
        
        # 3. Left arrow (going down)
        draw_thick_arrow(img, (20, 28), (20, 40), white, thickness=thick, tip_size=arrow_tip)
        
    return img

def apply_occlusion(img):
    """ Covers 1/4 of the image with noise. """
    h, w = img.shape[:2]
    occ_img = img.copy()
    
    type_idx = random.randint(0, 4)
    noise = np.random.randint(0, 255, (h, w, 3), dtype='uint8')
    
    if type_idx == 0: # Right
        occ_img[:, 3*w//4:] = noise[:, 3*w//4:]
    elif type_idx == 1: # Left
        occ_img[:, :w//4] = noise[:, :w//4]
    elif type_idx == 2: # Top
        occ_img[:h//4, :] = noise[:h//4, :]
    elif type_idx == 3: # Bottom
        occ_img[3*h//4:, :] = noise[3*h//4:, :]
    elif type_idx == 4: # Center
        y1, y2 = h//4, 3*h//4
        x1, x2 = w//4, 3*w//4
        occ_img[y1:y2, x1:x2] = noise[y1:y2, x1:x2]
        
    return occ_img

def apply_rotation(img):
    """ Rotates image around its center by the given angle. """
    angle = random.uniform(-15, 15)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def create_synthetic_data(img_size=(64,64)):
    print("Generating Synthetic Data (Bold Arrows)...")
    X = []
    y = []
    classes = [0, 1, 2] 
    
    X.append(create_base_sign(0, img_size))
    y.append(0)
    X.append(create_base_sign(1, img_size))
    y.append(1)
    X.append(create_base_sign(2, img_size))
    y.append(2)

    blur_levels = [(0,0), (3,3), (5,5), (7,7), (9,9)] 
    REPETITIONS = 20 

    for _ in range(REPETITIONS):
        for label in classes:
            base = create_base_sign(label, img_size)
            
            for ksize in blur_levels:
                if ksize == (0, 0):
                    blurred = base.copy()
                else:
                    blurred = cv2.GaussianBlur(base, ksize, 0)
                
                rotated = apply_rotation(blurred) if label in [0,1] else blurred.copy()
                X.append(rotated)
                y.append(label)

                mix = apply_occlusion(rotated)
                X.append(mix)
                y.append(label) 

                noise = apply_occlusion(blurred)
                X.append(noise)
                y.append(label)


                
    return np.array(X), np.array(y)

def visualize_data(X, y, samples=20):
    print(f"Total images generated: {len(X)}")
    print("Displaying preview...")
    images_to_show = []
    indices = np.linspace(0, len(X)-1, samples, dtype=int)
    
    for i in indices:
        img = X[i].copy() 
        label = y[i]
        cv2.rectangle(img, (0,0), (40, 20), (0,0,0), -1)
        cv2.putText(img, f"L:{label}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        images_to_show.append(img)
    
    rows = []
    cols_per_row = 5
    row_imgs = []
    for i, img in enumerate(images_to_show):
        row_imgs.append(img)
        if len(row_imgs) == cols_per_row:
            rows.append(cv2.hconcat(row_imgs))
            row_imgs = []
    if row_imgs:
        while len(row_imgs) < cols_per_row: row_imgs.append(np.zeros_like(images_to_show[0]))
        rows.append(cv2.hconcat(row_imgs))

    if rows:
        final_grid = cv2.vconcat(rows)
        final_grid = cv2.resize(final_grid, (0,0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Bold Arrow Reference Match", final_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def save_data(X, y, base_dir="./trainingData/synthetic/"):
    """
    Saves images into subfolders:
    trainingData/synthetic/0_left
    trainingData/synthetic/1_right
    trainingData/synthetic/2_turn_around
    """
    
    # Map IDs to Folder Names
    class_map = {
        0: "0_left",
        1: "1_right",
        2: "2_turn_around"
    }

    # Create directories if they don't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    for key in class_map:
        path = os.path.join(base_dir, class_map[key])
        if not os.path.exists(path):
            os.makedirs(path)

    print(f"Saving {len(X)} images to '{base_dir}'...")
    
    count = 0
    for img, label in zip(X, y):
        folder_name = class_map[label]
        # Create a unique filename
        filename = f"syn_{count}.png"
        save_path = os.path.join(base_dir, folder_name, filename)
        
        cv2.imwrite(save_path, img)
        count += 1
        
    print("Save complete.")

if __name__ == "__main__":
    X_syn, y_syn = create_synthetic_data()
    visualize_data(X_syn, y_syn, samples=20)
    save_data(X_syn, y_syn)