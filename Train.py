from tensorflow.keras.utils import to_categorical # type: ignore
from Model import build_traffic_model 
from ImageProcessing import apply_custom_filter, find_sign_region
import cv2 # type: ignore
import numpy as np
import os
import random

# Configuration
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 20

# ==========================================
#           Load Plain Circles
# ==========================================
def load_plain_circles(directory):

    print(f"Loading base circles from {directory}...")
    circles = []
    if os.path.exists(directory):
        for f in os.listdir(directory):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(directory, f)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, IMG_SIZE)
                    circles.append(img)
    return circles

# ==========================================
#      Create Synthetic & Altered Data
# ==========================================
def apply_occlusion(img):
    """
    Covers 1/4 or 1/3 of the image (Top, Bottom, Left, Right, Corners).
    """
    h, w = img.shape
    occ_img = img.copy()
    
    type_idx = random.randint(0, 5) 
    
    if type_idx == 0: return occ_img
    
    block_h = h // 3
    block_w = w // 3
    
    noise = np.random.randint(0, 255, (h, w), dtype='uint8')
    
    if type_idx == 1: # Right
        occ_img[:, w-block_w:] = noise[:, w-block_w:]
    elif type_idx == 2: # Left
        occ_img[:, :block_w] = noise[:, :block_w]
    elif type_idx == 3: # Top
        occ_img[:block_h, :] = noise[:block_h, :]
    elif type_idx == 4: # Bottom
        occ_img[h-block_h:, :] = noise[h-block_h:, :]
    elif type_idx == 5: # Random Corner
        cx = random.choice([0, w-block_w])
        cy = random.choice([0, h-block_h])
        occ_img[cy:cy+block_h, cx:cx+block_w] = noise[cy:cy+block_h, cx:cx+block_w]
        
    return occ_img

def create_synthetic_data(base_circles, num_samples=1000):
    """
    Generates signs, applies Matrix, Fills Whites, Occlusions.
    """
    print("Generating Synthetic Data with Custom Filters...")
    X = []
    y = []
    
    if not base_circles:
        print("No base circles provided!")
        return np.array(X), np.array(y)

    for _ in range(num_samples):
        # 1. Start with a plain circle
        base = random.choice(base_circles).copy()
        label = random.randint(0, 2)
        
        # 2. Draw the Raw Sign Logic (Pre-Filter)
        if label == 0: # Left
            cv2.arrowedLine(base, (54, 32), (10, 32), (255), 4, tipLength=0.3)
        elif label == 1: # Right
            cv2.arrowedLine(base, (10, 32), (54, 32), (255), 4, tipLength=0.3)
        elif label == 2: # U-Turn
            cv2.arrowedLine(base, (24, 28), (24, 52), (255), 4, tipLength=0.3)
            cv2.line(base, (40, 52), (40, 28), (255), 4)
            cv2.ellipse(base, (32, 28), (8, 8), 0, 180, 360, (255), 4)

        # 3. Apply the Custom Matrix 
        filtered = apply_custom_filter(base)
        
        # 4. "Fill only the circle band with white"
        cv2.circle(filtered, (32, 32), 30, (255), 2)
        
        # 5. "Fill the sign in the middle with white"
        if label == 0: 
            cv2.arrowedLine(filtered, (54, 32), (10, 32), (255), 2, tipLength=0.3)
        elif label == 1:
            cv2.arrowedLine(filtered, (10, 32), (54, 32), (255), 2, tipLength=0.3)
        elif label == 2:
            cv2.arrowedLine(filtered, (24, 28), (24, 52), (255), 2, tipLength=0.3)
            cv2.line(filtered, (40, 52), (40, 28), (255), 2)
            cv2.ellipse(filtered, (32, 28), (8, 8), 0, 180, 360, (255), 2)

        # 6. "Fill only the blue background" -> Mask out background?
        _, contrast_img = cv2.threshold(filtered, 127, 255, cv2.THRESH_BINARY)
        
        # 7. Apply Occlusion / Noise
        final_img = apply_occlusion(contrast_img)
        
        X.append(final_img)
        y.append(label)
        
    return np.array(X), np.array(y)

# ==========================================
#         Hybrid Training Pipeline
# ==========================================
def train_hybrid_pipeline(plain_circles_path, real_data_path):
    
    # --- Step A: Generate Synthetic ---
    circles = load_plain_circles(plain_circles_path)
    X_synth, y_synth = create_synthetic_data(circles, num_samples=1500)
    
    # --- Step B: Load & Process Real Data ---
    print(f"Loading Real Data from {real_data_path}...")
    X_real = []
    y_real = []
    classes = ['0_left', '1_right', '2_turn_around']
    
    for idx, cls in enumerate(classes):
        folder = os.path.join(real_data_path, cls)
        if not os.path.exists(folder): continue
        
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            frame = cv2.imread(path) 
            if frame is None: continue
            
            crop = find_sign_region(frame)
            
            if crop is not None:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, IMG_SIZE)
                filtered = apply_custom_filter(resized)

                final_real = apply_occlusion(filtered)
                
                X_real.append(final_real)
                y_real.append(idx)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, IMG_SIZE)
                filtered = apply_custom_filter(resized)
                X_real.append(filtered)
                y_real.append(idx)

    X_real = np.array(X_real)
    y_real = np.array(y_real)
    
    # --- Step C: Merge & Train ---
    print(f"Stats: {len(X_synth)} Synthetic, {len(X_real)} Real images.")
    
    if len(X_synth) == 0 and len(X_real) == 0:
        print("No data to train!")
        return

    if len(X_real) > 0:
        X_final = np.concatenate((X_synth, X_real))
        y_final = np.concatenate((y_synth, y_real))
    else:
        X_final = X_synth
        y_final = y_synth
        
    X_final = X_final.astype('float32') / 255.0
    X_final = X_final.reshape(-1, 64, 64, 1)
    y_final = to_categorical(y_final, num_classes=3)
    
    model = build_traffic_model(input_shape=(64, 64, 1), num_classes=3)
    
    print("Starting Training...")
    model.fit(X_final, y_final, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
    model.save('traffic_classifier.h5')
    print("Done. Model saved.")

if __name__ == "__main__":
    PLAIN_CIRCLES = './circles'
    REAL_PHOTOS = './real_traffic_signs'
    
    train_hybrid_pipeline(PLAIN_CIRCLES, REAL_PHOTOS)