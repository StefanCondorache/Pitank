import os
import numpy as np
import cv2                                                                              #type: ignore
from tensorflow.keras.models import Sequential, load_model                              #type: ignore
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten       #type: ignore
from tensorflow.keras.optimizers import Adam                                            #type: ignore  
from tensorflow.keras.utils import to_categorical                                       #type: ignore

from image_processing import apply_custom_filter
from data_utils import visualize_data

def build_traffic_model(input_shape=(64, 64, 3), num_classes=3):
    """
    CNN architecture for 3-class Traffic Sign Recognition.
    """
    model = Sequential()
    
    # Block 1
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2)) 
    
    # Block 2
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Block 3
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    # Classification
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def load_dataset(root_folder, target_size=(64, 64)):
    """
    Scans root_folder for subfolders: 0_left, 1_right, 2_turn_around.
    Returns: X (Normalized 0-1), y (One-Hot Encoded)
    """
    X = []
    y = []
    
    # Map folder names to class IDs
    class_map = { '0_left': 0, '1_right': 1, '2_turn_around': 2}
    
    print(f"Loading data from: {root_folder}")
    
    if not os.path.exists(root_folder):
        print(f"  [!] Folder not found: {root_folder}")
        return np.array([]), np.array([])

    for folder_name, label_id in class_map.items():
        folder_path = os.path.join(root_folder, folder_name)
        
        if not os.path.exists(folder_path):
            continue
            
        files = os.listdir(folder_path)
        print(f"  - Found {len(files)} images in {folder_name}")
        
        for f in files:
            file_path = os.path.join(folder_path, f)
            img = cv2.imread(file_path)
            
            if img is None: continue
            
            # 1. Resize
            resized = cv2.resize(img, target_size)
            
            # 2. Filter (Sharpening) - Keep Color!
            filtered = apply_custom_filter(resized)
            
            # 3. Normalize (0-1)
            normalized = filtered.astype('float32') / 255.0
            
            X.append(normalized)
            y.append(label_id)
            
    if len(X) == 0:
        return np.array([]), np.array([])
        
    # Reshape X for Keras (N, 64, 64, 3)
    X = np.array(X)
    X = X.reshape(-1, target_size[0], target_size[1], 3)
    
    # Encode y (0 -> [1,0,0])
    y = to_categorical(np.array(y), num_classes=3)
    
    return X, y

def preview_dataset(X, y_one_hot):
    """ 
    Helper to convert data back to format needed for visualize_data 
    (Float 0-1 -> Int 0-255, OneHot -> Label Index)
    """
    if len(X) == 0: return
    
    # Convert One-Hot back to Index (e.g. [0,1,0] -> 1)
    y_indices = np.argmax(y_one_hot, axis=1)
    
    # Convert Normalized Float back to Uint8 Image
    X_uint8 = (X * 255).astype('uint8')
    
    # Use the visualizer from data_utils
    visualize_data(X_uint8, y_indices, samples=15)

def train_pipeline():
    BATCH_SIZE = 32
    MODEL_PATH = 'traffic_classifier.keras'
    
    # 1. Load Existing Model OR Build New
    if os.path.exists(MODEL_PATH):
        print(f"\n>> Found existing model: {MODEL_PATH}. Loading...")
        model = load_model(MODEL_PATH)
    else:
        print("\n>> No existing model found. Building new model...")
        model = build_traffic_model(input_shape=(64, 64, 3), num_classes=3)

    # ==========================================
    # STAGE 1: SYNTHETIC (Perfect Geometry)
    # ==========================================
    print("\n--- STAGE 1: Training on SYNTHETIC SIGNS ---")
    X2, y2 = load_dataset('trainingData/synthetic')
    
    if len(X2) > 0:
        preview_dataset(X2, y2) # Visual Check
        model.fit(X2, y2, epochs=10, batch_size=BATCH_SIZE, validation_split=0.2)
        model.save(MODEL_PATH)
        print(">> Stage 1 Complete. Saved.")
    else:
        print(">> Skipping Stage 1 (Run data_utils.py first).")

    # ==========================================
    # STAGE 2: EXTRACTED CROPS (Real World Adaptation)
    # ==========================================
    print("\n--- STAGE 2: Fine-Tuning on REAL CROPS ---")
    X3, y3 = load_dataset('trainingData/extracted_crops')
    
    if len(X3) > 0:
        preview_dataset(X3, y3) # Visual Check
        # Train slightly longer on real data to adapt weights
        model.fit(X3, y3, epochs=15, batch_size=BATCH_SIZE, validation_split=0.2)
        model.save(MODEL_PATH)
        print(">> Stage 2 Complete. Final Model Saved.")
    else:
        print(">> Skipping Stage 2 (No real data found).")
    
if __name__ == "__main__":
    train_pipeline()