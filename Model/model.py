import os
import numpy as np
import cv2 #type: ignore
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical #type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from model import build_traffic_model 
from image_processing import apply_custom_filter 
from data_utils import load_plain_circles, create_synthetic_data

def build_traffic_model(input_shape=(64, 64, 1), num_classes=3):
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_pipeline():
    IMG_SIZE = (64, 64)
    BATCH_SIZE = 32
    EPOCHS = 20
    
    # UPDATE THESE PATHS
    PLAIN_CIRCLES = '/home/steppan/Documents/DataScienceAI/1PitankProject/circles'
    REAL_PHOTOS = '/home/steppan/Documents/DataScienceAI/1PitankProject/real_traffic_signs'

    # --- 1. Synthetic Data ---
    circles = load_plain_circles(PLAIN_CIRCLES, IMG_SIZE)
    X_synth, y_synth = create_synthetic_data(circles, num_samples=1500, img_size=IMG_SIZE)

    # Visualization of what we feed the model
    if len(X_synth) > 0:
        plt.figure(figsize=(8, 8))
        for i in range(min(9, len(X_synth))):
            plt.subplot(3, 3, i+1)
            plt.imshow(X_synth[i], cmap='gray')
            plt.title(f"Label: {y_synth[i]}")
            plt.axis('off')
        plt.suptitle("Generated Synthetic Data (Gray + Filtered)")
        plt.tight_layout()
        plt.show()
    
    # --- 2. Real Data ---
    print(f"Loading Real Data from {REAL_PHOTOS}...")
    X_real, y_real = [], []
    classes = ['0_left', '1_right', '2_turn_around']

    for idx, cls in enumerate(classes):
        class_folder = os.path.join(REAL_PHOTOS, cls)
        if not os.path.exists(class_folder): continue
            
        for f in os.listdir(class_folder):
            path = os.path.join(class_folder, f)
            if not os.path.isfile(path) or f.startswith('.'): continue
            
            frame = cv2.imread(path)
            if frame is None: continue
            
            # Real data pipeline
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
                
            resized = cv2.resize(gray, IMG_SIZE)
            filtered = apply_custom_filter(resized) # Same filter as synthetic
            
            X_real.append(filtered)
            y_real.append(idx)
            
    X_real, y_real = np.array(X_real), np.array(y_real)
    
    # --- 3. Merge & Train ---
    if len(X_real) > 0:
        X_final = np.concatenate((X_synth, X_real))
        y_final = np.concatenate((y_synth, y_real))
    else:
        print("Warning: Only using synthetic data.")
        X_final, y_final = X_synth, y_synth

    if len(X_final) == 0:
        print("Error: No data found.")
        return

    # Normalize & Reshape
    X_final = X_final.astype('float32') / 255.0
    X_final = X_final.reshape(-1, 64, 64, 1)
    y_final = to_categorical(y_final, num_classes=3)

    model = build_traffic_model(input_shape=(64, 64, 1), num_classes=3)
    
    print("Starting Training...")
    model.fit(X_final, y_final, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
    model.save('traffic_classifier.keras')
    print("Training complete. Model saved.")