import os
import numpy as np
import cv2 # type: ignore

from tensorflow.keras.utils import to_categorical # type: ignore
from model import build_traffic_model
from image_processing import apply_custom_filter, find_sign_region
from data_utils import load_plain_circles, create_synthetic_data, apply_occlusion

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 20

def train_pipeline(plain_circles_path, real_data_path):
    # Generate synthetic data
    circles = load_plain_circles(plain_circles_path, IMG_SIZE)
    X_synth, y_synth = create_synthetic_data(circles, num_samples=1500, img_size=IMG_SIZE)

    # Load and process real data
    print(f"Loading Real Data from {real_data_path}...")
    X_real, y_real = [], []
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
    X_real, y_real = np.array(X_real), np.array(y_real)

    # Merge datasets
    print(f"Dataset: {len(X_synth)} Synthetic, {len(X_real)} Real images.")
    if len(X_synth) == 0 and len(X_real) == 0:
        print("No data to train!")
        return
    if len(X_real) > 0:
        X_final = np.concatenate((X_synth, X_real))
        y_final = np.concatenate((y_synth, y_real))
    else:
        X_final, y_final = X_synth, y_synth

    X_final = X_final.astype('float32') / 255.0
    X_final = X_final.reshape(-1, 64, 64, 1)
    y_final = to_categorical(y_final, num_classes=3)

    model = build_traffic_model(input_shape=(64, 64, 1), num_classes=3)
    print("Starting Training...")
    model.fit(X_final, y_final, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
    model.save('traffic_classifier.h5')
    print("Training complete. Model saved as traffic_classifier.h5.")

if __name__ == "__main__":
    PLAIN_CIRCLES = '../circles'
    REAL_PHOTOS = '../real_traffic_signs'
    train_pipeline(PLAIN_CIRCLES, REAL_PHOTOS)