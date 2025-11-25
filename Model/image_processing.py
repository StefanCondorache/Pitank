import cv2              #type: ignore
import numpy as np

def fix_camera_orientation(frame):
    return cv2.rotate(frame, cv2.ROTATE_180)

def apply_custom_filter(img):
    """
    Sharpening filter (Sum of kernel = 1).
    Works on BGR images automatically (applies to all channels).
    """
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])
    
    filtered = cv2.filter2D(img, -1, kernel)
    return filtered

def preprocess_for_model(img_roi, target_size=(64, 64)):
    """
    Prepares an image crop for the AI (Resize -> Filter -> Normalize).
    KEEPS COLOR (3 Channels).
    """
    # Resize
    resized = cv2.resize(img_roi, target_size)
    
    # Filter (Sharpening)
    filtered = apply_custom_filter(resized)
    
    # Normalize 0-1
    normalized = filtered.astype('float32') / 255.0
    
    # Reshape for Keras: (Batch, Height, Width, Channels)
    final_input = normalized.reshape(1, target_size[0], target_size[1], 3)
    
    return final_input, filtered

def find_sign_region(frame):
    # (This function remains exactly the same as previous step)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_crop = None
    max_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 800:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if 0.7 < aspect_ratio < 1.3:
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0 and (float(area)/hull_area) > 0.8:
                    if area > max_area:
                        max_area = area
                        pad = 10
                        y1 = max(0, y-pad)
                        x1 = max(0, x-pad)
                        y2 = min(frame.shape[0], y+h+pad)
                        x2 = min(frame.shape[1], x+w+pad)
                        best_crop = frame[y1:y2, x1:x2]
    return best_crop

def decide_action(prediction_array):
    class_idx = int(np.argmax(prediction_array))
    confidence = np.max(prediction_array)
    
    actions = {
        0: "TURN LEFT",
        1: "TURN RIGHT",
        2: "TURN AROUND"
    }
    
    if confidence < 0.6:
        return "UNCERTAIN"
    return actions.get(class_idx, "UNKNOWN")