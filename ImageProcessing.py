import cv2 # type: ignore
import numpy as np

def fix_camera_orientation(frame):
    return cv2.rotate(frame, cv2.ROTATE_180)

def apply_custom_filter(img):
    """
    Applies the specific matrix requested to highlight edges/features.
    Matrix: [[-1, 5, -1], [5, -1, 5], [-1, 5, -1]]
    """

    kernel = np.array([[-1, 5, -1], 
                       [ 5, -1, 5], 
                       [-1, 5, -1]])
    
    filtered = cv2.filter2D(img, -1, kernel)
    return filtered

def find_sign_region(frame):
    """
    Finds the sign in the large frame and returns the crop.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_crop = None
    max_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            
            if 0.7 < aspect_ratio < 1.3:
                if area > max_area:
                    max_area = area
                    best_crop = frame[y:y+h, x:x+w]
                    
    return best_crop

def preprocess_for_model(img_roi, target_size=(64, 64)):
    """
    The final pipeline that prepares data for the AI.
    Used by BOTH training (Real Data) and Live Inference.
    """
    # 1. Grayscale
    if len(img_roi.shape) == 3:
        gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_roi

    # 2. Resize
    resized = cv2.resize(gray, target_size)
    
    # 3. Apply the CUSTOM FILTER (The "Useful Part")
    filtered = apply_custom_filter(resized)
    
    # 4. Normalize
    normalized = filtered.astype('float32') / 255.0
    final_input = normalized.reshape(1, target_size[0], target_size[1], 1)
    
    return final_input, filtered

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