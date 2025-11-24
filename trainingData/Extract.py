import cv2 #type: ignore
import numpy as np
import os

def extract_sign_regions(img, min_area=1000, aspect_range=(0.7, 1.3)):
    """
    Extracts signs based on BLUE color detection.
    Much more robust than edge detection for specific traffic signs.
    """
    # 1. Convert to HSV Color Space (Better for color segmentation than BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. Define Blue Color Range
    # Hue (H): OpenCV uses 0-180. Blue is roughly 100-130.
    # Saturation (S): 0-255. High S means vivid color. Low S means gray/white.
    # Value (V): 0-255. High V means bright. Low V means dark.
    
    # Range: Allows for dark blue to bright blue, but rejects grey/white/black.
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    
    # 3. Create Mask
    # Returns a binary image: White where pixels are blue, Black elsewhere.
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # 4. Clean up noise (Optional but recommended)
    # Erode/Dilate removes tiny blue specs (noise) and fills small holes in the sign
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 5. Find Contours on the MASK (not the raw image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crops = []
    
    # Sort by area to process largest blobs first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filter small specs
        if area < min_area:
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        
        # Filter by Shape (Square-ish / Circular)
        if aspect_range[0] < aspect_ratio < aspect_range[1]:
            
            # Filter by Solidity (Is it a solid blob or a hollow C-shape?)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue
            solidity = float(area) / hull_area
            
            # Signs are solid (circle/square). Noise is often messy.
            if solidity > 0.8:
                # Add padding
                pad = 15
                h_img, w_img = img.shape[:2]
                y1 = max(0, y - pad)
                y2 = min(h_img, y + h + pad)
                x1 = max(0, x - pad)
                x2 = min(w_img, x + w + pad)
                
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)
                    
    return crops

def process_directory(input_dir='real_traffic_signs', output_dir='extracted_crops'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_exts = '.jpg'
    count = 0
    print(f"Scanning '{input_dir}' for images (Looking for BLUE objects)...")
    
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(valid_exts):
            file_path = os.path.join(input_dir, fname)
            img = cv2.imread(file_path)
            if img is None: continue
                
            crops = extract_sign_regions(img)
            
            if crops:
                print(f"  {fname}: Found {len(crops)} signs.")
                for i, crop in enumerate(crops):
                    base_name = os.path.splitext(fname)[0]
                    save_name = f"{base_name}_crop_{i}.png"
                    cv2.imwrite(os.path.join(output_dir, save_name), crop)
                    count += 1
            else:
                # Useful for debugging: prints if an image had NO blue blobs
                pass 

    print(f"\nDone. Extracted {count} signs to '{output_dir}/'")

if __name__ == "__main__":
    process_directory()