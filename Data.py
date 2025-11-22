import cv2 #type: ignore
import os
import random

# Define paths
INPUT_CIRCLES_DIR = './circles/' 
OUTPUT_DIR = './synthetic_traffic_signs/'

classes = ['0_left', '1_right', '2_turn_around']

for c in classes:
    os.makedirs(os.path.join(OUTPUT_DIR, c), exist_ok=True)

def load_plain_circles(directory=INPUT_CIRCLES_DIR):
    plain_images = []
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} not found.")
        return []
        
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg')):
            img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                plain_images.append(cv2.resize(img, (64, 64)))
    return plain_images

def create_synthetic_data(circles, num_samples=1000):
    if not circles:
        print("No source circles found to generate data!")
        return

    count = 0
    print(f"Generating {num_samples} images (64x64) for 3 classes...")

    folder = ''
    
    for _ in range(num_samples):
        base = random.choice(circles).copy()
        label_idx = random.randint(0, 2)
        
        if label_idx == 0: # LEFT ARROW
            cv2.arrowedLine(base, (54, 32), (10, 32), (0), 4, tipLength=0.3)
            folder = classes[0]
            
        elif label_idx == 1: # RIGHT ARROW
            cv2.arrowedLine(base, (10, 32), (54, 32), (0), 4, tipLength=0.3)
            folder = classes[1]
            
        elif label_idx == 2: # TURN AROUND (n-shape)
            cv2.arrowedLine(base, (24, 28), (24, 52), (0), 4, tipLength=0.3)
            cv2.line(base, (40, 52), (40, 28), (0), 4)
            cv2.ellipse(base, (32, 28), (8, 8), 0, 180, 360, (0), 4)
            
            folder = classes[2]
            
        cv2.imwrite(f"{OUTPUT_DIR}/{folder}/sign_{count}.png", base)
        count += 1

    print(f"Success! Generated {count} synthetic traffic signs in '{OUTPUT_DIR}'.")