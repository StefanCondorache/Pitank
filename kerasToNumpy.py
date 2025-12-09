import numpy as np
import cv2 #type: ignore
from tensorflow.keras.models import load_model #type: ignore

MODEL_PATH = 'traffic_classifier.keras' 

def extract_weights_from_file(model_path):
    """
    Loads a Keras model file and returns a dictionary of weights.
    """
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    weights = {}
    
    # Extracting based on your specific architecture order:
    # Layers: [0]Conv, [1]Pool, [2]Drop, [3]Conv, [4]Pool, [5]Drop, [6]Conv, [7]Pool, [8]Drop, [9]Flat, [10]Dense, [11]Drop, [12]Output
    
    # Block 1 (Layer 0)
    w, b = model.layers[0].get_weights()
    weights['c1_w'], weights['c1_b'] = w, b
    
    # Block 2 (Layer 3)
    w, b = model.layers[3].get_weights()
    weights['c2_w'], weights['c2_b'] = w, b
    
    # Block 3 (Layer 6)
    w, b = model.layers[6].get_weights()
    weights['c3_w'], weights['c3_b'] = w, b
    
    # Dense 1 (Layer 10)
    w, b = model.layers[10].get_weights()
    weights['d1_w'], weights['d1_b'] = w, b
    
    # Output Dense (Layer 12)
    w, b = model.layers[12].get_weights()
    weights['d2_w'], weights['d2_b'] = w, b
    
    print("Weights extracted successfully.")
    return model, weights

class NumPyTrafficModel:
    def __init__(self, weights_dict):
        self.w = weights_dict

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return np.round(e_x / e_x.sum(axis=-1, keepdims=True), 4)

    def conv2d_same(self, x, W, b):
        """
        Naive implementation of Conv2D with padding='same'.
        Input: (H, W, C_in) | Weights: (3, 3, C_in, C_out)
        """
        H, W_dim, C_in = x.shape
        Kh, Kw, _, C_out = W.shape
        
        pad = Kh // 2
        x_padded = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
        
        output = np.zeros((H, W_dim, C_out))
        
        # Convolution Loop
        for h in range(H):
            for w in range(W_dim):
                # Extract 3x3 patch
                patch = x_padded[h:h+Kh, w:w+Kw, :]
                # Apply filters
                for c in range(C_out):
                    output[h, w, c] = np.sum(patch * W[:, :, :, c]) + b[c]
        return output

    def maxpool_2x2(self, x):
        """
        Max Pooling with 2x2 window.
        """
        H, W_dim, C = x.shape
        H_out, W_out = H // 2, W_dim // 2
        
        # Fast reshape method for pooling
        x_reshaped = x[:H_out*2, :W_out*2, :].reshape(H_out, 2, W_out, 2, C)
        return x_reshaped.max(axis=(1, 3))

    def predict(self, image):
        """
        Runs the forward pass.
        image: shape (64, 64, 3)
        """
        # Block 1
        x = self.conv2d_same(image, self.w['c1_w'], self.w['c1_b'])
        x = self.relu(x)
        x = self.maxpool_2x2(x)
        
        # Block 2
        x = self.conv2d_same(x, self.w['c2_w'], self.w['c2_b'])
        x = self.relu(x)
        x = self.maxpool_2x2(x)

        # Block 3
        x = self.conv2d_same(x, self.w['c3_w'], self.w['c3_b'])
        x = self.relu(x)
        x = self.maxpool_2x2(x)
        
        # Flatten
        x = x.flatten()
        
        # Dense 1
        x = np.dot(x, self.w['d1_w']) + self.w['d1_b']
        x = self.relu(x)
        
        # Output Dense
        x = np.dot(x, self.w['d2_w']) + self.w['d2_b']
        x = self.softmax(x)
        
        return x


if __name__ == "__main__":
    # 1. Load Keras model and get weights
    keras_model, weights_dict = extract_weights_from_file(MODEL_PATH)
    
    # 2. Instantiate Custom NumPy Model
    numpy_model = NumPyTrafficModel(weights_dict)
    
    # 3. Create a dummy image for testing (normalized 0-1)
    image1 = cv2.imread("trainingData/extracted_crops/0_left/a1_crop_1.png")
    image1 = cv2.resize(image1, (64,64))

    image2 = cv2.imread("trainingData/extracted_crops/1_right/a1_crop_3.png")
    image2 = cv2.resize(image2, (64,64))

    image3 = cv2.imread("trainingData/extracted_crops/2_turn_around/a1_crop_2.png")
    image3 = cv2.resize(image3, (64,64))

    images = [image1, image2, image3]

    print("\n--- Running Inference Comparison ---")
    for image in images:
        keras_pred = keras_model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
        numpy_pred = numpy_model.predict(image)
    
        print(f"Keras Prediction: {keras_pred}")
        print(f"NumPy Prediction: {numpy_pred}")
