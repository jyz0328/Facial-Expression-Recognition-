#!/usr/bin/env python
import os
import cv2
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# -------------------------------
# Parameters
# -------------------------------
IMG_SIZE = 48           # Final image size (48x48, matching FER2013)
NUM_CLASSES = 8         # All 8 AffectNet emotion classes (0-7, including contempt)
CSV_PATH = 'affectnet.csv'  # Path to your AffectNet CSV file (generated using the updated affectnetCSV.py)
MARGIN = 0.15           # Use 15% margin to mimic official boundary expansion

# -------------------------------
# Utility: Crop Face Using Landmarks
# -------------------------------
def crop_face_with_landmarks(img, landmarks, margin=MARGIN):
    """
    Compute the bounding box from landmarks, expand it by a margin, and crop the image.
    """
    landmarks = np.array(landmarks)
    x_min, y_min = np.min(landmarks, axis=0)
    x_max, y_max = np.max(landmarks, axis=0)
    w = x_max - x_min
    h = y_max - y_min
    x_margin = int(w * margin)
    y_margin = int(h * margin)
    x1 = int(max(0, x_min - x_margin))
    y1 = int(max(0, y_min - y_margin))
    x2 = int(min(img.shape[1], x_max + x_margin))
    y2 = int(min(img.shape[0], y_max + y_margin))
    return img[y1:y2, x1:x2]

# -------------------------------
# Load and Preprocess Data
# -------------------------------
def load_and_preprocess_data(csv_path, img_size=IMG_SIZE, num_classes=NUM_CLASSES):
    """
    Loads AffectNet data from CSV, crops the face region using landmarks,
    converts the image to grayscale, resizes it, normalizes pixel values,
    and one-hot encodes the label.
    """
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    
    for idx, row in df.iterrows():
        img_path = row['image_path']
        label = int(row['expression'])
        landmark_str = row['landmarks']
        
        # Load image from file (supports JPEG, PNG, etc.)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image at {img_path}")
            continue
        
        # Parse landmarks using JSON
        try:
            landmarks = json.loads(landmark_str)
        except Exception as e:
            print(f"Error parsing landmarks for {img_path}: {e}")
            continue
        
        # Convert landmarks to a numpy array and reshape if necessary
        landmarks = np.array(landmarks)
        if landmarks.ndim == 1 and landmarks.size % 2 == 0:
            landmarks = landmarks.reshape(-1, 2)
        
        # Crop image using landmarks
        cropped_img = crop_face_with_landmarks(img, landmarks)
        if cropped_img.size == 0:
            print(f"Warning: Empty crop for image {img_path}")
            continue
        
        # Convert to grayscale
        gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        
        # Resize to (img_size, img_size)
        resized_img = cv2.resize(gray_img, (img_size, img_size))
        
        # Normalize pixel values to [0, 1]
        norm_img = resized_img.astype('float32') / 255.0
        
        # Expand dimensions to add the channel (resulting in shape: img_size x img_size x 1)
        final_img = np.expand_dims(norm_img, axis=-1)
        
        images.append(final_img)
        labels.append(label)
    
    X = np.array(images)
    y = to_categorical(labels, num_classes)
    return X, y

# -------------------------------
# Split and Save Data (Training & Test only)
# -------------------------------
def split_and_save_data(X, y, output_dir='preprocessed_data'):
    """
    Splits data into training (80%) and test (20%) sets,
    then saves the arrays to the specified directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    print("Data saved to directory:", output_dir)
    print("Training samples:", X_train.shape[0])
    print("Test samples:", X_test.shape[0])

# -------------------------------
# Main Execution
# -------------------------------
def main():
    X, y = load_and_preprocess_data(CSV_PATH)
    print("Total samples loaded:", X.shape[0])
    split_and_save_data(X, y)

if __name__ == '__main__':
    main()
