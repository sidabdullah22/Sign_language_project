import os
import numpy as np
import cv2
from utils import mediapipe_detection, extract_landmarks, mp_hands, ACTIONS

# The deep path from your tree command
DATA_PATH = os.path.join('dataset', 'asl_alphabet_train', 'asl_alphabet_train')

X = []
y = []

# Initialize MediaPipe
with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
    for idx, action in enumerate(ACTIONS):
        action_path = os.path.join(DATA_PATH, action)
        
        if not os.path.exists(action_path):
            print(f"Folder not found: {action_path}")
            continue

        print(f"Processing class: {action}...")
        
        # Get list of images and limit to 500 per class for manageability
        img_list = os.listdir(action_path)[:500] 
        
        for img_name in img_list:
            img_path = os.path.join(action_path, img_name)
            frame = cv2.imread(img_path)
            
            if frame is None: continue
            
            # Extract Landmarks
            _, results = mediapipe_detection(frame, hands)
            landmarks = extract_landmarks(results)
            
            # Only add if a hand was actually detected (not all zeros)
            if not np.all(landmarks == 0):
                X.append(landmarks)
                y.append(idx)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print(f"Total samples collected: {len(X)}")

# Save for Google Colab
np.save('X_data.npy', X)
np.save('y_data.npy', y)
print("Files saved: X_data.npy and y_data.npy")