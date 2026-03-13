import cv2
import numpy as np
import os
import mediapipe.python.solutions.hands as mp_hands_module
import mediapipe.python.solutions.drawing_utils as mp_drawing_module

# MediaPipe Aliases
mp_hands = mp_hands_module
mp_drawing = mp_drawing_module

# --- CONFIGURATION ---
DATA_PATH = os.path.join('dataset', 'asl_alphabet_train', 'asl_alphabet_train') 

# Updated to match your folder structure exactly
ACTIONS = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                    'del', 'nothing', 'space'])

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

def extract_landmarks(results):
    # Extracts only the first hand detected (Standard for ASL Alphabets)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            return np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
    return np.zeros(21*3) # Return zeros if no hand found