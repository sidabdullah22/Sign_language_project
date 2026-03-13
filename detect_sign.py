import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from utils import mediapipe_detection, extract_landmarks, mp_holistic

class SignLanguageApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sign Language Translator")
        self.geometry("900x700")

        # Logic
        self.is_translating = False
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # UI
        self.video_label = ctk.CTkLabel(self, text="", width=640, height=480, fg_color="black")
        self.video_label.pack(pady=20)
        
        # Add your Start/Pause/Complete 

if __name__ == "__main__":
    app = SignLanguageApp()
    app.mainloop()