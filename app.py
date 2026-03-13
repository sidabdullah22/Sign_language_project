import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from utils import mediapipe_detection, extract_landmarks, mp_hands, draw_landmarks, ACTIONS

class SignLanguageApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Sign Language Alphabet Translator")
        self.geometry("1000x800")

        # 1. Load the AI Model
        try:
            self.model = load_model('sign_language_model.keras')
            print("AI Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

        # Logic Flags
        self.is_running = False
        self.cap = None
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

        # UI Elements
        self.video_label = ctk.CTkLabel(self, text="System Offline", width=640, height=480, fg_color="black")
        self.video_label.pack(pady=20)

        self.prediction_label = ctk.CTkLabel(self, text="Translation: ---", font=("Arial", 32, "bold"), text_color="yellow")
        self.prediction_label.pack(pady=10)

        # Buttons
        self.btn_frame = ctk.CTkFrame(self)
        self.btn_frame.pack(pady=20)

        ctk.CTkButton(self.btn_frame, text="Start Translator", fg_color="green", command=self.start_logic).grid(row=0, column=0, padx=10)
        ctk.CTkButton(self.btn_frame, text="Pause", fg_color="orange", command=self.pause_logic).grid(row=0, column=1, padx=10)
        ctk.CTkButton(self.btn_frame, text="Quit", fg_color="red", command=self.quit_app).grid(row=0, column=2, padx=10)

    def start_logic(self):
        self.is_running = True
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def pause_logic(self):
        self.is_running = False

    def quit_app(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.destroy()

    def update_frame(self):
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, self.hands)
                draw_landmarks(image, results)

                # --- PREDICTION LOGIC ---
                landmarks = extract_landmarks(results)
                
                # Only predict if landmarks are NOT all zeros (hand is visible)
                if not np.all(landmarks == 0):
                    # Reshape to (1, 63) as expected by the model
                    prediction = self.model.predict(np.expand_dims(landmarks, axis=0), verbose=0)
                    char_index = np.argmax(prediction)
                    confidence = prediction[0][char_index]

                    if confidence > 0.7: # Only show if model is >70% sure
                        detected_char = ACTIONS[char_index]
                        self.prediction_label.configure(text=f"Translation: {detected_char} ({confidence*100:.1f}%)")
                else:
                    self.prediction_label.configure(text="Translation: ---")

                # Update GUI Image
                cv2_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2_img)
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
                self.video_label.configure(image=ctk_img, text="")
                self.video_label.image = ctk_img
            
            self.after(10, self.update_frame)

if __name__ == "__main__":
    app = SignLanguageApp()
    app.mainloop()