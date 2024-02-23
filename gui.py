import tkinter as tk
import cv2
from tkinter import messagebox
from threading import Thread, Event
from keras.models import load_model
import numpy as np

class DrowsinessDetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Real-time Drowsiness Detection")

        self.label = tk.Label(self, text="Real-time Drowsiness Detection", font=("Helvetica", 16))
        self.label.pack(pady=10)

        self.start_button = tk.Button(self, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=5)

        self.stop_button = tk.Button(self, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        self.cap = None
        self.thread = None
        self.is_running = False
        self.stop_event = Event()  # Event for signaling the stop of the thread
        self.warning_shown = False  # Flag to track if warning has been shown

        self.model = load_model(r'C:\Users\hp\OneDrive\Desktop\drowness of eye detection\drowness of eye detection\dataset\best_model.h5')

        # Bind the destroy event of the window to a method
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def start_detection(self):
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.is_running = True

        self.cap = cv2.VideoCapture(0)  # Use 0 for default webcam
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open webcam")
            self.stop_detection()
            return

        self.thread = Thread(target=self.detect_drowsiness)
        self.thread.start()

    def stop_detection(self):
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.is_running = False
        self.stop_event.set()  # Set the event to signal the thread to stop

        if self.thread:
            self.thread.join()

        if self.cap:
            self.cap.release()

    def detect_drowsiness(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture frame")
                break

            # Resize frame to match model input shape (if needed)
            frame = cv2.resize(frame, (224, 224))

            # Preprocess frame (if needed)
            frame = frame / 255.0  # Normalize pixel values

            # Perform prediction
            prediction = self.model.predict(np.expand_dims(frame, axis=0))

            # Assuming binary classification (0: not drowsy, 1: drowsy)
            if np.any(prediction > 0.5) and not self.warning_shown:
                messagebox.showwarning("Warning", "Driver appears drowsy!")
                self.warning_shown = True

            # Display the frame (optional)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Check if the stop event is set
            if self.stop_event.is_set():
                break

        cv2.destroyAllWindows()

    def on_close(self):
        if self.is_running:
            self.stop_detection()
        self.destroy()

if __name__ == "__main__":
    app = DrowsinessDetectionApp()
    app.mainloop()
