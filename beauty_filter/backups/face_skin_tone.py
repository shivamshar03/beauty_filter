import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.cluster import KMeans
import mediapipe as mp
from PIL import Image, ImageTk

class SkinToneApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Skin Tone Analyzer (Capture Mode)")

        # Mediapipe face detection
        self.mp_face = mp.solutions.face_detection
        self.face_detection = self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

        self.cap = cv2.VideoCapture(0)
        self.current_frame = None
        self.skin_tone = None

        # UI
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        self.skin_tone_label = tk.Label(self.root, text="Skin Tone: Unknown", font=("Arial", 14))
        self.skin_tone_label.pack(pady=5)

        self.color_box = tk.Label(self.root, text="   ", width=20, height=2, bg="white")
        self.color_box.pack(pady=10)

        self.capture_button = tk.Button(self.root, text="Capture Face & Analyze", command=self.capture_face)
        self.capture_button.pack(pady=10)

        self.update_video()

    def update_video(self):
        """Show live webcam feed"""
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Draw face bounding box for clarity
            results = self.face_detection.process(frame_rgb)
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, width, height = (
                        int(bbox.xmin * w), int(bbox.ymin * h),
                        int(bbox.width * w), int(bbox.height * h)
                    )
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        self.root.after(50, self.update_video)

    def capture_face(self):
        """Analyze skin tone only when capture button is pressed"""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No image available for analysis")
            return

        try:
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame_rgb.shape

                    # Extract face region
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)

                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, w - x)
                    height = min(height, h - y)

                    face_region = frame_rgb[y:y + height, x:x + width]

                    if face_region.size > 0:
                        # Get dominant skin color using KMeans
                        face_pixels = face_region.reshape((-1, 3))
                        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                        kmeans.fit(face_pixels)

                        colors = kmeans.cluster_centers_
                        labels = kmeans.labels_

                        unique_labels, counts = np.unique(labels, return_counts=True)
                        dominant_color_idx = unique_labels[np.argmax(counts)]
                        dominant_color = colors[dominant_color_idx]

                        # Convert to int + hex
                        dominant_color_int = tuple(dominant_color.astype(int))
                        hex_color = '#%02x%02x%02x' % dominant_color_int

                        # Classify skin tone
                        self.skin_tone = self.classify_skin_tone(dominant_color_int)

                        # Update UI
                        self.skin_tone_label.config(text=f"Skin Tone: {self.skin_tone}")
                        self.color_box.config(bg=hex_color)
                        break
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")

    def classify_skin_tone(self, rgb):
        """Classify skin tone into depth + undertone categories"""
        r, g, b = rgb
        brightness = (r + g + b) / 3

        # Depth (lightness/darkness)
        if brightness < 60:
            depth = "Deep"
        elif brightness < 120:
            depth = "Tan/Medium"
        elif brightness < 180:
            depth = "Light-Medium"
        else:
            depth = "Fair/Light"

        # Undertone
        if r > g and r > b:
            undertone = "Warm"
        elif b > r and b > g:
            undertone = "Cool"
        else:
            undertone = "Neutral"

        return f"{depth} ({undertone})"

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = SkinToneApp(root)
    root.mainloop()
