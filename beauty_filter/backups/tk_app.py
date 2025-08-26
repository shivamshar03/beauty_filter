import cv2
import numpy as np
import mediapipe as mp
from sklearn.cluster import KMeans
import colorsys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import json
import glob
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class FaceFilterApp:
    def __init__(self):
        # Initialize MediaPipe Face Mesh and Face Detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Advanced Face Filter & Virtual Try-On with PNG Filters")
        self.root.geometry("1600x1000")

        # Variables
        self.current_frame = None
        self.skin_tone = None
        self.season_palette = None

        # Filter assets storage
        self.filter_assets = {
            'lipstick': [],
            'blush': [],
            'eyeshadow': [],
            'jewelry': [],
            'hairstyles': [],
            'dresses': [],
            'accessories': []
        }

        self.current_filters = {
            'lipstick': {'active': False, 'asset': None, 'color': (200, 50, 50)},
            'blush': {'active': False, 'asset': None, 'color': (255, 150, 150)},
            'eyeshadow': {'active': False, 'asset': None, 'color': (150, 100, 200)},
            'foundation': {'active': False, 'color': (220, 180, 150)},
            'jewelry': {'active': False, 'asset': None, 'type': 'earrings'},
            'hairstyle': {'active': False, 'asset': None, 'style': 'none'},
            'dress': {'active': False, 'asset': None, 'style': 'none'},
            'accessories': {'active': False, 'asset': None, 'type': 'none'}
        }

        # Color palettes for different skin tones and seasons
        self.color_palettes = {
            'light': {
                'spring': [(255, 182, 193), (255, 218, 185), (255, 240, 245), (173, 216, 230)],
                'summer': [(176, 196, 222), (221, 160, 221), (255, 182, 193), (240, 248, 255)],
                'autumn': [(210, 180, 140), (188, 143, 143), (255, 218, 185), (245, 222, 179)],
                'winter': [(220, 20, 60), (72, 61, 139), (25, 25, 112), (255, 255, 255)]
            },
            'medium': {
                'spring': [(255, 140, 0), (255, 165, 0), (255, 215, 0), (154, 205, 50)],
                'summer': [(70, 130, 180), (123, 104, 238), (186, 85, 211), (255, 192, 203)],
                'autumn': [(160, 82, 45), (205, 133, 63), (222, 184, 135), (244, 164, 96)],
                'winter': [(139, 0, 0), (75, 0, 130), (25, 25, 112), (255, 255, 255)]
            },
            'dark': {
                'spring': [(255, 69, 0), (255, 140, 0), (255, 215, 0), (50, 205, 50)],
                'summer': [(65, 105, 225), (138, 43, 226), (219, 112, 147), (255, 20, 147)],
                'autumn': [(139, 69, 19), (160, 82, 45), (205, 133, 63), (210, 180, 140)],
                'winter': [(128, 0, 0), (72, 61, 139), (0, 0, 139), (255, 255, 255)]
            }
        }
        self.load_filter_assets()
        self.setup_gui()

    def load_filter_assets(self):
        """Load PNG filter assets from directories"""
        filter_paths = {
            'lipstick': 'filters/makeup/lipstick/*.png',
            'blush': 'filters/makeup/blush/*.png',
            'eyeshadow': 'filters/makeup/eyeshadow/*.png',
            'jewelry': 'filters/jewelry/*/*.png',
            'hairstyles': 'filters/hairstyles/*.png',
            'dresses': 'filters/dresses/*/*.png',
            'accessories': 'filters/accessories/*/*.png'
        }

        for category, path_pattern in filter_paths.items():
            try:
                files = glob.glob(path_pattern)
                self.filter_assets[category] = files
            except Exception as e:
                print(f"Error loading {category} assets: {e}")
                self.filter_assets[category] = []

        print(f"Loaded filter assets: {[(k, len(v)) for k, v in self.filter_assets.items()]}")

    def setup_gui(self):
        # Create scrollable main frame
        main_canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )

        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)

        # Main container
        main_frame = ttk.Frame(scrollable_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for controls
        left_panel = ttk.Frame(main_frame, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # Right panel for video
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Video display
        self.video_label = ttk.Label(right_panel)
        self.video_label.pack(expand=True)

        # Control buttons
        control_frame = ttk.Frame(left_panel)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(control_frame, text="Start Camera", command=self.start_camera).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Stop Camera", command=self.stop_camera).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Refresh Filters", command=self.refresh_filters).pack(fill=tk.X, pady=2)

        # Skin tone analysis
        skin_frame = ttk.LabelFrame(left_panel, text="Skin Tone Analysis")
        skin_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(skin_frame, text="Analyze Skin Tone", command=self.analyze_skin_tone).pack(fill=tk.X, pady=2)
        self.skin_tone_label = ttk.Label(skin_frame, text="Skin Tone: Not analyzed")
        self.skin_tone_label.pack(fill=tk.X, pady=2)

        # Season selection
        season_frame = ttk.LabelFrame(left_panel, text="Season Selection")
        season_frame.pack(fill=tk.X, pady=(0, 10))

        self.season_var = tk.StringVar(value="spring")
        seasons = ["spring", "summer", "autumn", "winter"]
        for season in seasons:
            ttk.Radiobutton(season_frame, text=season.capitalize(),
                            variable=self.season_var, value=season,
                            command=self.update_color_suggestions).pack(anchor=tk.W)

        # Color suggestions
        color_frame = ttk.LabelFrame(left_panel, text="Suggested Colors")
        color_frame.pack(fill=tk.X, pady=(0, 10))

        self.color_canvas = tk.Canvas(color_frame, height=60)
        self.color_canvas.pack(fill=tk.X, pady=2)

        # Foundation (color-based, not PNG)
        foundation_frame = ttk.LabelFrame(left_panel, text="Foundation")
        foundation_frame.pack(fill=tk.X, pady=(0, 10))

        foundation_control_frame = ttk.Frame(foundation_frame)
        foundation_control_frame.pack(fill=tk.X, pady=2)
        self.foundation_var = tk.BooleanVar()
        ttk.Checkbutton(foundation_control_frame, text="Foundation",
                        variable=self.foundation_var,
                        command=self.toggle_foundation).pack(side=tk.LEFT)
        ttk.Button(foundation_control_frame, text="Color",
                   command=lambda: self.choose_color('foundation')).pack(side=tk.RIGHT)

        # PNG Jewelry filters
        jewelry_frame = ttk.LabelFrame(left_panel, text="PNG Jewelry Filters")
        jewelry_frame.pack(fill=tk.X, pady=(0, 10))

        jewelry_control_frame = ttk.Frame(jewelry_frame)
        jewelry_control_frame.pack(fill=tk.X, pady=2)
        self.jewelry_var = tk.BooleanVar()
        ttk.Checkbutton(jewelry_control_frame, text="Jewelry",
                        variable=self.jewelry_var,
                        command=self.toggle_jewelry).pack(side=tk.LEFT)
        self.jewelry_combo = ttk.Combobox(jewelry_control_frame, width=15)
        self.jewelry_combo.pack(side=tk.RIGHT)
        self.update_filter_combo('jewelry', self.jewelry_combo)

        # PNG Hair filters
        hair_frame = ttk.LabelFrame(left_panel, text="PNG Hair Filters")
        hair_frame.pack(fill=tk.X, pady=(0, 10))

        hair_control_frame = ttk.Frame(hair_frame)
        hair_control_frame.pack(fill=tk.X, pady=2)
        self.hair_var = tk.BooleanVar()
        ttk.Checkbutton(hair_control_frame, text="Hairstyle",
                        variable=self.hair_var,
                        command=self.toggle_hairstyle).pack(side=tk.LEFT)
        self.hair_combo = ttk.Combobox(hair_control_frame, width=15)
        self.hair_combo.pack(side=tk.RIGHT)
        self.update_filter_combo('hairstyles', self.hair_combo)

        # PNG Dress filters
        dress_frame = ttk.LabelFrame(left_panel, text="PNG Dress Filters")
        dress_frame.pack(fill=tk.X, pady=(0, 10))

        dress_control_frame = ttk.Frame(dress_frame)
        dress_control_frame.pack(fill=tk.X, pady=2)
        self.dress_var = tk.BooleanVar()
        ttk.Checkbutton(dress_control_frame, text="Dress/Outfit",
                        variable=self.dress_var,
                        command=self.toggle_dress).pack(side=tk.LEFT)
        self.dress_combo = ttk.Combobox(dress_control_frame, width=15)
        self.dress_combo.pack(side=tk.RIGHT)
        self.update_filter_combo('dresses', self.dress_combo)

        # PNG Accessory filters
        accessory_frame = ttk.LabelFrame(left_panel, text="PNG Accessory Filters")
        accessory_frame.pack(fill=tk.X, pady=(0, 10))

        accessory_control_frame = ttk.Frame(accessory_frame)
        accessory_control_frame.pack(fill=tk.X, pady=2)
        self.accessory_var = tk.BooleanVar()
        ttk.Checkbutton(accessory_control_frame, text="Accessories",
                        variable=self.accessory_var,
                        command=self.toggle_accessories).pack(side=tk.LEFT)
        self.accessory_combo = ttk.Combobox(accessory_control_frame, width=15)
        self.accessory_combo.pack(side=tk.RIGHT)
        self.update_filter_combo('accessories', self.accessory_combo)

        # Filter intensity controls
        intensity_frame = ttk.LabelFrame(left_panel, text="Filter Intensity")
        intensity_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(intensity_frame, text="Makeup Intensity:").pack(anchor=tk.W)
        self.makeup_intensity = tk.Scale(intensity_frame, from_=0.1, to=1.0,
                                         resolution=0.1, orient=tk.HORIZONTAL)
        self.makeup_intensity.set(0.6)
        self.makeup_intensity.pack(fill=tk.X)

        ttk.Label(intensity_frame, text="Filter Opacity:").pack(anchor=tk.W)
        self.filter_opacity = tk.Scale(intensity_frame, from_=0.1, to=1.0,
                                       resolution=0.1, orient=tk.HORIZONTAL)
        self.filter_opacity.set(0.8)
        self.filter_opacity.pack(fill=tk.X)

        # Pack scrollable components
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.cap = None
        self.is_camera_active = False

    def refresh_filters(self):
        """Refresh and reload all filter assets"""
        self.load_filter_assets()
        self.update_filter_combo('jewelry', self.jewelry_combo)
        self.update_filter_combo('hairstyles', self.hair_combo)
        self.update_filter_combo('dresses', self.dress_combo)
        self.update_filter_combo('accessories', self.accessory_combo)
        messagebox.showinfo("Success", "Filters refreshed successfully!")

    def update_filter_combo(self, category, combobox):
        """Update combobox with available filter assets"""
        assets = self.filter_assets.get(category, [])
        asset_names = [os.path.basename(asset).replace('.png', '') for asset in assets]
        combobox['values'] = asset_names
        if asset_names:
            combobox.set(asset_names[0])

    def start_camera(self):
        """Start camera capture"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot open camera")
                return
            self.is_camera_active = True
            self.update_frame()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {e}")

    def stop_camera(self):
        """Stop camera capture"""
        self.is_camera_active = False
        if self.cap:
            self.cap.release()

    def update_frame(self):
        """Update video frame continuously"""
        if self.is_camera_active and self.cap:
            try:
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)  # Mirror effect
                    self.current_frame = frame.copy()
                    self.process_and_display_frame()
                else:
                    print("Failed to read frame from camera")
            except Exception as e:
                print(f"Error reading frame: {e}")

        if self.is_camera_active:
            self.root.after(30, self.update_frame)

    def process_and_display_frame(self):
        """Process frame with filters and display"""
        if self.current_frame is None:
            return

        try:
            frame = self.current_frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process face mesh for facial features
            face_results = self.face_mesh.process(frame_rgb)

            # Process pose for body/dress placement
            pose_results = self.pose.process(frame_rgb)

            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # Apply PNG-based filters
                    frame = self.apply_png_jewelry_filter(frame, face_landmarks)
                    frame = self.apply_png_hair_filter(frame, face_landmarks)
                    frame = self.apply_png_accessory_filter(frame, face_landmarks)

                    # Apply foundation (color-based)
                    if self.current_filters['foundation']['active']:
                        frame = self.apply_foundation(frame, face_landmarks)

            # Apply dress filters using pose landmarks
            if pose_results.pose_landmarks:
                frame = self.apply_png_dress_filter(frame, pose_results.pose_landmarks)

            # Convert to display format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil = frame_pil.resize((900, 700), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(frame_pil)

            self.video_label.configure(image=photo)
            self.video_label.image = photo

        except Exception as e:
            print(f"Error processing frame: {e}")

    def load_png_with_alpha(self, image_path):
        """Load PNG image with alpha channel preserved"""
        try:
            # Load with PIL to preserve alpha
            pil_image = Image.open(image_path).convert("RGBA")
            # Convert to numpy array
            png_array = np.array(pil_image)
            # Convert RGBA to BGRA for OpenCV
            png_bgra = cv2.cvtColor(png_array, cv2.COLOR_RGBA2BGRA)
            return png_bgra
        except Exception as e:
            print(f"Error loading PNG {image_path}: {e}")
            return None

    def overlay_png_on_frame(self, frame, png_image, x, y, scale=1.0):
        """Overlay PNG image with transparency on frame"""
        if png_image is None:
            return frame

        try:
            # Resize PNG if scale is provided
            if scale != 1.0:
                height, width = png_image.shape[:2]
                new_width = int(width * scale)
                new_height = int(height * scale)
                if new_width > 0 and new_height > 0:
                    png_image = cv2.resize(png_image, (new_width, new_height))

            h, w = png_image.shape[:2]
            frame_h, frame_w = frame.shape[:2]

            # Calculate overlay position
            x = max(0, min(x, frame_w - w))
            y = max(0, min(y, frame_h - h))

            # Ensure we don't go out of bounds
            if x + w > frame_w:
                w = frame_w - x
                png_image = png_image[:, :w]
            if y + h > frame_h:
                h = frame_h - y
                png_image = png_image[:h, :]

            if w <= 0 or h <= 0:
                return frame

            # Extract alpha channel
            if png_image.shape[2] == 4:  # BGRA
                bgr = png_image[:, :, :3]
                alpha = png_image[:, :, 3] / 255.0

                # Get the region of interest
                roi = frame[y:y + h, x:x + w]

                # Blend the images
                for c in range(0, 3):
                    roi[:, :, c] = (alpha * bgr[:, :, c] +
                                    (1 - alpha) * roi[:, :, c])

                frame[y:y + h, x:x + w] = roi
            else:
                # No alpha channel, direct overlay
                frame[y:y + h, x:x + w] = png_image

        except Exception as e:
            print(f"Error overlaying PNG: {e}")

        return frame

    def apply_png_jewelry_filter(self, frame, face_landmarks):
        """Apply PNG-based jewelry filters"""
        if not self.current_filters['jewelry']['active']:
            return frame

        selected_asset = self.get_selected_asset('jewelry', self.jewelry_combo)
        if not selected_asset:
            return frame

        png_image = self.load_png_with_alpha(selected_asset)
        if png_image is None:
            return frame

        height, width = frame.shape[:2]
        landmarks = []

        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks.append((x, y))

        opacity = self.filter_opacity.get()

        # Determine jewelry type and position
        asset_name = os.path.basename(selected_asset).lower()



        if 'necklace' in asset_name:
            # Apply necklace
            neck_pos = self.get_neck_position(landmarks)
            if neck_pos:
                x, y = neck_pos
                scale = 0.6 * opacity
                frame = self.overlay_png_on_frame(frame, png_image,
                                                  x - int(png_image.shape[1] * scale / 2),
                                                  y - int(png_image.shape[0] * scale / 2), scale)

        return frame

    def apply_png_hair_filter(self, frame, face_landmarks):
        """Apply PNG-based hair filters"""
        if not self.current_filters['hairstyle']['active']:
            return frame

        selected_asset = self.get_selected_asset('hairstyles', self.hair_combo)
        if not selected_asset:
            return frame

        png_image = self.load_png_with_alpha(selected_asset)
        if png_image is None:
            return frame

        height, width = frame.shape[:2]
        landmarks = []

        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks.append((x, y))

        opacity = self.filter_opacity.get()

        # Position hair on top of head
        hair_pos = self.get_hair_position(landmarks)
        if hair_pos:
            x, y = hair_pos
            scale = 0.8 * opacity
            frame = self.overlay_png_on_frame(frame, png_image,
                                              x - int(png_image.shape[1] * scale / 2),
                                              y - int(png_image.shape[0] * scale), scale)

        return frame

    def apply_png_accessory_filter(self, frame, face_landmarks):
        """Apply PNG-based accessory filters"""
        if not self.current_filters['accessories']['active']:
            return frame

        selected_asset = self.get_selected_asset('accessories', self.accessory_combo)
        if not selected_asset:
            return frame

        png_image = self.load_png_with_alpha(selected_asset)
        if png_image is None:
            return frame

        height, width = frame.shape[:2]
        landmarks = []

        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks.append((x, y))

        opacity = self.filter_opacity.get()
        asset_name = os.path.basename(selected_asset).lower()

        if 'hat' in asset_name or 'cap' in asset_name:
            # Apply hat/cap on head
            head_pos = self.get_head_top_position(landmarks)
            if head_pos:
                x, y = head_pos
                scale = 0.7 * opacity
                frame = self.overlay_png_on_frame(frame, png_image,
                                                  x - int(png_image.shape[1] * scale / 2),
                                                  y - int(png_image.shape[0] * scale), scale)

        elif 'glasses' in asset_name:
            # Apply glasses on eyes
            glasses_pos = self.get_glasses_position(landmarks)
            if glasses_pos:
                x, y = glasses_pos
                scale = 0.5 * opacity
                frame = self.overlay_png_on_frame(frame, png_image,
                                                  x - int(png_image.shape[1] * scale / 2),
                                                  y - int(png_image.shape[0] * scale / 2), scale)

        return frame

    def apply_png_dress_filter(self, frame, pose_landmarks):
        """Apply PNG-based dress filters using pose landmarks"""
        if not self.current_filters['dress']['active']:
            return frame

        selected_asset = self.get_selected_asset('dresses', self.dress_combo)
        if not selected_asset:
            return frame

        png_image = self.load_png_with_alpha(selected_asset)
        if png_image is None:
            return frame

        height, width = frame.shape[:2]

        # Get body landmarks
        if pose_landmarks:
            try:
                # Shoulder points
                left_shoulder = pose_landmarks.landmark[11]  # Left shoulder
                right_shoulder = pose_landmarks.landmark[12]  # Right shoulder

                # Torso points
                left_hip = pose_landmarks.landmark[23]  # Left hip
                right_hip = pose_landmarks.landmark[24]  # Right hip

                # Convert to pixel coordinates
                left_shoulder_px = (int(left_shoulder.x * width), int(left_shoulder.y * height))
                right_shoulder_px = (int(right_shoulder.x * width), int(right_shoulder.y * height))
                left_hip_px = (int(left_hip.x * width), int(left_hip.y * height))
                right_hip_px = (int(right_hip.x * width), int(right_hip.y * height))

                # Calculate dress position and scale
                shoulder_width = abs(right_shoulder_px[0] - left_shoulder_px[0])
                torso_height = abs(left_hip_px[1] - left_shoulder_px[1])

                if shoulder_width > 0 and torso_height > 0:
                    # Calculate dress center
                    center_x = (left_shoulder_px[0] + right_shoulder_px[0]) // 2
                    center_y = (left_shoulder_px[1] + left_hip_px[1]) // 2

                    # Scale dress based on body size
                    dress_scale = min(shoulder_width / png_image.shape[1],
                                      torso_height / png_image.shape[0]) * 1.5

                    opacity = self.filter_opacity.get()
                    dress_scale *= opacity

                    # Position dress on torso
                    dress_x = center_x - int(png_image.shape[1] * dress_scale / 2)
                    dress_y = left_shoulder_px[1] - int(png_image.shape[0] * dress_scale * 0.1)

                    frame = self.overlay_png_on_frame(frame, png_image, dress_x, dress_y, dress_scale)

            except Exception as e:
                print(f"Error applying dress filter: {e}")

        return frame

    def get_selected_asset(self, category, combobox):
        """Get the selected asset path from combobox"""
        try:
            selected_name = combobox.get()
            if not selected_name:
                return None

            assets = self.filter_assets.get(category, [])
            for asset_path in assets:
                if selected_name in os.path.basename(asset_path):
                    return asset_path
        except Exception as e:
            print(f"Error getting selected asset: {e}")
        return None

    # Landmark position calculation methods
    def get_ear_positions(self, landmarks):
        """Get positions for both ears"""
        try:
            if len(landmarks) > 300:
                # Approximate ear positions based on face landmarks
                left_ear_x = landmarks[234][0] - 40 if len(landmarks) > 234 else None
                left_ear_y = landmarks[234][1] if len(landmarks) > 234 else None
                left_ear = (left_ear_x, left_ear_y) if left_ear_x and left_ear_y else None

                right_ear_x = landmarks[454][0] + 40 if len(landmarks) > 454 else None
                right_ear_y = landmarks[454][1] if len(landmarks) > 454 else None
                right_ear = (right_ear_x, right_ear_y) if right_ear_x and right_ear_y else None

                return [left_ear, right_ear]
        except Exception as e:
            print(f"Error getting ear positions: {e}")
        return [None, None]

    def get_neck_position(self, landmarks):
        """Get neck position for necklace"""
        try:
            if len(landmarks) > 10:
                # Approximate neck position below chin
                chin_point = landmarks[175] if len(landmarks) > 175 else landmarks[10]
                neck_x = chin_point[0]
                neck_y = chin_point[1] + 50  # Below chin
                return (neck_x, neck_y)
        except Exception as e:
            print(f"Error getting neck position: {e}")
        return None

    def get_hair_position(self, landmarks):
        """Get position for hair on top of head"""
        try:
            if len(landmarks) > 10:
                # Top of forehead
                forehead_point = landmarks[10]
                hair_x = forehead_point[0]
                hair_y = forehead_point[1] - 100  # Above forehead
                return (hair_x, hair_y)
        except Exception as e:
            print(f"Error getting hair position: {e}")
        return None

    def get_head_top_position(self, landmarks):
        """Get top of head position for hats"""
        try:
            if len(landmarks) > 10:
                forehead_point = landmarks[10]
                hat_x = forehead_point[0]
                hat_y = forehead_point[1] - 120  # Above head
                return (hat_x, hat_y)
        except Exception as e:
            print(f"Error getting head top position: {e}")
        return None

    def get_glasses_position(self, landmarks):
        """Get position for glasses"""
        try:
            if len(landmarks) > 200:
                # Between both eyes
                left_eye = landmarks[33] if len(landmarks) > 33 else None
                right_eye = landmarks[362] if len(landmarks) > 362 else None

                if left_eye and right_eye:
                    glasses_x = (left_eye[0] + right_eye[0]) // 2
                    glasses_y = (left_eye[1] + right_eye[1]) // 2
                    return (glasses_x, glasses_y)
        except Exception as e:
            print(f"Error getting glasses position: {e}")
        return None

    def get_scarf_position(self, landmarks):
        """Get position for scarf around neck"""
        try:
            if len(landmarks) > 175:
                chin_point = landmarks[175]
                scarf_x = chin_point[0]
                scarf_y = chin_point[1] + 80  # Below neck
                return (scarf_x, scarf_y)
        except Exception as e:
            print(f"Error getting scarf position: {e}")
        return None

    # Existing methods for skin tone analysis and basic makeup
    def analyze_skin_tone(self):
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

    def update_color_suggestions(self):
        """Update color suggestions based on skin tone and season"""
        if self.skin_tone is None:
            return

        try:
            season = self.season_var.get()
            colors = self.color_palettes[self.skin_tone][season]

            self.color_canvas.delete("all")
            canvas_width = self.color_canvas.winfo_reqwidth() or 300
            color_width = max(1, canvas_width // len(colors))

            for i, color in enumerate(colors):
                hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
                x1 = i * color_width
                x2 = (i + 1) * color_width
                self.color_canvas.create_rectangle(x1, 0, x2, 60, fill=hex_color, outline="")
        except Exception as e:
            print(f"Error updating color suggestions: {e}")

    def choose_color(self, filter_type):
        """Choose color for a specific filter"""
        try:
            from tkinter import colorchooser
            color = colorchooser.askcolor()
            if color[0]:  # If color was selected
                rgb_color = tuple(int(c) for c in color[0])
                bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])  # Convert RGB to BGR for OpenCV
                self.current_filters[filter_type]['color'] = bgr_color
        except Exception as e:
            print(f"Error choosing color: {e}")

    # Toggle methods for filters
    def toggle_foundation(self):
        """Toggle foundation filter"""
        self.current_filters['foundation']['active'] = self.foundation_var.get()

    def toggle_jewelry(self):
        """Toggle jewelry filter"""
        self.current_filters['jewelry']['active'] = self.jewelry_var.get()

    def toggle_hairstyle(self):
        """Toggle hairstyle filter"""
        self.current_filters['hairstyle']['active'] = self.hair_var.get()

    def toggle_dress(self):
        """Toggle dress filter"""
        self.current_filters['dress']['active'] = self.dress_var.get()

    def toggle_accessories(self):
        """Toggle accessories filter"""
        self.current_filters['accessories']['active'] = self.accessory_var.get()

    def apply_foundation(self, frame, face_landmarks):
        """Apply foundation (color-based, not PNG)"""
        height, width = frame.shape[:2]
        landmarks = []

        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks.append((x, y))

        try:
            # Get face contour points (simplified)
            if len(landmarks) > 400:
                face_contour_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                                        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                                        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

                face_points = []
                for idx in face_contour_indices:
                    if idx < len(landmarks):
                        face_points.append(landmarks[idx])

                if len(face_points) > 10:
                    face_points = np.array(face_points, dtype=np.int32)

                    # Create face mask
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [face_points], 255)

                    # Apply foundation
                    color = self.current_filters['foundation']['color']
                    overlay = frame.copy()
                    overlay[mask > 0] = [int(c * 0.3 + frame[mask > 0, i] * 0.7)
                                         for i, c in enumerate(color)]

                    # Blend
                    alpha = 0.2 * self.makeup_intensity.get()
                    frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

        except Exception as e:
            print(f"Error applying foundation: {e}")

        return frame

    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

    def run(self):
        """Run the application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("Application interrupted by user")
        except Exception as e:
            print(f"Application error: {e}")
        finally:
            # Cleanup
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        # Run the application
        app = FaceFilterApp()
        app.run()
    except Exception as e:
        print(f"Failed to start application: {e}")
        import traceback

        traceback.print_exc()


1