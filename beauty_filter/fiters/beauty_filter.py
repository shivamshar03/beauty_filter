class BeautyFilter:
    """Enhanced beauty filter with comprehensive error handling"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Makeup state
        self.lipstick_enabled = False
        self.blush_enabled = False
        self.eyeshadow_enabled = False
        self.makeup_enabled = True

        # Color indices with bounds checking
        self.lipstick_color_idx = 0
        self.blush_color_idx = 0
        self.eyeshadow_color_idx = 0

        self.makeup_intensity = config.MAKEUP_INTENSITY

        self.logger.info("BeautyFilter initialized")

    def _validate_landmarks(self, landmarks, required_count: int = 468) -> bool:
        """Validate MediaPipe landmarks"""
        try:
            if not landmarks or not hasattr(landmarks, 'landmark'):
                return False
            return len(landmarks.landmark) >= required_count
        except Exception as e:
            self.logger.warning(f"Landmark validation failed: {e}")
            return False

    def get_lip_landmarks(self) -> List[int]:
        """Get lip landmarks with validation"""
        try:
            # Enhanced lip boundary for better coverage
            upper_lip = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402]
            lower_lip = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 78]
            landmarks = upper_lip + lower_lip

            # Validate landmark indices
            validated_landmarks = [idx for idx in landmarks if 0 <= idx < 468]
            if len(validated_landmarks) < len(landmarks):
                self.logger.warning(f"Some lip landmarks out of range: {set(landmarks) - set(validated_landmarks)}")

            return validated_landmarks if validated_landmarks else [61, 84, 17, 314]  # Minimal fallback

        except Exception as e:
            self.logger.error(f"Error getting lip landmarks: {e}")
            return [61, 84, 17, 314]  # Basic fallback

    def get_cheek_landmarks(self) -> Tuple[List[int], List[int]]:
        """Get cheek landmarks with validation"""
        try:
            # Enhanced cheek areas for natural blush
            left_cheek = [116, 117, 118, 119, 120, 126, 142, 36, 205, 206, 207, 213, 192, 147, 187, 207, 216]
            right_cheek = [345, 346, 347, 348, 349, 355, 371, 266, 425, 426, 427, 436, 416, 376, 411, 427, 436]

            # Validate and filter landmarks
            left_valid = [idx for idx in left_cheek if 0 <= idx < 468]
            right_valid = [idx for idx in right_cheek if 0 <= idx < 468]

            # Fallback if too few valid landmarks
            if len(left_valid) < 3:
                left_valid = [116, 117, 118]
            if len(right_valid) < 3:
                right_valid = [345, 346, 347]

            return left_valid, right_valid

        except Exception as e:
            self.logger.error(f"Error getting cheek landmarks: {e}")
            return [116, 117, 118], [345, 346, 347]

    def get_eye_landmarks(self) -> Tuple[List[int], List[int]]:
        """Get eye landmarks with validation"""
        try:
            # Enhanced eyelid areas for eyeshadow
            left_eye = [157, 158, 159, 160, 161, 163, 144, 145, 153, 154, 155, 133, 173, 246]
            right_eye = [384, 385, 386, 387, 388, 390, 373, 374, 380, 381, 382, 362, 398, 466]

            # Validate landmarks
            left_valid = [idx for idx in left_eye if 0 <= idx < 468]
            right_valid = [idx for idx in right_eye if 0 <= idx < 468]

            # Fallback if too few valid landmarks
            if len(left_valid) < 3:
                left_valid = [157, 158, 159]
            if len(right_valid) < 3:
                right_valid = [384, 385, 386]

            return left_valid, right_valid

        except Exception as e:
            self.logger.error(f"Error getting eye landmarks: {e}")
            return [157, 158, 159], [384, 385, 386]

    def create_smooth_mask(self, frame_shape: Tuple[int, int], landmarks, indices: List[int], blur_size: int = 25) -> \
    Optional[np.ndarray]:
        """Create smooth mask with comprehensive error handling"""
        try:
            if not frame_shape or len(frame_shape) < 2:
                self.logger.error("Invalid frame shape")
                return None

            h, w = frame_shape[:2]
            if h <= 0 or w <= 0:
                self.logger.error(f"Invalid frame dimensions: {h}x{w}")
                return None

            # Validate inputs
            if not self._validate_landmarks(landmarks) or not indices:
                self.logger.warning("Invalid landmarks or indices for mask creation")
                return None

            mask = np.zeros((h, w), dtype=np.uint8)
            points = []

            # Extract valid points
            for idx in indices:
                try:
                    if 0 <= idx < len(landmarks.landmark):
                        landmark = landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)

                        # Validate coordinates
                        if 0 <= x < w and 0 <= y < h:
                            points.append([x, y])
                        else:
                            self.logger.debug(f"Point {idx} out of bounds: ({x}, {y})")

                except (AttributeError, IndexError, TypeError) as e:
                    self.logger.debug(f"Error processing landmark {idx}: {e}")
                    continue

            # Need at least 3 points for a polygon
            if len(points) < 3:
                self.logger.warning(f"Insufficient valid points for mask: {len(points)}")
                return mask

            # Create polygon mask
            points_array = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points_array], 255)

            # Apply blur with validation
            blur_size = max(1, min(blur_size, min(h, w) // 4))  # Ensure blur size is reasonable
            if blur_size % 2 == 0:
                blur_size += 1  # Ensure odd number for Gaussian blur

            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

            return mask

        except Exception as e:
            self.logger.error(f"Error creating smooth mask: {e}")
            return np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8) if len(frame_shape) >= 2 else None

    def apply_enhanced_color(self, frame: np.ndarray, mask: Optional[np.ndarray], color: Tuple[int, int, int],
                             intensity: float, blend_mode: str = 'normal') -> np.ndarray:
        """Enhanced color application with comprehensive error handling"""
        try:
            # Validate inputs
            if frame is None or not isinstance(frame, np.ndarray):
                self.logger.error("Invalid frame for color application")
                return frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

            if mask is None or np.sum(mask) == 0:
                return frame

            if not ColorPalette.validate_color(color):
                self.logger.warning(f"Invalid color {color}, using default")
                color = (128, 128, 128)

            # Clamp intensity
            intensity = max(0.0, min(1.0, intensity))

            # Ensure frame and mask compatibility
            if frame.shape[:2] != mask.shape[:2]:
                self.logger.error(f"Frame and mask shape mismatch: {frame.shape[:2]} vs {mask.shape[:2]}")
                return frame

            # Normalize mask and ensure proper dimensions
            mask_norm = mask.astype(np.float32) / 255.0
            mask_norm *= intensity

            # Expand mask dimensions for broadcasting
            if len(mask_norm.shape) == 2:
                mask_norm = np.expand_dims(mask_norm, axis=2)

            # Create color overlay
            overlay = np.full_like(frame, color, dtype=np.float32)
            frame_float = frame.astype(np.float32)

            # Apply blending
            if blend_mode == 'multiply':
                # Multiply blend for more natural colors
                result = frame_float * (1 - mask_norm) + (frame_float * overlay / 255.0) * mask_norm
            elif blend_mode == 'soft_light':
                # Soft light blend for subtle effects
                blend = 2 * frame_float * overlay / 255.0
                result = frame_float * (1 - mask_norm) + blend * mask_norm
            else:
                # Normal blend
                result = frame_float * (1 - mask_norm) + overlay * mask_norm

            return np.clip(result, 0, 255).astype(np.uint8)

        except Exception as e:
            self.logger.error(f"Error applying enhanced color: {e}")
            return frame

    def apply_lipstick(self, frame: np.ndarray, face_landmarks) -> np.ndarray:
        """Enhanced lipstick application with error handling"""
        try:
            if not self.lipstick_enabled or not self._validate_landmarks(face_landmarks):
                return frame

            lip_landmarks = self.get_lip_landmarks()
            mask = self.create_smooth_mask(frame.shape, face_landmarks, lip_landmarks, blur_size=15)

            if mask is None:
                return frame

            color = ColorPalette.get_safe_color(ColorPalette.LIPSTICK_COLORS, self.lipstick_color_idx)
            return self.apply_enhanced_color(frame, mask, color, self.makeup_intensity * 0.8, 'multiply')

        except Exception as e:
            self.logger.error(f"Error applying lipstick: {e}")
            return frame

    def apply_blush(self, frame: np.ndarray, face_landmarks) -> np.ndarray:
        """Enhanced blush application with error handling"""
        try:
            if not self.blush_enabled or not self._validate_landmarks(face_landmarks):
                return frame

            left_cheek, right_cheek = self.get_cheek_landmarks()
            color = ColorPalette.get_safe_color(ColorPalette.BLUSH_COLORS, self.blush_color_idx)

            # Apply to both cheeks with larger blur for natural look
            left_mask = self.create_smooth_mask(frame.shape, face_landmarks, left_cheek, blur_size=35)
            right_mask = self.create_smooth_mask(frame.shape, face_landmarks, right_cheek, blur_size=35)

            if left_mask is not None:
                frame = self.apply_enhanced_color(frame, left_mask, color, self.makeup_intensity * 0.4, 'soft_light')
            if right_mask is not None:
                frame = self.apply_enhanced_color(frame, right_mask, color, self.makeup_intensity * 0.4, 'soft_light')

            return frame

        except Exception as e:
            self.logger.error(f"Error applying blush: {e}")
            return frame

    def apply_eyeshadow(self, frame: np.ndarray, face_landmarks) -> np.ndarray:
        """Enhanced eyeshadow application with error handling"""
        try:
            if not self.eyeshadow_enabled or not self._validate_landmarks(face_landmarks):
                return frame

            left_eye, right_eye = self.get_eye_landmarks()
            color = ColorPalette.get_safe_color(ColorPalette.EYESHADOW_COLORS, self.eyeshadow_color_idx)

            # Apply to both eyes
            left_mask = self.create_smooth_mask(frame.shape, face_landmarks, left_eye, blur_size=20)
            right_mask = self.create_smooth_mask(frame.shape, face_landmarks, right_eye, blur_size=20)

            if left_mask is not None:
                frame = self.apply_enhanced_color(frame, left_mask, color, self.makeup_intensity * 0.5)
            if right_mask is not None:
                frame = self.apply_enhanced_color(frame, right_mask, color, self.makeup_intensity * 0.5)

            return frame

        except Exception as e:
            self.logger.error(f"Error applying eyeshadow: {e}")
            return frame

    def cycle_color(self, makeup_type: str) -> Tuple[int, str]:
        """Safely cycle through makeup colors"""
        try:
            color_maps = {
                'lipstick': (ColorPalette.LIPSTICK_COLORS, ColorPalette.LIPSTICK_NAMES, 'lipstick_color_idx'),
                'blush': (ColorPalette.BLUSH_COLORS, ColorPalette.BLUSH_NAMES, 'blush_color_idx'),
                'eyeshadow': (ColorPalette.EYESHADOW_COLORS, ColorPalette.EYESHADOW_NAMES, 'eyeshadow_color_idx')
            }

            if makeup_type not in color_maps:
                return 0, "Unknown"

            colors, names, attr = color_maps[makeup_type]
            current_idx = getattr(self, attr, 0)

            # Safely cycle index
            new_idx = (current_idx + 1) % len(colors)
            setattr(self, attr, new_idx)

            color_name = names[new_idx] if new_idx < len(names) else f"Color {new_idx + 1}"
            return new_idx, color_name

        except Exception as e:
            self.logger.error(f"Error cycling {makeup_type} color: {e}")
            return 0, "Error"

    def adjust_intensity(self, delta: float) -> float:
        """Safely adjust makeup intensity"""
        try:
            new_intensity = self.makeup_intensity + delta
            self.makeup_intensity = max(self.config.MIN_INTENSITY,
                                        min(self.config.MAX_INTENSITY, new_intensity))
            return self.makeup_intensity
        except Exception as e:
            self.logger.error(f"Error adjusting intensity: {e}")
            return self.makeup_intensity
