class SkinToneDetector:
    """Enhanced skin tone detector with comprehensive error handling"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.calibrated_skin_tone = None
        self.calibration_samples = []
        self.is_calibrating = False
        self.calibration_count = 0
        self.calibration_start_time = None

        self.logger.info("SkinToneDetector initialized")

    def extract_face_skin_color(self, frame: np.ndarray, face_landmarks, frame_shape: Tuple[int, int]) -> Optional[
        np.ndarray]:
        """Extract skin color with comprehensive error handling"""
        try:
            if frame is None or not isinstance(frame, np.ndarray):
                return None

            if not face_landmarks or not hasattr(face_landmarks, 'landmark'):
                return None

            if len(frame_shape) < 2 or any(dim <= 0 for dim in frame_shape[:2]):
                return None

            h, w = frame_shape[:2]

            # Use forehead and cheek areas for better skin tone detection
            forehead_landmarks = [9, 10, 151, 337, 299, 333, 298, 301]
            cheek_landmarks = [116, 117, 118, 50, 205, 206, 207, 213, 345, 346, 347, 280, 425, 426, 427, 436]
            all_landmarks = forehead_landmarks + cheek_landmarks

            points = []
            for idx in all_landmarks:
                try:
                    if 0 <= idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)

                        # Validate coordinates
                        if 0 <= x < w and 0 <= y < h:
                            points.append((x, y))

                except (AttributeError, IndexError, TypeError):
                    continue

            if len(points) < 8:
                self.logger.debug("Insufficient face points for skin color extraction")
                return None

            # Create mask for skin areas
            points_np = np.array(points, dtype=np.int32)
            hull = cv2.convexHull(points_np)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [hull], 255)

            # Extract skin pixels
            skin_pixels = frame[mask > 0]

            if len(skin_pixels) < self.config.MIN_SKIN_PIXELS:
                self.logger.debug(f"Insufficient skin pixels: {len(skin_pixels)}")
                return None

            # Use K-means to find dominant color with error handling
            skin_pixels_reshaped = skin_pixels.reshape(-1, 3)

            # Validate pixel values
            if skin_pixels_reshaped.shape[0] == 0:
                return None

            try:
                kmeans = KMeans(n_clusters=self.config.KMEANS_CLUSTERS, random_state=42, n_init=10)
                kmeans.fit(skin_pixels_reshaped)

                # Get the most frequent cluster (dominant skin tone)
                labels = kmeans.labels_
                unique, counts = np.unique(labels, return_counts=True)
                dominant_cluster = unique[np.argmax(counts)]
                skin_tone = kmeans.cluster_centers_[dominant_cluster]

                # Validate result
                if len(skin_tone) >= 3 and all(0 <= val <= 255 for val in skin_tone[:3]):
                    return skin_tone.astype(np.uint8)
                else:
                    self.logger.warning(f"Invalid skin tone detected: {skin_tone}")
                    return None

            except Exception as e:
                self.logger.error(f"K-means clustering failed: {e}")
                # Fallback to simple mean
                mean_color = np.mean(skin_pixels_reshaped, axis=0)
                if len(mean_color) >= 3 and all(0 <= val <= 255 for val in mean_color[:3]):
                    return mean_color.astype(np.uint8)
                return None

        except Exception as e:
            self.logger.error(f"Error extracting face skin color: {e}")
            return None

    def calibrate_skin_tone(self, frame: np.ndarray, face_landmarks, frame_shape: Tuple[int, int]) -> bool:
        """Calibrate skin tone with enhanced error handling"""
        try:
            if not self.is_calibrating:
                return False

            skin_color = self.extract_face_skin_color(frame, face_landmarks, frame_shape)

            if skin_color is not None:
                self.calibration_samples.append(skin_color)
                self.calibration_count += 1

                if self.calibration_count >= self.config.CALIBRATION_FRAMES:
                    try:
                        # Calculate average with outlier removal
                        samples_array = np.array(self.calibration_samples)

                        # Remove outliers using IQR method
                        Q1 = np.percentile(samples_array, 25, axis=0)
                        Q3 = np.percentile(samples_array, 75, axis=0)
                        IQR = Q3 - Q1

                        # Filter outliers
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        filtered_samples = []
                        for sample in samples_array:
                            if all(lower_bound <= sample) and all(sample <= upper_bound):
                                filtered_samples.append(sample)

                        if filtered_samples:
                            self.calibrated_skin_tone = np.mean(filtered_samples, axis=0).astype(np.uint8)
                        else:
                            # Fallback to simple mean if all samples are outliers
                            self.calibrated_skin_tone = np.mean(samples_array, axis=0).astype(np.uint8)

                        self.is_calibrating = False
                        calibration_time = time.time() - (self.calibration_start_time or time.time())

                        self.logger.info(
                            f"Skin tone calibrated in {calibration_time:.1f}s: BGR{tuple(self.calibrated_skin_tone)}")
                        return True

                    except Exception as e:
                        self.logger.error(f"Error finalizing calibration: {e}")
                        self.is_calibrating = False
                        return False

            return False

        except Exception as e:
            self.logger.error(f"Error in skin tone calibration: {e}")
            return False

    def start_calibration(self):
        """Start skin tone calibration with error handling"""
        try:
            self.calibration_samples = []
            self.calibration_count = 0
            self.is_calibrating = True
            self.calibration_start_time = time.time()
            self.logger.info("Starting skin tone calibration...")

        except Exception as e:
            self.logger.error(f"Error starting calibration: {e}")

    def reset_calibration(self):
        """Reset calibration data"""
        try:
            self.calibrated_skin_tone = None
            self.calibration_samples = []
            self.calibration_count = 0
            self.is_calibrating = False
            self.calibration_start_time = None
            self.logger.info("Skin tone calibration reset")

        except Exception as e:
            self.logger.error(f"Error resetting calibration: {e}")

    def draw_skin_color_display(self, frame: np.ndarray, x: int, y: int, size: int = 50):
        """Draw skin color swatch with error handling"""
        try:
            if frame is None or not isinstance(frame, np.ndarray):
                return

            h, w = frame.shape[:2]

            # Validate coordinates
            if not (0 <= x < w - size and 0 <= y < h - size):
                return

            if self.calibrated_skin_tone is not None:
                color = tuple(int(c) for c in self.calibrated_skin_tone[:3])

                # Draw color swatch
                cv2.rectangle(frame, (x, y), (x + size, y + size), color, -1)
                cv2.rectangle(frame, (x, y), (x + size, y + size), (255, 255, 255), 2)

                # Add text label
                cv2.putText(frame, "Skin", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        except Exception as e:
            self.logger.error(f"Error drawing skin color display: {e}")

    def get_calibration_progress(self) -> float:
        """Get calibration progress as percentage"""
        try:
            if not self.is_calibrating:
                return 100.0 if self.calibrated_skin_tone is not None else 0.0
            return (self.calibration_count / self.config.CALIBRATION_FRAMES) * 100.0
        except Exception:
            return 0.0

