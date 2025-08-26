class CameraManager:
    """Enhanced camera management with comprehensive error handling"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cap = None
        self.retry_count = 0
        self.frame_skip_count = 0

    def initialize(self) -> bool:
        """Initialize camera with error handling and retries"""
        for attempt in range(self.config.MAX_RETRIES):
            try:
                self.logger.info(f"Attempting to initialize camera (attempt {attempt + 1})")

                # Try different camera indices if primary fails
                camera_indices = [self.config.CAM_INDEX, 0, 1, 2] if self.config.CAM_INDEX != 0 else [0, 1, 2]

                for cam_idx in camera_indices:
                    try:
                        self.cap = cv2.VideoCapture(cam_idx)

                        if self.cap.isOpened():
                            # Set camera properties
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
                            self.cap.set(cv2.CAP_PROP_FPS, 30)
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency

                            # Test camera by reading a frame
                            ret, frame = self.cap.read()
                            if ret and frame is not None:
                                self.logger.info(f"Camera initialized successfully on index {cam_idx}")
                                return True
                            else:
                                self.cap.release()

                    except Exception as e:
                        self.logger.warning(f"Failed to initialize camera {cam_idx}: {e}")
                        if self.cap:
                            self.cap.release()
                            self.cap = None
                        continue

                time.sleep(1)  # Wait before retry

            except Exception as e:
                self.logger.error(f"Camera initialization attempt {attempt + 1} failed: {e}")
                time.sleep(2)

        raise CameraError("Failed to initialize camera after all attempts")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame with error handling"""
        try:
            if not self.cap or not self.cap.isOpened():
                return False, None

            ret, frame = self.cap.read()

            if not ret or frame is None:
                self.frame_skip_count += 1

                if self.frame_skip_count > self.config.FRAME_SKIP_THRESHOLD:
                    self.logger.warning("Too many consecutive frame read failures")
                    return False, None

                return False, None

            # Reset skip count on successful read
            self.frame_skip_count = 0

            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)

            return True, frame

        except Exception as e:
            self.logger.error(f"Error reading camera frame: {e}")
            return False, None

    def release(self):
        """Release camera resources"""
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
                self.logger.info("Camera released")
        except Exception as e:
            self.logger.error(f"Error releasing camera: {e}")

    def get_frame_info(self) -> Dict[str, Any]:
        """Get camera frame information"""
        try:
            if not self.cap or not self.cap.isOpened():
                return {}

            return {
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'format': int(self.cap.get(cv2.CAP_PROP_FORMAT))
            }
        except Exception as e:
            self.logger.error(f"Error getting frame info: {e}")
            return {}
