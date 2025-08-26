import os
import sys
import logging
import json
from dataclasses import dataclass, asdict
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    import mediapipe as mp
    from math import atan2, degrees, sqrt
    from sklearn.cluster import KMeans
except ImportError as e:
    print(f"[ERROR] Missing required package: {e}")
    print("Please install required packages: pip install opencv-python mediapipe numpy scikit-learn")
    sys.exit(1)



# ------------------------- Fixed Configuration with Better Scaling -------------------------
@dataclass
class Config:
    """Configuration class with improved dress scaling"""
    # Directory settings
    ASSETS_DIR: str = "assets/dresses"
    OUTPUT_DIR: str = "output"
    LOGS_DIR: str = "logs"
    CONFIG_FILE: str = "beauty_filter_config.json"

    # ✅ IMPROVED: Better dress scaling parameters
    WIDTH_FACTOR: float = 1.5299999999999996
    HEIGHT_FACTOR: float = 1.1899999999999997
    VERTICAL_OFFSET_FACTOR: float = 0.37

    # Video settings
    CAM_INDEX: int = 0
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720

    # Makeup settings
    MAKEUP_INTENSITY: float = 0.5
    MAX_INTENSITY: float = 1.0
    MIN_INTENSITY: float = 0.1
    INTENSITY_STEP: float = 0.1

    # Skin tone calibration
    CALIBRATION_FRAMES: int = 30
    MIN_SKIN_PIXELS: int = 100
    KMEANS_CLUSTERS: int = 2

    # Performance settings
    MAX_RETRIES: int = 3
    FRAME_SKIP_THRESHOLD: int = 5

    def validate(self) -> bool:
        """Validate configuration values"""
        try:
            assert 0.5 <= self.WIDTH_FACTOR <= 10.0, "WIDTH_FACTOR must be between 0.5 and 10.0"
            assert 0.5 <= self.HEIGHT_FACTOR <= 10.0, "HEIGHT_FACTOR must be between 0.5 and 10.0"
            assert -2.0 <= self.VERTICAL_OFFSET_FACTOR <= 2.0, "VERTICAL_OFFSET_FACTOR must be between -2.0 and 2.0"
            assert 480 <= self.FRAME_WIDTH <= 3840, "FRAME_WIDTH must be between 480 and 3840"
            assert 320 <= self.FRAME_HEIGHT <= 2160, "FRAME_HEIGHT must be between 320 and 2160"
            assert 0.0 <= self.MAKEUP_INTENSITY <= 1.0, "MAKEUP_INTENSITY must be between 0.0 and 1.0"
            assert self.CALIBRATION_FRAMES > 0, "CALIBRATION_FRAMES must be positive"
            return True
        except AssertionError as e:
            logging.error(f"Configuration validation failed: {e}")
            return False

    def save(self, filepath: str) -> bool:
        """Save configuration to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(asdict(self), f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
            return False

    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """Load configuration from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                config = cls(**data)
                if config.validate():
                    return config
                else:
                    logging.warning("Invalid configuration, using defaults")
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")

        return cls()  # Return default configuration