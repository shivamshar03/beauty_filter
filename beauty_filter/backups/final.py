
import cv2
import numpy as np
import os
import sys
import traceback
import logging
from pathlib import Path
import time
import json
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
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

# ------------------------- Configuration Management -------------------------
@dataclass
class Config:
    """Configuration class with validation"""
    # Directory settings
    ASSETS_DIR: str = "../assets/dresses"
    OUTPUT_DIR: str = "../output"
    LOGS_DIR: str = "../logs"
    CONFIG_FILE: str = "../beauty_filter_config.json"

    # Dress scaling factors
    WIDTH_FACTOR: float = 1.5
    HEIGHT_FACTOR: float = 2.3













    VERTICAL_OFFSET_FACTOR: float = 0.3

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
            assert 0.5 <= self.WIDTH_FACTOR <= 5.0, "WIDTH_FACTOR must be between 0.5 and 5.0"
            assert 0.5 <= self.HEIGHT_FACTOR <= 5.0, "HEIGHT_FACTOR must be between 0.5 and 5.0"
            assert 0.0 <= self.VERTICAL_OFFSET_FACTOR <= 1.0, "VERTICAL_OFFSET_FACTOR must be between 0.0 and 1.0"
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

# ------------------------- Logging Setup -------------------------
def setup_logging(log_dir: str) -> logging.Logger:
    """Setup comprehensive logging system"""
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        log_filename = f"beauty_filter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_filepath = os.path.join(log_dir, log_filename)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            handlers=[
                logging.FileHandler(log_filepath, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )

        logger = logging.getLogger(__name__)
        logger.info("Logging system initialized")
        return logger

    except Exception as e:
        print(f"[ERROR] Failed to setup logging: {e}")
        # Fallback to basic console logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

# ------------------------- Enhanced Color Definitions -------------------------
class ColorPalette:
    """Enhanced color palette with validation"""

    # Enhanced color palettes (BGR format)
    LIPSTICK_COLORS = [
        (45, 45, 200),    # Classic Red
        (80, 60, 180),    # Berry Rose
        (30, 80, 220),    # Bright Red
        (20, 30, 140),    # Deep Red
        (120, 100, 200),  # Pink Red
        (40, 20, 100),    # Wine Berry
        (90, 70, 160),    # Mauve
        (100, 80, 190),   # Coral Pink
        (60, 40, 160),    # Berry
        (110, 90, 210),   # Rose Gold
    ]

    LIPSTICK_NAMES = [
        "Classic Red", "Berry Rose", "Bright Red", "Deep Red", "Pink Red",
        "Wine Berry", "Mauve", "Coral Pink", "Berry", "Rose Gold"
    ]

    BLUSH_COLORS = [
        (130, 160, 255),  # Soft Pink
        (140, 180, 240),  # Peach
        (100, 130, 200),  # Rose
        (120, 160, 220),  # Coral
        (150, 170, 255),  # Light Pink
        (90, 120, 180),   # Natural Rose
        (110, 140, 200),  # Dusty Rose
        (120, 150, 210),  # Warm Pink
        (100, 140, 190),  # Sunset
        (130, 150, 230),  # Blushing
    ]

    BLUSH_NAMES = [
        "Soft Pink", "Peach", "Rose", "Coral", "Light Pink",
        "Natural Rose", "Dusty Rose", "Warm Pink", "Sunset", "Blushing"
    ]

    EYESHADOW_COLORS = [
        (120, 100, 80),   # Warm Brown
        (160, 130, 100),  # Bronze
        (140, 100, 120),  # Plum
        (100, 130, 80),   # Olive Green
        (180, 170, 100),  # Gold
        (100, 80, 60),    # Deep Brown
        (120, 120, 140),  # Taupe
        (100, 100, 100),  # Smokey Grey
        (130, 110, 90),   # Copper
        (90, 80, 120),    # Chocolate
    ]

    EYESHADOW_NAMES = [
        "Warm Brown", "Bronze", "Plum", "Olive Green", "Gold",
        "Deep Brown", "Taupe", "Smokey Grey", "Copper", "Chocolate"
    ]

    @classmethod
    def validate_color(cls, color: Tuple[int, int, int]) -> bool:
        """Validate BGR color values"""
        try:
            b, g, r = color
            return all(0 <= val <= 255 for val in [b, g, r])
        except (ValueError, TypeError):
            return False

    @classmethod
    def get_safe_color(cls, color_list: List[Tuple[int, int, int]], index: int) -> Tuple[int, int, int]:
        """Safely get color from list with bounds checking"""
        try:
            if not color_list:
                return (128, 128, 128)  # Default gray

            safe_index = max(0, min(index, len(color_list) - 1))
            color = color_list[safe_index]

            if cls.validate_color(color):
                return color
            else:
                logging.warning(f"Invalid color at index {safe_index}: {color}")
                return (128, 128, 128)  # Default gray

        except Exception as e:
            logging.error(f"Error getting safe color: {e}")
            return (128, 128, 128)

# ------------------------- Exception Classes -------------------------
class BeautyFilterError(Exception):
    """Base exception for beauty filter"""
    pass

class CameraError(BeautyFilterError):
    """Camera related errors"""
    pass

class MediaPipeError(BeautyFilterError):
    """MediaPipe processing errors"""
    pass

class ImageProcessingError(BeautyFilterError):
    """Image processing errors"""
    pass

class FileSystemError(BeautyFilterError):
    """File system related errors"""
    pass

# ------------------------- Advanced Dress Recommendation System -------------------------
class DressRecommendationSystem:
    """Enhanced dress recommendation system with error handling"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.current_season = self.get_current_season()

        # Enhanced color recommendations with error handling
        self.season_colors = self._load_season_colors()
        self.makeup_recommendations = self._load_makeup_recommendations()

        self.logger.info(f"DressRecommendationSystem initialized for season: {self.current_season}")

    def _load_season_colors(self) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        """Load season color recommendations with validation"""
        try:
            return {
                "spring": {
                    "warm": {
                        "light": ["coral", "peach", "mint", "soft yellow", "cream"],
                        "medium": ["salmon", "warm pink", "aqua", "butter yellow", "ivory"],
                        "deep": ["terracotta", "warm coral", "teal", "golden yellow", "champagne"]
                    },
                    "cool": {
                        "light": ["baby pink", "lavender", "powder blue", "mint green", "white"],
                        "medium": ["rose pink", "periwinkle", "sky blue", "sage green", "pearl"],
                        "deep": ["fuchsia", "violet", "royal blue", "emerald", "platinum"]
                    },
                    "neutral": {
                        "light": ["nude", "blush", "sage", "vanilla", "opal"],
                        "medium": ["taupe", "dusty rose", "eucalyptus", "cream", "champagne"],
                        "deep": ["mocha", "mauve", "forest green", "camel", "bronze"]
                    }
                },
                "summer": {
                    "warm": {
                        "light": ["coral", "peach", "turquoise", "sunny yellow", "cream"],
                        "medium": ["orange", "warm red", "aqua blue", "golden yellow", "ivory"],
                        "deep": ["rust", "brick red", "teal", "amber", "bronze"]
                    },
                    "cool": {
                        "light": ["powder blue", "lavender", "mint", "lemon", "white"],
                        "medium": ["sky blue", "cool pink", "seafoam", "silver", "pearl"],
                        "deep": ["navy", "royal purple", "emerald", "platinum", "ice blue"]
                    },
                    "neutral": {
                        "light": ["beige", "soft grey", "sage", "vanilla", "nude"],
                        "medium": ["khaki", "dusty blue", "olive", "taupe", "mushroom"],
                        "deep": ["charcoal", "forest", "chocolate", "pewter", "stone"]
                    }
                },
                "fall": {
                    "warm": {
                        "light": ["caramel", "warm beige", "golden brown", "rust", "amber"],
                        "medium": ["orange", "burgundy", "chocolate", "gold", "copper"],
                        "deep": ["mahogany", "deep red", "espresso", "bronze", "burnt orange"]
                    },
                    "cool": {
                        "light": ["grey", "lavender grey", "dusty blue", "silver", "platinum"],
                        "medium": ["navy", "plum", "forest green", "charcoal", "steel blue"],
                        "deep": ["black", "deep purple", "hunter green", "midnight", "onyx"]
                    },
                    "neutral": {
                        "light": ["mushroom", "warm grey", "olive", "taupe", "sand"],
                        "medium": ["brown", "burgundy", "forest", "pewter", "stone"],
                        "deep": ["chocolate", "wine", "deep olive", "charcoal", "espresso"]
                    }
                },
                "winter": {
                    "warm": {
                        "light": ["cream", "warm white", "camel", "gold", "champagne"],
                        "medium": ["red", "burgundy", "chocolate", "bronze", "copper"],
                        "deep": ["deep red", "mahogany", "espresso", "gold", "burnt orange"]
                    },
                    "cool": {
                        "light": ["pure white", "ice blue", "silver", "platinum", "pearl"],
                        "medium": ["royal blue", "emerald", "black", "navy", "purple"],
                        "deep": ["jet black", "deep purple", "hunter green", "midnight blue", "onyx"]
                    },
                    "neutral": {
                        "light": ["grey", "dove", "silver", "pewter", "stone"],
                        "medium": ["charcoal", "navy", "forest", "black", "steel"],
                        "deep": ["black", "deep grey", "midnight", "charcoal", "onyx"]
                    }
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to load season colors: {e}")
            return {}

    def _load_makeup_recommendations(self) -> Dict[str, Dict[str, Dict[str, List[int]]]]:
        """Load makeup recommendations with validation"""
        try:
            return {
                "warm": {
                    "light": {
                        "lipstick": [0, 4, 7, 9],
                        "blush": [1, 3, 7, 8],
                        "eyeshadow": [0, 1, 4, 8]
                    },
                    "medium": {
                        "lipstick": [1, 2, 4, 8],
                        "blush": [0, 1, 3, 9],
                        "eyeshadow": [0, 1, 4, 9]
                    },
                    "deep": {
                        "lipstick": [2, 3, 5, 8],
                        "blush": [2, 3, 6, 8],
                        "eyeshadow": [0, 5, 6, 9]
                    }
                },
                "cool": {
                    "light": {
                        "lipstick": [4, 6, 7, 9],
                        "blush": [0, 4, 7, 9],
                        "eyeshadow": [2, 3, 7, 6]
                    },
                    "medium": {
                        "lipstick": [1, 2, 6, 9],
                        "blush": [2, 4, 5, 9],
                        "eyeshadow": [2, 3, 5, 7]
                    },
                    "deep": {
                        "lipstick": [3, 5, 6, 8],
                        "blush": [2, 5, 6, 8],
                        "eyeshadow": [5, 7, 2, 9]
                    }
                },
                "neutral": {
                    "light": {
                        "lipstick": [0, 4, 7, 9],
                        "blush": [0, 1, 4, 9],
                        "eyeshadow": [0, 6, 7, 8]
                    },
                    "medium": {
                        "lipstick": [1, 2, 4, 8],
                        "blush": [0, 2, 3, 8],
                        "eyeshadow": [0, 1, 6, 9]
                    },
                    "deep": {
                        "lipstick": [2, 3, 5, 8],
                        "blush": [2, 3, 5, 8],
                        "eyeshadow": [0, 5, 7, 9]
                    }
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to load makeup recommendations: {e}")
            return {}

    def get_current_season(self) -> str:
        """Get current season with error handling"""
        try:
            month = datetime.now().month
            season_map = {
                12: "winter", 1: "winter", 2: "winter",
                3: "spring", 4: "spring", 5: "spring",
                6: "summer", 7: "summer", 8: "summer",
                9: "fall", 10: "fall", 11: "fall"
            }
            return season_map.get(month, "spring")  # default fallback
        except Exception as e:
            self.logger.error(f"Error determining current season: {e}")
            return "spring"


    def classify_skin_tone_depth(self, skin_tone_bgr: Optional[np.ndarray]) -> str:
        """Classify skin tone depth with enhanced error handling"""
        try:
            if skin_tone_bgr is None or len(skin_tone_bgr) != 3:
                return "medium"

            b, g, r = map(float, skin_tone_bgr[:3])

            # Validate color values
            if not all(0 <= val <= 255 for val in [b, g, r]):
                self.logger.warning(f"Invalid skin tone values: {skin_tone_bgr}")
                return "medium"

            # Calculate overall lightness using weighted average
            lightness = (0.299 * r + 0.587 * g + 0.114 * b)

            if lightness > 180:
                return "light"
            elif lightness > 120:
                return "medium"
            else:
                return "deep"

        except Exception as e:
            self.logger.error(f"Error classifying skin tone depth: {e}")
            return "medium"

    def classify_skin_tone_type(self, skin_tone_bgr: Optional[np.ndarray]) -> str:
        """Enhanced skin tone classification with better error handling"""
        try:
            if skin_tone_bgr is None or len(skin_tone_bgr) != 3:
                return "neutral"

            b, g, r = map(float, skin_tone_bgr[:3])

            # Validate color values
            if not all(0 <= val <= 255 for val in [b, g, r]):
                self.logger.warning(f"Invalid skin tone values: {skin_tone_bgr}")
                return "neutral"

            # Prevent division by zero with safe denominators
            safe_denom_1 = max(b, 1.0)
            safe_denom_2 = max(r + g, 1.0)
            safe_denom_3 = max(r + b, 1.0)

            # More sophisticated undertone analysis
            red_yellow_ratio = (r + g) / safe_denom_1
            blue_dominance = b / safe_denom_2
            green_balance = g / safe_denom_3

            # Enhanced classification logic
            if red_yellow_ratio > 1.4 and blue_dominance < 0.35:
                return "warm"
            elif blue_dominance > 0.42 or (b > r and b > g):
                return "cool"
            else:
                return "neutral"

        except Exception as e:
            self.logger.error(f"Error classifying skin tone type: {e}")
            return "neutral"

    def get_skin_tone_analysis(self, skin_tone_bgr: Optional[np.ndarray]) -> Tuple[str, str, str]:
        """Complete skin tone analysis with error handling"""
        try:
            if skin_tone_bgr is None:
                return "neutral", "medium", "Unknown skin tone (calibration needed)"

            undertone = self.classify_skin_tone_type(skin_tone_bgr)
            depth = self.classify_skin_tone_depth(skin_tone_bgr)

            # Generate description
            depth_desc = {"light": "Light", "medium": "Medium", "deep": "Deep"}
            undertone_desc = {"warm": "Warm", "cool": "Cool", "neutral": "Neutral"}

            depth_text = depth_desc.get(depth, "Medium")
            undertone_text = undertone_desc.get(undertone, "Neutral")

            description = f"{depth_text} skin with {undertone_text.lower()} undertones"

            return undertone, depth, description

        except Exception as e:
            self.logger.error(f"Error in skin tone analysis: {e}")
            return "neutral", "medium", "Analysis failed"

    def recommend_dress_colors(self, skin_tone_bgr: Optional[np.ndarray], season: Optional[str] = None) -> Tuple[List[str], str, str, str]:
        """Advanced dress color recommendations with error handling"""
        try:
            if season is None:
                season = self.current_season

            if season not in self.season_colors:
                self.logger.warning(f"Invalid season: {season}, using current season")
                season = self.current_season

            undertone, depth, description = self.get_skin_tone_analysis(skin_tone_bgr)

            # Get recommendations with fallback
            try:
                recommended_colors = self.season_colors[season][undertone][depth]
            except KeyError:
                # Fallback to neutral medium if classification fails
                self.logger.warning(f"No recommendations for {undertone}-{depth}, using fallback")
                recommended_colors = self.season_colors.get(season, {}).get("neutral", {}).get("medium", [])

                # Final fallback
                if not recommended_colors:
                    recommended_colors = ["navy", "black", "white", "beige", "grey"]

            return recommended_colors, undertone, depth, description

        except Exception as e:
            self.logger.error(f"Error in dress color recommendations: {e}")
            return ["navy", "black", "white"], "neutral", "medium", "Error in analysis"

    def recommend_makeup_colors(self, skin_tone_bgr: Optional[np.ndarray]) -> Dict[str, List[int]]:
        """Recommend makeup colors with error handling"""
        try:
            if skin_tone_bgr is None:
                return {
                    "lipstick": [0, 1, 2],
                    "blush": [0, 1, 2],
                    "eyeshadow": [0, 1, 2]
                }

            undertone, depth, _ = self.get_skin_tone_analysis(skin_tone_bgr)

            try:
                recommendations = self.makeup_recommendations[undertone][depth]

                # Validate recommendations
                for makeup_type, indices in recommendations.items():
                    if not isinstance(indices, list) or not indices:
                        recommendations[makeup_type] = [0, 1, 2]

                return recommendations

            except KeyError:
                # Fallback to neutral medium
                return self.makeup_recommendations.get("neutral", {}).get("medium", {
                    "lipstick": [0, 1, 2],
                    "blush": [0, 1, 2],
                    "eyeshadow": [0, 1, 2]
                })

        except Exception as e:
            self.logger.error(f"Error in makeup color recommendations: {e}")
            return {
                "lipstick": [0, 1, 2],
                "blush": [0, 1, 2],
                "eyeshadow": [0, 1, 2]
            }

    def analyze_dress_suitability(self, dress_name: str, skin_tone_bgr: Optional[np.ndarray], season: Optional[str] = None) -> Tuple[int, str, List[str]]:
        """Analyze dress suitability with comprehensive error handling"""
        try:
            if not dress_name or not isinstance(dress_name, str):
                return 5, "Unable to analyze: invalid dress name", []

            if season is None:
                season = self.current_season

            recommended_colors, undertone, depth, _ = self.recommend_dress_colors(skin_tone_bgr, season)

            if not recommended_colors:
                return 5, "Unable to analyze: no color recommendations available", []

            dress_name_lower = dress_name.lower()

            # Check for exact color matches
            exact_matches = []
            partial_matches = []

            for color in recommended_colors:
                if not color:
                    continue

                color_lower = color.lower()
                if color_lower in dress_name_lower:
                    exact_matches.append(color)
                else:
                    # Check for partial matches
                    color_parts = color_lower.split()
                    for part in color_parts:
                        if len(part) > 2 and part in dress_name_lower:
                            partial_matches.append(color)
                            break

            # Calculate suitability score
            if exact_matches:
                score = 10
                reason = f"Perfect match! Contains {', '.join(exact_matches)} which is ideal for your {undertone} {depth} skin tone."
            elif partial_matches:
                score = 7
                reason = f"Good match! {', '.join(partial_matches)} works well with your {undertone} {depth} skin tone."
            else:
                # Check complementary colors
                complementary_score = self._check_complementary_colors(dress_name_lower, undertone, depth)
                if complementary_score > 0:
                    score = complementary_score
                    reason = f"Decent choice for your {undertone} {depth} skin tone, though not optimal for {season}."
                else:
                    score = 3
                    reason = f"May not be the most flattering for your {undertone} {depth} skin tone in {season}."

            return score, reason, exact_matches + partial_matches

        except Exception as e:
            self.logger.error(f"Error analyzing dress suitability: {e}")
            return 5, "Analysis failed due to error", []

    def _check_complementary_colors(self, dress_name: str, undertone: str, depth: str) -> int:
        """Check complementary colors with error handling"""
        try:
            if not dress_name or not isinstance(dress_name, str):
                return 0

            # Universal flattering colors
            universal_colors = ["navy", "black", "white", "cream", "beige"]

            # Colors that work across different undertones
            color_maps = {
                "warm": ["coral", "peach", "gold", "brown"],
                "cool": ["navy", "purple", "emerald", "silver"],
                "neutral": ["grey", "taupe", "olive", "burgundy"]
            }

            # Check universal colors first
            for color in universal_colors:
                if color in dress_name:
                    return 6

            # Check undertone-specific colors
            versatile_colors = color_maps.get(undertone, [])
            for color in versatile_colors:
                if color in dress_name:
                    return 5

            return 0

        except Exception as e:
            self.logger.error(f"Error checking complementary colors: {e}")
            return 0

# ------------------------- Enhanced Beauty Filter -------------------------
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

    def create_smooth_mask(self, frame_shape: Tuple[int, int], landmarks, indices: List[int], blur_size: int = 25) -> Optional[np.ndarray]:
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

    def apply_enhanced_color(self, frame: np.ndarray, mask: Optional[np.ndarray], color: Tuple[int, int, int], intensity: float, blend_mode: str = 'normal') -> np.ndarray:
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

# ------------------------- Enhanced Skin Tone Detection -------------------------
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

    def extract_face_skin_color(self, frame: np.ndarray, face_landmarks, frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
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

                        self.logger.info(f"Skin tone calibrated in {calibration_time:.1f}s: BGR{tuple(self.calibrated_skin_tone)}")
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

# ------------------------- Enhanced Dress Management -------------------------
class DressManager:
    """Enhanced dress management with comprehensive error handling"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.assets_dir = Path(config.ASSETS_DIR)

        self.seasons = ["spring", "summer", "fall", "winter"]
        self.seasonal_dresses = {}
        self.current_season_idx = 0
        self.current_dress_idx = 0

        self._load_dresses()

    def _load_dresses(self):
        """Load dresses with comprehensive error handling"""
        try:
            self.seasonal_dresses = {season: [] for season in self.seasons}
            total_loaded = 0

            # Create directories if they don't exist
            for season in self.seasons:
                season_path = self.assets_dir / season
                season_path.mkdir(parents=True, exist_ok=True)

            # Load seasonal dresses
            for season in self.seasons:
                season_path = self.assets_dir / season
                if season_path.exists():
                    loaded_count = self._load_season_dresses(season_path, season)
                    total_loaded += loaded_count
                    self.logger.info(f"Loaded {loaded_count} dresses for {season}")

            # Fallback: load from main directory if no seasonal dresses
            if total_loaded == 0:
                main_loaded = self._load_main_directory_dresses()
                total_loaded += main_loaded
                if main_loaded > 0:
                    self.logger.info(f"Loaded {main_loaded} dresses from main directory")

            self.logger.info(f"Total dresses loaded: {total_loaded}")

            if total_loaded == 0:
                self.logger.warning("No dresses loaded. Please add PNG files to the assets directory.")
                self._create_sample_dress_structure()

        except Exception as e:
            self.logger.error(f"Error loading dresses: {e}")
            self._create_sample_dress_structure()

    def _load_season_dresses(self, season_path: Path, season: str) -> int:
        """Load dresses for a specific season"""
        try:
            loaded_count = 0
            supported_extensions = {".png", ".PNG"}

            if not season_path.exists():
                return 0

            files = sorted([p for p in season_path.glob("*") if p.suffix in supported_extensions])

            for file_path in files:
                try:
                    img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)

                    if img is not None:
                        # Check if image has alpha channel
                        if len(img.shape) == 3 and img.shape[2] == 4:
                            self.seasonal_dresses[season].append((file_path.name, img))
                            loaded_count += 1
                        elif len(img.shape) == 3 and img.shape[2] == 3:
                            # Convert RGB to RGBA
                            rgba_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                            self.seasonal_dresses[season].append((file_path.name, rgba_img))
                            loaded_count += 1
                            self.logger.info(f"Converted RGB to RGBA: {file_path.name}")
                        else:
                            self.logger.warning(f"Unsupported image format: {file_path.name}")
                    else:
                        self.logger.warning(f"Failed to load image: {file_path.name}")

                except Exception as e:
                    self.logger.error(f"Error processing {file_path.name}: {e}")
                    continue

            return loaded_count

        except Exception as e:
            self.logger.error(f"Error loading season dresses for {season}: {e}")
            return 0

    def _load_main_directory_dresses(self) -> int:
        """Load dresses from main directory and distribute across seasons"""
        try:
            loaded_count = 0
            files = sorted([p for p in self.assets_dir.glob("*.png")])

            for file_path in files:
                try:
                    img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)

                    if img is not None and len(img.shape) == 3 and img.shape[2] == 4:
                        # Distribute dresses across seasons using hash
                        season_idx = hash(file_path.name) % len(self.seasons)
                        season = self.seasons[season_idx]
                        self.seasonal_dresses[season].append((file_path.name, img))
                        loaded_count += 1

                except Exception as e:
                    self.logger.error(f"Error processing main directory dress {file_path.name}: {e}")
                    continue

            return loaded_count

        except Exception as e:
            self.logger.error(f"Error loading main directory dresses: {e}")
            return 0

    def get_current_dress(self) -> Optional[Tuple[str, np.ndarray]]:
        """Get current dress with error handling"""
        try:
            season = self.seasons[self.current_season_idx]
            dresses = self.seasonal_dresses.get(season, [])

            if dresses and 0 <= self.current_dress_idx < len(dresses):
                return dresses[self.current_dress_idx]
            return None

        except Exception as e:
            self.logger.error(f"Error getting current dress: {e}")
            return None

    def next_dress(self) -> Optional[str]:
        """Navigate to next dress safely"""
        try:
            season = self.seasons[self.current_season_idx]
            dresses = self.seasonal_dresses.get(season, [])

            if dresses:
                self.current_dress_idx = (self.current_dress_idx + 1) % len(dresses)
                return dresses[self.current_dress_idx][0]
            return None

        except Exception as e:
            self.logger.error(f"Error navigating to next dress: {e}")
            return None

    def previous_dress(self) -> Optional[str]:
        """Navigate to previous dress safely"""
        try:
            season = self.seasons[self.current_season_idx]
            dresses = self.seasonal_dresses.get(season, [])

            if dresses:
                self.current_dress_idx = (self.current_dress_idx - 1) % len(dresses)
                return dresses[self.current_dress_idx][0]
            return None

        except Exception as e:
            self.logger.error(f"Error navigating to previous dress: {e}")
            return None

    def next_season(self) -> str:
        """Navigate to next season safely"""
        try:
            self.current_season_idx = (self.current_season_idx + 1) % len(self.seasons)
            self.current_dress_idx = 0  # Reset to first dress in new season
            season = self.seasons[self.current_season_idx]
            return season

        except Exception as e:
            self.logger.error(f"Error navigating to next season: {e}")
            return self.seasons[0]

    def get_season_info(self) -> Tuple[str, int]:
        """Get current season and dress count"""
        try:
            season = self.seasons[self.current_season_idx]
            dress_count = len(self.seasonal_dresses.get(season, []))
            return season, dress_count
        except Exception as e:
            self.logger.error(f"Error getting season info: {e}")
            return "spring", 0

    def find_best_dress(self, recommended_colors: List[str]) -> Optional[Tuple[int, str, int]]:
        """Find best matching dress for recommended colors"""
        try:
            season = self.seasons[self.current_season_idx]
            dresses = self.seasonal_dresses.get(season, [])

            if not dresses or not recommended_colors:
                return None

            best_score = 0
            best_index = 0
            best_matches = []

            for i, (name, _) in enumerate(dresses):
                score = 0
                matches = []
                name_lower = name.lower()

                # Check for color matches in filename
                for color in recommended_colors:
                    if color.lower() in name_lower:
                        score += 10
                        matches.append(color)
                    else:
                        # Check for partial matches
                        for word in color.lower().split():
                            if len(word) > 2 and word in name_lower:
                                score += 5
                                matches.append(color)
                                break

                if score > best_score:
                    best_score = score
                    best_index = i
                    best_matches = matches

            if best_score > 0:
                return best_index, dresses[best_index][0], best_score
            return None

        except Exception as e:
            self.logger.error(f"Error finding best dress: {e}")
            return None

# ------------------------- Enhanced Image Processing Utilities -------------------------
class ImageProcessor:
    """Enhanced image processing with comprehensive error handling"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def validate_image(self, img: np.ndarray, required_channels: int = 4) -> bool:
        """Validate image data"""
        try:
            if img is None or not isinstance(img, np.ndarray):
                return False

            if len(img.shape) != 3:
                return False

            h, w, c = img.shape
            if h <= 0 or w <= 0 or c != required_channels:
                return False

            return True

        except Exception:
            return False

    def overlay_rgba_onto_bgr(self, frame_bgr: np.ndarray, overlay_rgba: np.ndarray, center_xy: Tuple[float, float]) -> np.ndarray:
        """Alpha blend overlay onto frame with comprehensive error handling"""
        try:
            # Validate inputs
            if not self.validate_image(frame_bgr, 3):
                self.logger.error("Invalid BGR frame for overlay")
                return frame_bgr if frame_bgr is not None else np.zeros((480, 640, 3), dtype=np.uint8)

            if not self.validate_image(overlay_rgba, 4):
                self.logger.warning("Invalid RGBA overlay")
                return frame_bgr

            if not center_xy or len(center_xy) != 2:
                self.logger.warning("Invalid center coordinates")
                return frame_bgr

            h, w = overlay_rgba.shape[:2]
            cx, cy = int(center_xy[0]), int(center_xy[1])

            # Calculate overlay bounds
            x1 = cx - w // 2
            y1 = cy - h // 2
            x2 = x1 + w
            y2 = y1 + h

            # Get frame dimensions
            fh, fw = frame_bgr.shape[:2]

            # Clip to frame bounds
            x1_clip = max(0, x1)
            y1_clip = max(0, y1)
            x2_clip = min(fw, x2)
            y2_clip = min(fh, y2)

            # Check if overlay is within frame
            if x1_clip >= x2_clip or y1_clip >= y2_clip:
                self.logger.debug("Overlay completely outside frame bounds")
                return frame_bgr

            # Calculate overlay crop coordinates
            ox1 = x1_clip - x1
            oy1 = y1_clip - y1
            ox2 = ox1 + (x2_clip - x1_clip)
            oy2 = oy1 + (y2_clip - y1_clip)

            # Validate crop coordinates
            if ox1 < 0 or oy1 < 0 or ox2 > w or oy2 > h:
                self.logger.warning("Invalid overlay crop coordinates")
                return frame_bgr

            # Extract regions
            roi = frame_bgr[y1_clip:y2_clip, x1_clip:x2_clip]
            overlay_crop = overlay_rgba[oy1:oy2, ox1:ox2]

            # Validate region shapes
            if roi.shape[:2] != overlay_crop.shape[:2]:
                self.logger.error(f"Region shape mismatch: {roi.shape[:2]} vs {overlay_crop.shape[:2]}")
                return frame_bgr

            # Perform alpha blending
            overlay_rgb = overlay_crop[..., :3].astype(np.float32)
            alpha = overlay_crop[..., 3:4].astype(np.float32) / 255.0

            roi_float = roi.astype(np.float32)
            blended = alpha * overlay_rgb + (1.0 - alpha) * roi_float

            # Update frame
            frame_bgr[y1_clip:y2_clip, x1_clip:x2_clip] = blended.astype(np.uint8)

            return frame_bgr

        except Exception as e:
            self.logger.error(f"Error in RGBA overlay: {e}")
            return frame_bgr

    def scale_and_rotate_rgba(self, img_rgba: np.ndarray, target_w: int, target_h: int, angle_deg: float) -> Optional[np.ndarray]:
        """Resize and rotate image with comprehensive error handling"""
        try:
            if not self.validate_image(img_rgba, 4):
                self.logger.error("Invalid RGBA image for scaling")
                return None

            # Validate and clamp dimensions
            target_w = max(1, min(int(target_w), 4000))
            target_h = max(1, min(int(target_h), 4000))

            # Clamp angle
            angle_deg = angle_deg % 360

            # Resize image
            try:
                resized = cv2.resize(img_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)
            except Exception as e:
                self.logger.error(f"Error resizing image: {e}")
                return img_rgba

            # Rotate image
            try:
                (h, w) = resized.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

                rotated = cv2.warpAffine(
                    resized, M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0)
                )

                return rotated

            except Exception as e:
                self.logger.error(f"Error rotating image: {e}")
                return resized

        except Exception as e:
            self.logger.error(f"Error in scale and rotate: {e}")
            return img_rgba

    def calculate_pose_metrics(self, pose_landmarks, frame_shape: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """Calculate pose metrics with error handling"""
        try:
            if not pose_landmarks or not hasattr(pose_landmarks, 'landmark'):
                return None

            if len(frame_shape) < 2:
                return None

            h, w = frame_shape[:2]
            landmarks = pose_landmarks.landmark

            # Validate landmark count
            if len(landmarks) < 25:  # Minimum required landmarks
                return None

            def get_xy(i):
                if i < len(landmarks):
                    lm = landmarks[i]
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    # Validate coordinates
                    if 0 <= x < w and 0 <= y < h:
                        return (x, y)
                return None

            # Get key points with validation
            points = {
                'left_shoulder': get_xy(11),    # MediaPipe left shoulder
                'right_shoulder': get_xy(12),   # MediaPipe right shoulder
                'left_hip': get_xy(23),         # MediaPipe left hip
                'right_hip': get_xy(24),        # MediaPipe right hip
            }

            # Check if all required points are valid
            if not all(point is not None for point in points.values()):
                missing = [k for k, v in points.items() if v is None]
                self.logger.debug(f"Missing pose points: {missing}")
                return None

            # Calculate derived points
            shoulder_mid = (
                (points['left_shoulder'][0] + points['right_shoulder'][0]) // 2,
                (points['left_shoulder'][1] + points['right_shoulder'][1]) // 2
            )

            hip_mid = (
                (points['left_hip'][0] + points['right_hip'][0]) // 2,
                (points['left_hip'][1] + points['right_hip'][1]) // 2
            )

            torso_mid = (
                (shoulder_mid[0] + hip_mid[0]) // 2,
                (shoulder_mid[1] + hip_mid[1]) // 2
            )

            # Calculate metrics
            shoulder_width = self._distance(points['left_shoulder'], points['right_shoulder'])
            torso_height = self._distance(shoulder_mid, hip_mid)

            # Calculate rotation angle
            angle = degrees(atan2(
                points['right_shoulder'][1] - points['left_shoulder'][1],
                points['right_shoulder'][0] - points['left_shoulder'][0]
            ))

            # Normalize angle
            if angle < -90:
                angle += 180
            elif angle > 90:
                angle -= 180

            return {
                'points': points,
                'shoulder_mid': shoulder_mid,
                'hip_mid': hip_mid,
                'torso_mid': torso_mid,
                'shoulder_width': shoulder_width,
                'torso_height': torso_height,
                'angle': angle
            }

        except Exception as e:
            self.logger.error(f"Error calculating pose metrics: {e}")
            return None

    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        try:
            return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        except Exception:
            return 0.0

# ------------------------- Enhanced Camera Manager -------------------------
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

# ------------------------- Enhanced UI Manager -------------------------
class UIManager:
    """Enhanced UI management with comprehensive error handling"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # UI settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 1
        self.line_height = 25

        # Colors
        self.text_color = (255, 255, 255)
        self.background_color = (0, 0, 0)
        self.highlight_color = (0, 255, 0)
        self.warning_color = (0, 255, 255)

    def draw_text_with_background(self, frame: np.ndarray, text: str, position: Tuple[int, int],
                                 font_scale: float = None, color: Tuple[int, int, int] = None,
                                 background: bool = True) -> np.ndarray:
        """Draw text with optional background"""
        try:
            if frame is None or not text:
                return frame

            font_scale = font_scale or self.font_scale
            color = color or self.text_color

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, self.font, font_scale, self.thickness)

            x, y = position

            # Validate position
            h, w = frame.shape[:2]
            if x < 0 or y < 0 or x >= w or y >= h:
                return frame

            # Draw background if requested
            if background:
                padding = 2
                cv2.rectangle(frame,
                            (x - padding, y - text_height - padding),
                            (x + text_width + padding, y + baseline + padding),
                            self.background_color, -1)

            # Draw text
            cv2.putText(frame, text, (x, y), self.font, font_scale, color, self.thickness)

            return frame

        except Exception as e:
            self.logger.error(f"Error drawing text: {e}")
            return frame

    def draw_progress_bar(self, frame: np.ndarray, progress: float, position: Tuple[int, int],
                         size: Tuple[int, int] = (200, 20), color: Tuple[int, int, int] = None) -> np.ndarray:
        """Draw progress bar"""
        try:
            if frame is None:
                return frame

            progress = max(0.0, min(1.0, progress))
            color = color or self.highlight_color

            x, y = position
            w, h = size

            # Validate bounds
            frame_h, frame_w = frame.shape[:2]
            if x < 0 or y < 0 or x + w >= frame_w or y + h >= frame_h:
                return frame

            # Draw background
            cv2.rectangle(frame, (x, y), (x + w, y + h), (64, 64, 64), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

            # Draw progress
            progress_width = int(w * progress)
            if progress_width > 0:
                cv2.rectangle(frame, (x + 1, y + 1), (x + progress_width - 1, y + h - 1), color, -1)

            return frame

        except Exception as e:
            self.logger.error(f"Error drawing progress bar: {e}")
            return frame

    def draw_status_panel(self, frame: np.ndarray, season: str, dress_name: str,
                         calibration_progress: float, makeup_status: Dict[str, Any]) -> np.ndarray:
        """Draw comprehensive status panel"""
        try:
            if frame is None:
                return frame

            y_offset = 20

            # Main status line
            status_text = f"Season: {season.title()} | Dress: {dress_name}"
            if calibration_progress < 100:
                status_text += f" | Calibrating: {calibration_progress:.0f}%"

            frame = self.draw_text_with_background(frame, status_text, (20, y_offset), 0.7)
            y_offset += self.line_height

            # Calibration progress bar
            if calibration_progress < 100:
                frame = self.draw_progress_bar(frame, calibration_progress / 100, (20, y_offset))
                y_offset += 30

            # Makeup status
            makeup_items = []
            if makeup_status.get('enabled', False):
                if makeup_status.get('lipstick', False):
                    makeup_items.append(f"Lipstick({makeup_status.get('lipstick_idx', 0) + 1})")
                if makeup_status.get('blush', False):
                    makeup_items.append(f"Blush({makeup_status.get('blush_idx', 0) + 1})")
                if makeup_status.get('eyeshadow', False):
                    makeup_items.append(f"Eyeshadow({makeup_status.get('eyeshadow_idx', 0) + 1})")

            makeup_text = "Makeup: " + (", ".join(makeup_items) if makeup_items else "OFF")
            makeup_text += f" | Intensity: {makeup_status.get('intensity', 0.5):.1f}"

            frame = self.draw_text_with_background(frame, makeup_text, (20, y_offset), 0.6)
            y_offset += self.line_height

            # Controls help
            controls = [
                "[ / ] dress | d recommend | o season | 1/2/3 colors | 4 best look | h help",
                "l/b/e toggle | +/- intensity | m makeup | s save | r recalibrate | q quit"
            ]

            for control_text in controls:
                frame = self.draw_text_with_background(frame, control_text, (20, y_offset), 0.5, (200, 200, 200))
                y_offset += self.line_height - 5

            return frame

        except Exception as e:
            self.logger.error(f"Error drawing status panel: {e}")
            return frame

    def show_help_menu(self, frame: np.ndarray) -> np.ndarray:
        """Show comprehensive help menu"""
        try:
            if frame is None:
                return frame

            # Create semi-transparent overlay
            overlay = frame.copy()
            h, w = frame.shape[:2]

            # Help panel background
            panel_x, panel_y = 50, 50
            panel_w, panel_h = w - 100, h - 100

            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                         (0, 0, 0), -1)

            # Blend overlay
            alpha = 0.8
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Help content
            help_lines = [
                "SMART BEAUTY FILTER - HELP",
                "=" * 30,
                "",
                "DRESS CONTROLS:",
                "  [ ]     - Previous/Next dress",
                "  d       - Get personalized dress recommendations",
                "  o       - Change season (Spring/Summer/Fall/Winter)",
                "",
                "MAKEUP CONTROLS:",
                "  1/2/3   - Cycle through lipstick/blush/eyeshadow colors",
                "  l/b/e   - Toggle lipstick/blush/eyeshadow on/off",
                "  4       - Apply best recommended makeup look",
                "  5       - Cycle through recommended colors only",
                "  m       - Toggle all makeup on/off",
                "  +/-     - Increase/decrease makeup intensity",
                "",
                "SYSTEM CONTROLS:",
                "  s       - Save current frame to ./output/",
                "  t       - Show skin tone information",
                "  r       - Reset/recalibrate skin tone",
                "  c       - Show configuration menu",
                "  h       - Show/hide this help menu",
                "  q       - Quit with style profile summary",
                "",
                "TIPS:",
                "  • Look directly at camera for accurate skin tone detection",
                "  • Use dress filenames with color keywords for better recommendations",
                "  • Calibration takes ~5 seconds for accurate results",
                "",
                "Press 'h' again to close this help menu"
            ]

            text_y = panel_y + 30
            for line in help_lines:
                if line.startswith("SMART BEAUTY FILTER"):
                    color = self.highlight_color
                    font_scale = 0.8
                elif line.startswith("="):
                    color = self.highlight_color
                    font_scale = 0.6
                elif line.endswith(":"):
                    color = self.warning_color
                    font_scale = 0.7
                else:
                    color = self.text_color
                    font_scale = 0.6

                cv2.putText(frame, line, (panel_x + 20, text_y), self.font, font_scale, color, 1)
                text_y += int(self.line_height * (1.2 if font_scale > 0.6 else 1.0))

                if text_y > panel_y + panel_h - 30:
                    break

            return frame

        except Exception as e:
            self.logger.error(f"Error showing help menu: {e}")
            return frame

# ------------------------- Main Application Class -------------------------
class SmartBeautyFilter:
    """Main application class with comprehensive error handling"""

    def __init__(self):
        # Load configuration
        self.config = Config.load("../beauty_filter_config.json")

        # Setup logging
        self.logger = setup_logging(self.config.LOGS_DIR)
        self.logger.info("Smart Beauty Filter starting up...")

        # Initialize components
        try:
            self.camera_manager = CameraManager(self.config)
            self.dress_manager = DressManager(self.config)
            self.skin_detector = SkinToneDetector(self.config)
            self.beauty_filter = BeautyFilter(self.config)
            self.dress_recommender = DressRecommendationSystem(self.config)
            self.image_processor = ImageProcessor(self.config)
            self.ui_manager = UIManager(self.config)

            # Application state
            self.running = False
            self.show_help = False
            self.frame_count = 0
            self.start_time = time.time()

        except Exception as e:
            self.logger.critical(f"Failed to initialize components: {e}")
            raise

    def initialize(self) -> bool:
        """Initialize the application"""
        try:
            self.logger.info("Initializing Smart Beauty Filter...")

            # Initialize camera
            if not self.camera_manager.initialize():
                return False

            # Initialize MediaPipe
            if not self._initialize_mediapipe():
                return False

            # Start skin tone calibration
            self.skin_detector.start_calibration()

            # Create output directory
            Path(self.config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

            self.logger.info("Initialization complete")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    def _initialize_mediapipe(self) -> bool:
        """Initialize MediaPipe components"""
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_face_mesh = mp.solutions.face_mesh

            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            return True

        except Exception as e:
            self.logger.error(f"MediaPipe initialization failed: {e}")
            return False

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with comprehensive error handling"""
        try:
            if frame is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)

            h, w = frame.shape[:2]

            # Convert BGR to RGB for MediaPipe
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                self.logger.warning(f"Color conversion failed: {e}")
                return frame

            # Process with MediaPipe
            pose_results = None
            face_results = None

            try:
                pose_results = self.pose.process(rgb)
                face_results = self.face_mesh.process(rgb)
            except Exception as e:
                self.logger.warning(f"MediaPipe processing failed: {e}")

            # Skin tone calibration
            if (self.skin_detector.is_calibrating and face_results and
                hasattr(face_results, 'multi_face_landmarks') and face_results.multi_face_landmarks):
                try:
                    self.skin_detector.calibrate_skin_tone(frame, face_results.multi_face_landmarks[0], (h, w))
                except Exception as e:
                    self.logger.warning(f"Skin tone calibration error: {e}")

            # Apply beauty filters
            if (self.beauty_filter.makeup_enabled and face_results and
                hasattr(face_results, 'multi_face_landmarks') and face_results.multi_face_landmarks):
                try:
                    face_landmarks = face_results.multi_face_landmarks[0]

                    # Apply makeup in correct order
                    frame = self.beauty_filter.apply_eyeshadow(frame, face_landmarks)
                    frame = self.beauty_filter.apply_blush(frame, face_landmarks)
                    frame = self.beauty_filter.apply_lipstick(frame, face_landmarks)

                except Exception as e:
                    self.logger.warning(f"Beauty filter application error: {e}")

            # Apply dress overlay
            if pose_results and hasattr(pose_results, 'pose_landmarks') and pose_results.pose_landmarks:
                try:
                    current_dress = self.dress_manager.get_current_dress()
                    if current_dress:
                        name, dress_rgba = current_dress
                        pose_metrics = self.image_processor.calculate_pose_metrics(pose_results.pose_landmarks, (h, w))

                        if pose_metrics:
                            # Calculate dress positioning
                            target_w = max(150, min(self.config.WIDTH_FACTOR * pose_metrics['shoulder_width'], w))
                            target_h = max(230, min(self.config.HEIGHT_FACTOR * pose_metrics['torso_height'], h))
                            offset_y = self.config.VERTICAL_OFFSET_FACTOR * pose_metrics['torso_height']

                            # Keep dress visible if torso is cut off
                            dress_center = (
                                pose_metrics['torso_mid'][0],
                                min(int(pose_metrics['torso_mid'][1] + offset_y), h - target_h // 2)
                            )

                            # Scale and rotate dress
                            processed_dress = self.image_processor.scale_and_rotate_rgba(
                                dress_rgba, target_w, target_h, pose_metrics['angle']
                            )

                            if processed_dress is not None:
                                frame = self.image_processor.overlay_rgba_onto_bgr(
                                    frame, processed_dress, dress_center
                                )

                except Exception as e:
                    self.logger.warning(f"Dress overlay error: {e}")

            return frame

        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return frame

    def handle_keyboard_input(self, key: int) -> bool:
        """Handle keyboard input with comprehensive error handling"""
        try:
            if key == -1:  # No key pressed
                return True

            key = key & 0xFF

            # Quit
            if key == ord('q'):
                self.logger.info("User requested quit")
                return False

            # Help menu
            elif key == ord('h'):
                self.show_help = not self.show_help
                self.logger.info(f"Help menu {'shown' if self.show_help else 'hidden'}")

            # Save frame
            elif key == ord('s'):
                self._save_current_frame()

            # Dress controls
            elif key == ord(']'):
                dress_name = self.dress_manager.next_dress()
                if dress_name:
                    self.logger.info(f"Next dress: {dress_name}")

            elif key == ord('['):
                dress_name = self.dress_manager.previous_dress()
                if dress_name:
                    self.logger.info(f"Previous dress: {dress_name}")

            # Season control
            elif key == ord('o'):
                season = self.dress_manager.next_season()
                self.logger.info(f"Changed to season: {season}")

            # Dress recommendation
            elif key == ord('d'):
                self._handle_dress_recommendation()

            # Makeup color cycling
            elif key == ord('1'):
                idx, name = self.beauty_filter.cycle_color('lipstick')
                is_recommended = self._check_if_recommended('lipstick', idx)
                rec_text = " ✨ (Recommended!)" if is_recommended else ""
                self.logger.info(f"Lipstick: {name}{rec_text}")

            elif key == ord('2'):
                idx, name = self.beauty_filter.cycle_color('blush')
                is_recommended = self._check_if_recommended('blush', idx)
                rec_text = " ✨ (Recommended!)" if is_recommended else ""
                self.logger.info(f"Blush: {name}{rec_text}")

            elif key == ord('3'):
                idx, name = self.beauty_filter.cycle_color('eyeshadow')
                is_recommended = self._check_if_recommended('eyeshadow', idx)
                rec_text = " ✨ (Recommended!)" if is_recommended else ""
                self.logger.info(f"Eyeshadow: {name}{rec_text}")

            # Quick makeup application
            elif key == ord('4'):
                self._apply_best_makeup()

            elif key == ord('5'):
                self._cycle_recommended_colors()

            # Makeup toggles
            elif key == ord('l'):
                self.beauty_filter.lipstick_enabled = not self.beauty_filter.lipstick_enabled
                self.logger.info(f"Lipstick: {'ON' if self.beauty_filter.lipstick_enabled else 'OFF'}")

            elif key == ord('b'):
                self.beauty_filter.blush_enabled = not self.beauty_filter.blush_enabled
                self.logger.info(f"Blush: {'ON' if self.beauty_filter.blush_enabled else 'OFF'}")

            elif key == ord('e'):
                self.beauty_filter.eyeshadow_enabled = not self.beauty_filter.eyeshadow_enabled
                self.logger.info(f"Eyeshadow: {'ON' if self.beauty_filter.eyeshadow_enabled else 'OFF'}")

            elif key == ord('m'):
                self.beauty_filter.makeup_enabled = not self.beauty_filter.makeup_enabled
                self.logger.info(f"All makeup: {'ON' if self.beauty_filter.makeup_enabled else 'OFF'}")

            # Intensity controls
            elif key == ord('+') or key == ord('='):
                intensity = self.beauty_filter.adjust_intensity(self.config.INTENSITY_STEP)
                self.logger.info(f"Makeup intensity: {intensity:.1f}")

            elif key == ord('-'):
                intensity = self.beauty_filter.adjust_intensity(-self.config.INTENSITY_STEP)
                self.logger.info(f"Makeup intensity: {intensity:.1f}")

            # Other controls
            elif key == ord('t'):
                self._show_skin_tone_info()

            elif key == ord('r'):
                self.skin_detector.reset_calibration()
                self.skin_detector.start_calibration()
                self.logger.info("Skin tone calibration reset and restarted")

            elif key == ord('c'):
                self._show_configuration_menu()

            return True

        except Exception as e:
            self.logger.error(f"Error handling keyboard input: {e}")
            return True

    def _save_current_frame(self):
        """Save current frame with error handling"""
        try:
            if hasattr(self, '_current_frame') and self._current_frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = Path(self.config.OUTPUT_DIR) / f"beauty_frame_{timestamp}.png"

                success = cv2.imwrite(str(filename), self._current_frame)
                if success:
                    self.logger.info(f"Frame saved: {filename}")
                else:
                    self.logger.error("Failed to save frame")
            else:
                self.logger.warning("No frame available to save")

        except Exception as e:
            self.logger.error(f"Error saving frame: {e}")

    def _handle_dress_recommendation(self):
        """Handle dress recommendation request"""
        try:
            if self.skin_detector.calibrated_skin_tone is not None:
                season, dress_count = self.dress_manager.get_season_info()

                # Get comprehensive recommendations
                recommended_colors, undertone, depth, description = self.dress_recommender.recommend_dress_colors(
                    self.skin_detector.calibrated_skin_tone, season)

                makeup_recommendations = self.dress_recommender.recommend_makeup_colors(
                    self.skin_detector.calibrated_skin_tone)

                self.logger.info("\n🎨 === PERSONALIZED STYLE ANALYSIS ===")
                self.logger.info(f"👤 Your skin: {description}")
                self.logger.info(f"🌸 Season: {season.title()}")
                self.logger.info(f"✨ Recommended dress colors: {', '.join(recommended_colors)}")

                # Find and switch to best dress match
                best_match = self.dress_manager.find_best_dress(recommended_colors)
                if best_match:
                    best_index, dress_name, score = best_match
                    self.dress_manager.current_dress_idx = best_index
                    self.logger.info(f"✅ Switched to best match: {dress_name} (Score: {score})")

                # Apply recommended makeup
                self.beauty_filter.lipstick_color_idx = makeup_recommendations["lipstick"][0]
                self.beauty_filter.blush_color_idx = makeup_recommendations["blush"][0]
                self.beauty_filter.eyeshadow_color_idx = makeup_recommendations["eyeshadow"][0]

                # Enable makeup if not already on
                if not self.beauty_filter.makeup_enabled:
                    self.beauty_filter.makeup_enabled = True
                    self.beauty_filter.lipstick_enabled = True
                    self.beauty_filter.blush_enabled = True
                    self.beauty_filter.eyeshadow_enabled = True
                    self.logger.info("✨ Applied recommended makeup automatically!")

            else:
                self.logger.info("Please complete skin tone calibration first (look at camera for a few seconds)")

        except Exception as e:
            self.logger.error(f"Error handling dress recommendation: {e}")

    def _check_if_recommended(self, makeup_type: str, color_idx: int) -> bool:
        """Check if color is recommended for user's skin tone"""
        try:
            if self.skin_detector.calibrated_skin_tone is None:
                return False

            recommendations = self.dress_recommender.recommend_makeup_colors(
                self.skin_detector.calibrated_skin_tone)

            return color_idx in recommendations.get(makeup_type, [])

        except Exception as e:
            self.logger.error(f"Error checking recommendation: {e}")
            return False

    def _apply_best_makeup(self):
        """Apply best recommended makeup look"""
        try:
            if self.skin_detector.calibrated_skin_tone is not None:
                makeup_recs = self.dress_recommender.recommend_makeup_colors(
                    self.skin_detector.calibrated_skin_tone)

                self.beauty_filter.lipstick_color_idx = makeup_recs["lipstick"][0]
                self.beauty_filter.blush_color_idx = makeup_recs["blush"][0]
                self.beauty_filter.eyeshadow_color_idx = makeup_recs["eyeshadow"][0]

                self.beauty_filter.makeup_enabled = True
                self.beauty_filter.lipstick_enabled = True
                self.beauty_filter.blush_enabled = True
                self.beauty_filter.eyeshadow_enabled = True

                self.logger.info("✨ Applied your most recommended makeup look!")
            else:
                self.logger.info("Complete skin tone calibration first for recommendations")

        except Exception as e:
            self.logger.error(f"Error applying best makeup: {e}")

    def _cycle_recommended_colors(self):
        """Cycle through recommended colors only"""
        try:
            if self.skin_detector.calibrated_skin_tone is not None:
                makeup_recs = self.dress_recommender.recommend_makeup_colors(
                    self.skin_detector.calibrated_skin_tone)

                # Cycle through recommended lipstick colors only
                current_rec_idx = 0
                try:
                    current_rec_idx = makeup_recs["lipstick"].index(self.beauty_filter.lipstick_color_idx)
                    next_rec_idx = (current_rec_idx + 1) % len(makeup_recs["lipstick"])
                except ValueError:
                    next_rec_idx = 0

                self.beauty_filter.lipstick_color_idx = makeup_recs["lipstick"][next_rec_idx]
                color_name = ColorPalette.LIPSTICK_NAMES[self.beauty_filter.lipstick_color_idx]
                self.logger.info(f"Recommended Lipstick: {color_name} ✨")
            else:
                self.logger.info("Complete skin tone calibration first")

        except Exception as e:
            self.logger.error(f"Error cycling recommended colors: {e}")

    def _show_skin_tone_info(self):
        """Show skin tone information"""
        try:
            if self.skin_detector.calibrated_skin_tone is not None:
                undertone, depth, description = self.dress_recommender.get_skin_tone_analysis(
                    self.skin_detector.calibrated_skin_tone)
                self.logger.info(f"Your skin tone: {description}")
            else:
                self.logger.info("Skin tone not calibrated yet. Look at camera for calibration.")

        except Exception as e:
            self.logger.error(f"Error showing skin tone info: {e}")

    def _show_configuration_menu(self):
        """Show configuration information"""
        try:
            self.logger.info("\n⚙️ === CONFIGURATION ===")
            self.logger.info(f"Camera Resolution: {self.config.FRAME_WIDTH}x{self.config.FRAME_HEIGHT}")
            self.logger.info(f"Makeup Intensity: {self.beauty_filter.makeup_intensity:.1f}")
            self.logger.info(f"Calibration Frames: {self.config.CALIBRATION_FRAMES}")

            # Get camera info
            camera_info = self.camera_manager.get_frame_info()
            if camera_info:
                self.logger.info(f"Actual Camera: {camera_info.get('width', 'Unknown')}x{camera_info.get('height', 'Unknown')}")

        except Exception as e:
            self.logger.error(f"Error showing configuration: {e}")

    def run(self):
        """Main application loop with comprehensive error handling"""
        try:
            if not self.initialize():
                self.logger.critical("Failed to initialize application")
                return False

            self.running = True
            self.logger.info("Smart Beauty Filter started successfully")
            self._print_startup_info()

            while self.running:
                try:
                    # Read frame
                    ret, frame = self.camera_manager.read_frame()
                    if not ret or frame is None:
                        self.logger.warning("Failed to read frame, continuing...")
                        continue

                    self.frame_count += 1
                    self._current_frame = frame.copy()

                    # Process frame
                    processed_frame = self.process_frame(frame)

                    # Draw skin color display
                    self.skin_detector.draw_skin_color_display(processed_frame,
                                                             processed_frame.shape[1] - 80, 20, 50)

                    # Draw UI
                    if not self.show_help:
                        season, dress_count = self.dress_manager.get_season_info()
                        current_dress = self.dress_manager.get_current_dress()
                        dress_name = current_dress[0] if current_dress else "None"

                        makeup_status = {
                            'enabled': self.beauty_filter.makeup_enabled,
                            'lipstick': self.beauty_filter.lipstick_enabled,
                            'blush': self.beauty_filter.blush_enabled,
                            'eyeshadow': self.beauty_filter.eyeshadow_enabled,
                            'lipstick_idx': self.beauty_filter.lipstick_color_idx,
                            'blush_idx': self.beauty_filter.blush_color_idx,
                            'eyeshadow_idx': self.beauty_filter.eyeshadow_color_idx,
                            'intensity': self.beauty_filter.makeup_intensity
                        }

                        calibration_progress = self.skin_detector.get_calibration_progress()

                        processed_frame = self.ui_manager.draw_status_panel(
                            processed_frame, season, dress_name, calibration_progress, makeup_status)
                    else:
                        processed_frame = self.ui_manager.show_help_menu(processed_frame)

                    # Show frame
                    cv2.imshow("Smart Beauty & Dress Filter", processed_frame)

                    # Handle keyboard input
                    key = cv2.waitKey(1)
                    if not self.handle_keyboard_input(key):
                        break

                except KeyboardInterrupt:
                    self.logger.info("Keyboard interrupt received")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    continue

        except Exception as e:
            self.logger.critical(f"Critical error in main application: {e}")
            return False
        finally:
            self.cleanup()

        return True

    def _print_startup_info(self):
        """Print startup information"""
        try:
            season, dress_count = self.dress_manager.get_season_info()

            print("\n" + "="*60)
            print("🎨 SMART BEAUTY & DRESS FILTER")
            print("="*60)
            print(f"Current Season: {season.title()}")
            print(f"Available Dresses: {dress_count}")
            print(f"Makeup Colors: {len(ColorPalette.LIPSTICK_COLORS)} lipstick, {len(ColorPalette.BLUSH_COLORS)} blush, {len(ColorPalette.EYESHADOW_COLORS)} eyeshadow")
            print("\nStarting skin tone calibration...")
            print("💡 Look directly at the camera for 5 seconds for accurate results!")
            print("\nPress 'h' for help menu")
            print("="*60)

        except Exception as e:
            self.logger.error(f"Error printing startup info: {e}")

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.logger.info("Cleaning up resources...")

            # Release camera
            if hasattr(self, 'camera_manager'):
                self.camera_manager.release()

            # Release MediaPipe resources
            if hasattr(self, 'pose'):
                self.pose.close()
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()

            # Close OpenCV windows
            cv2.destroyAllWindows()

            # Save configuration
            self.config.save("beauty_filter_config.json")

            # Print final summary
            if hasattr(self, 'skin_detector') and self.skin_detector.calibrated_skin_tone is not None:
                self._print_final_summary()

            self.logger.info("Cleanup complete")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def _print_final_summary(self):
        """Print final style summary"""
        try:
            print("\n" + "🎉" + "="*50 + "🎉")
            print("YOUR STYLE PROFILE SUMMARY")
            print("="*52)

            undertone, depth, description = self.dress_recommender.get_skin_tone_analysis(
                self.skin_detector.calibrated_skin_tone)
            season, _ = self.dress_manager.get_season_info()
            recommended_colors, _, _, _ = self.dress_recommender.recommend_dress_colors(
                self.skin_detector.calibrated_skin_tone, season)

            print(f"👤 Skin Analysis: {description}")
            print(f"🌸 Best Season: {season.title()}")
            print(f"✨ Your Colors: {', '.join(recommended_colors[:5])}")  # Show first 5
            print(f"💄 Last Makeup Setup:")
            print(f"   Lipstick: {ColorPalette.LIPSTICK_NAMES[self.beauty_filter.lipstick_color_idx]}")
            print(f"   Blush: {ColorPalette.BLUSH_NAMES[self.beauty_filter.blush_color_idx]}")
            print(f"   Eyeshadow: {ColorPalette.EYESHADOW_NAMES[self.beauty_filter.eyeshadow_color_idx]}")
            print(f"   Intensity: {self.beauty_filter.makeup_intensity:.1f}")

            # Runtime stats
            runtime = time.time() - self.start_time
            fps = self.frame_count / runtime if runtime > 0 else 0
            print(f"\n📊 Session Stats:")
            print(f"   Runtime: {runtime:.1f} seconds")
            print(f"   Frames Processed: {self.frame_count}")
            print(f"   Average FPS: {fps:.1f}")

            print("\n💡 Remember these recommendations for your future outfit choices!")
            print("="*52)
            print("✨ Thanks for using Smart Beauty & Dress Filter! Stay stylish! ✨")

        except Exception as e:
            self.logger.error(f"Error printing final summary: {e}")

# ------------------------- Main Entry Point -------------------------
def main():
    """Main entry point with comprehensive error handling"""
    try:
        app = SmartBeautyFilter()
        success = app.run()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n[INFO] Application interrupted by user")
        sys.exit(0)
    except CameraError as e:
        print(f"[ERROR] Camera error: {e}")
        print("[INFO] Please check camera connection and permissions")
        sys.exit(1)
    except ImportError as e:
        print(f"[ERROR] Missing required dependency: {e}")
        print("[INFO] Please install required packages: pip install opencv-python mediapipe numpy scikit-learn")
        sys.exit(1)
    except Exception as e:
        print(f"[CRITICAL] Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()