


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

    def recommend_dress_colors(self, skin_tone_bgr: Optional[np.ndarray], season: Optional[str] = None) -> Tuple[
        List[str], str, str, str]:
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

    def analyze_dress_suitability(self, dress_name: str, skin_tone_bgr: Optional[np.ndarray],
                                  season: Optional[str] = None) -> Tuple[int, str, List[str]]:
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
