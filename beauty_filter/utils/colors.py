

class ColorPalette:
    """Enhanced color palette with validation"""

    # Enhanced color palettes (BGR format)
    LIPSTICK_COLORS = [
        (45, 45, 200),  # Classic Red
        (80, 60, 180),  # Berry Rose
        (30, 80, 220),  # Bright Red
        (20, 30, 140),  # Deep Red
        (120, 100, 200),  # Pink Red
        (40, 20, 100),  # Wine Berry
        (90, 70, 160),  # Mauve
        (100, 80, 190),  # Coral Pink
        (60, 40, 160),  # Berry
        (110, 90, 210),  # Rose Gold
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
        (90, 120, 180),  # Natural Rose
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
        (120, 100, 80),  # Warm Brown
        (160, 130, 100),  # Bronze
        (140, 100, 120),  # Plum
        (100, 130, 80),  # Olive Green
        (180, 170, 100),  # Gold
        (100, 80, 60),  # Deep Brown
        (120, 120, 140),  # Taupe
        (100, 100, 100),  # Smokey Grey
        (130, 110, 90),  # Copper
        (90, 80, 120),  # Chocolate
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
