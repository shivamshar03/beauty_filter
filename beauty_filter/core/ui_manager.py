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

    # ------------------------- Updated Help Menu -------------------------
    def show_help_menu(self, frame: np.ndarray) -> np.ndarray:
        """Enhanced help menu with scaling controls"""
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

            # Enhanced help content with scaling controls
            help_lines = [
                "SMART BEAUTY FILTER - HELP (Enhanced)",
                "=" * 40,
                "",
                "DRESS SCALING (Real-time):",
                "  w/W     - Increase/Decrease dress width",
                "  y/Y     - Increase/Decrease dress height",
                "  u/U     - Move dress up/down (vertical offset)",
                "  0       - Reset scaling to defaults",
                "",
                "DRESS CONTROLS:",
                "  [ ]     - Previous/Next dress",
                "  d       - Get personalized recommendations",
                "  o       - Change season",
                "",
                "MAKEUP CONTROLS:",
                "  1/2/3   - Cycle lipstick/blush/eyeshadow colors",
                "  l/b/e   - Toggle lipstick/blush/eyeshadow",
                "  4       - Apply best recommended look",
                "  m       - Toggle all makeup",
                "  +/-     - Adjust intensity",
                "",
                "SYSTEM CONTROLS:",
                "  s       - Save frame",
                "  r       - Reset skin calibration",
                "  h       - Toggle this help",
                "  q       - Quit",
                "",
                "CURRENT SCALING:",
                f"  Width Factor: {self.config.WIDTH_FACTOR:.1f}",
                f"  Height Factor: {self.config.HEIGHT_FACTOR:.1f}",
                f"  Vertical Offset: {self.config.VERTICAL_OFFSET_FACTOR:.1f}",
                "",
                "Press 'h' again to close help"
            ]

            text_y = panel_y + 30
            for line in help_lines:
                if "SMART BEAUTY FILTER" in line:
                    color = (0, 255, 0)
                    font_scale = 0.8
                elif line.startswith("="):
                    color = (0, 255, 0)
                    font_scale = 0.6
                elif line.endswith(":"):
                    color = (0, 255, 255)
                    font_scale = 0.7
                elif "CURRENT SCALING:" in line:
                    color = (255, 255, 0)
                    font_scale = 0.7
                else:
                    color = (255, 255, 255)
                    font_scale = 0.6

                cv2.putText(frame, line, (panel_x + 20, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
                text_y += int(25 * (1.2 if font_scale > 0.6 else 1.0))

                if text_y > panel_y + panel_h - 30:
                    break

            return frame

        except Exception as e:
            self.logger.error(f"Error showing help menu: {e}")
            return frame
