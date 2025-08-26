# ------------------------- Main Application Class -------------------------
class SmartBeautyFilter:
    """Main application class with improved dress overlay system"""

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
        """
        Modified process_frame with improved dress positioning
        """
        try:
            if frame is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run MediaPipe Pose + Face Mesh
            pose_results = self.pose.process(rgb)
            face_results = self.face_mesh.process(rgb)

            # --- Skin tone calibration ---
            if (self.skin_detector.is_calibrating and face_results and
                    hasattr(face_results, 'multi_face_landmarks') and face_results.multi_face_landmarks):
                try:
                    self.skin_detector.calibrate_skin_tone(
                        frame, face_results.multi_face_landmarks[0], (h, w)
                    )
                except Exception as e:
                    self.logger.warning(f"Skin tone calibration error: {e}")

            # --- Apply beauty filters ---
            if (self.beauty_filter.makeup_enabled and face_results and
                    hasattr(face_results, 'multi_face_landmarks') and face_results.multi_face_landmarks):
                try:
                    face_landmarks = face_results.multi_face_landmarks[0]

                    if self.beauty_filter.eyeshadow_enabled:
                        frame = self.beauty_filter.apply_eyeshadow(frame, face_landmarks)
                    if self.beauty_filter.blush_enabled:
                        frame = self.beauty_filter.apply_blush(frame, face_landmarks)
                    if self.beauty_filter.lipstick_enabled:
                        frame = self.beauty_filter.apply_lipstick(frame, face_landmarks)

                except Exception as e:
                    self.logger.warning(f"Beauty filter error: {e}")

            # --- Dress overlay with shoulder anchoring ---
            if pose_results and pose_results.pose_landmarks:
                try:
                    current_dress = self.dress_manager.get_current_dress()
                    if current_dress:
                        name, dress_rgba = current_dress

                        # ✅ Use improved dress overlay that anchors at shoulders
                        frame = self.image_processor.apply_dress_overlay(
                            frame, pose_results.pose_landmarks, dress_rgba, self.config
                        )

                        # Show dress name
                        cv2.putText(frame, f"Dress: {name}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                except Exception as e:
                    self.logger.warning(f"Dress overlay error: {e}")
            else:
                cv2.putText(frame, "Pose not detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 255), 2, cv2.LINE_AA)

            return frame

        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return frame

    # ------------------------- Enhanced Key Handling for Real-time Scaling -------------------------
    def handle_keyboard_input(self, key: int) -> bool:
        """Enhanced keyboard input with real-time scaling controls"""
        try:
            if key == -1:  # No key pressed
                return True

            key = key & 0xFF

            # Existing controls (quit, help, etc.)
            if key == ord('q'):
                self.logger.info("User requested quit")
                return False
            elif key == ord('h'):
                self.show_help = not self.show_help
                self.logger.info(f"Help menu {'shown' if self.show_help else 'hidden'}")

            # ✅ NEW: Real-time dress scaling controls
            elif key == ord('w'):  # Increase width
                self.config.WIDTH_FACTOR = min(10.0, self.config.WIDTH_FACTOR + 0.01)
                self.logger.info(f"Width Factor: {self.config.WIDTH_FACTOR:.1f}")

            elif key == ord('W'):  # Decrease width (Shift+W)
                self.config.WIDTH_FACTOR = max(0.5, self.config.WIDTH_FACTOR - 0.01)
                self.logger.info(f"Width Factor: {self.config.WIDTH_FACTOR:.1f}")

            elif key == ord('y'):  # Increase height
                self.config.HEIGHT_FACTOR = min(10.0, self.config.HEIGHT_FACTOR + 0.01)
                self.logger.info(f"Height Factor: {self.config.HEIGHT_FACTOR:.1f}")

            elif key == ord('Y'):  # Decrease height (Shift+Y)
                self.config.HEIGHT_FACTOR = max(0.5, self.config.HEIGHT_FACTOR - 0.01)
                self.logger.info(f"Height Factor: {self.config.HEIGHT_FACTOR:.1f}")

            elif key == ord('u'):  # Move dress up
                self.config.VERTICAL_OFFSET_FACTOR = max(-2.0, self.config.VERTICAL_OFFSET_FACTOR - 0.01)
                self.logger.info(f"Vertical Offset: {self.config.VERTICAL_OFFSET_FACTOR:.1f}")

            elif key == ord('U'):  # Move dress down (Shift+U)
                self.config.VERTICAL_OFFSET_FACTOR = min(2.0, self.config.VERTICAL_OFFSET_FACTOR + 0.01)
                self.logger.info(f"Vertical Offset: {self.config.VERTICAL_OFFSET_FACTOR:.1f}")

            elif key == ord('0'):  # Reset to default scaling
                self.config.WIDTH_FACTOR = 2.0
                self.config.HEIGHT_FACTOR = 1.5
                self.config.VERTICAL_OFFSET_FACTOR = 0.0
                self.logger.info("Dress scaling reset to defaults")

            # Save configuration when scaling changes
            elif key in [ord('w'), ord('W'), ord('y'), ord('Y'), ord('u'), ord('U'), ord('0')]:
                self.config.save("beauty_filter_config.json")

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
            self.logger.info(f"Dress Scale Factors: W={self.config.WIDTH_FACTOR}, H={self.config.HEIGHT_FACTOR}")

            # Get camera info
            camera_info = self.camera_manager.get_frame_info()
            if camera_info:
                self.logger.info(
                    f"Actual Camera: {camera_info.get('width', 'Unknown')}x{camera_info.get('height', 'Unknown')}")

        except Exception as e:
            self.logger.error(f"Error showing configuration: {e}")

    def run(self):
        """Main application loop with improved dress overlay system"""
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

                    processed_frame = self.process_frame(frame)

                    # Draw skin color display
                    self.skin_detector.draw_skin_color_display(
                        processed_frame,
                        processed_frame.shape[1] - 80, 20, 50
                    )

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
                            processed_frame, season, dress_name,
                            calibration_progress, makeup_status
                        )
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

            print("\n" + "=" * 60)
            print("🎨 SMART BEAUTY & DRESS FILTER (IMPROVED)")
            print("=" * 60)
            print(f"Current Season: {season.title()}")
            print(f"Available Dresses: {dress_count}")
            print(
                f"Makeup Colors: {len(ColorPalette.LIPSTICK_COLORS)} lipstick, {len(ColorPalette.BLUSH_COLORS)} blush, {len(ColorPalette.EYESHADOW_COLORS)} eyeshadow")
            print(f"Dress Overlay: Improved system with better fitting & rotation")
            print("\nStarting skin tone calibration...")
            print("💡 Look directly at the camera for 5 seconds for accurate results!")
            print("\nPress 'h' for help menu")
            print("=" * 60)

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
            print("\n" + "🎉" + "=" * 50 + "🎉")
            print("YOUR STYLE PROFILE SUMMARY")
            print("=" * 52)

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
            print("=" * 52)
            print("✨ Thanks for using Smart Beauty & Dress Filter! Stay stylish! ✨")

        except Exception as e:
            self.logger.error(f"Error printing final summary: {e}")
