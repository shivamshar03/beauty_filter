class ImageProcessor:
    """Improved image processing with proper dress positioning"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def overlay_rgba_onto_bgr(self, frame_bgr: np.ndarray, overlay_rgba: np.ndarray,
                              center_xy: Tuple[float, float]) -> np.ndarray:
        """
        Alpha blend overlay_rgba (H,W,4) onto frame_bgr (H,W,3) centered at center_xy=(cx,cy).
        Allows dress to extend beyond screen bounds.
        """
        try:
            h, w = overlay_rgba.shape[:2]
            cx, cy = int(center_xy[0]), int(center_xy[1])

            # Calculate overlay bounds
            x1 = cx - w // 2
            y1 = cy - h // 2
            x2 = x1 + w
            y2 = y1 + h

            fh, fw = frame_bgr.shape[:2]

            # Clip to frame bounds while allowing overflow
            x1_clip = max(0, x1)
            y1_clip = max(0, y1)
            x2_clip = min(fw, x2)
            y2_clip = min(fh, y2)

            # If completely outside visible area, return original frame
            if x1_clip >= x2_clip or y1_clip >= y2_clip:
                return frame_bgr

            # Calculate corresponding region in overlay
            ox1 = max(0, x1_clip - x1)
            oy1 = max(0, y1_clip - y1)
            ox2 = ox1 + (x2_clip - x1_clip)
            oy2 = oy1 + (y2_clip - y1_clip)

            # Ensure we don't exceed overlay bounds
            ox2 = min(w, ox2)
            oy2 = min(h, oy2)

            # Extract regions
            roi = frame_bgr[y1_clip:y2_clip, x1_clip:x2_clip]
            overlay_crop = overlay_rgba[oy1:oy2, ox1:ox2]

            # Verify dimensions match
            if roi.shape[:2] != overlay_crop.shape[:2]:
                return frame_bgr

            # Alpha blending
            overlay_rgb = overlay_crop[..., :3].astype(np.float32)
            alpha = (overlay_crop[..., 3:4].astype(np.float32)) / 255.0

            roi_float = roi.astype(np.float32)
            blended = alpha * overlay_rgb + (1.0 - alpha) * roi_float

            # Apply blended result back to frame
            frame_bgr[y1_clip:y2_clip, x1_clip:x2_clip] = blended.astype(np.uint8)

            return frame_bgr

        except Exception as e:
            self.logger.error(f"Overlay error: {e}")
            return frame_bgr

    def scale_and_rotate_rgba(self, img_rgba: np.ndarray, target_w: int, target_h: int, angle_deg: float) -> Optional[
        np.ndarray]:
        """Scale and rotate dress with horizontal flip correction"""
        try:
            target_w = max(1, int(target_w))
            target_h = max(1, int(target_h))

            resized = cv2.resize(img_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)

            # Flip horizontally to correct inversion
            resized = cv2.flip(resized, 1)

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
            self.logger.error(f"Scale and rotate error: {e}")
            return None

    def calculate_pose_metrics(self, pose_landmarks, frame_shape: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """Calculate pose metrics with fallback for upper-body only"""
        try:
            if not pose_landmarks or not hasattr(pose_landmarks, 'landmark'):
                return None

            h, w = frame_shape[:2]
            landmarks = pose_landmarks.landmark

            def get_xy(i):
                if i < len(landmarks):
                    lm = landmarks[i]
                    return (int(lm.x * w), int(lm.y * h))
                return None

            # Get shoulders
            R_SHO = get_xy(11)  # Right shoulder
            L_SHO = get_xy(12)  # Left shoulder

            if None in [R_SHO, L_SHO]:
                return None

            SHO_MID = ((L_SHO[0] + R_SHO[0]) // 2, (L_SHO[1] + R_SHO[1]) // 2)

            # Try to get hips, fallback if missing
            R_HIP = get_xy(23)
            L_HIP = get_xy(24)

            if None not in [R_HIP, L_HIP]:
                HIP_MID = ((L_HIP[0] + R_HIP[0]) // 2, (L_HIP[1] + R_HIP[1]) // 2)
                torso_height = self._distance(SHO_MID, HIP_MID)
            else:
                # Fallback: Estimate based on shoulder width
                shoulder_width = self._distance(L_SHO, R_SHO)
                HIP_MID = (SHO_MID[0], SHO_MID[1] + int(1.3 * shoulder_width))
                torso_height = self._distance(SHO_MID, HIP_MID)

            # Calculate metrics
            TORSO_MID = ((SHO_MID[0] + HIP_MID[0]) // 2, (SHO_MID[1] + HIP_MID[1]) // 2)
            shoulder_width = self._distance(L_SHO, R_SHO)

            # Calculate shoulder angle
            angle = degrees(atan2(R_SHO[1] - L_SHO[1], R_SHO[0] - L_SHO[0]))
            if angle < -90:
                angle += 180
            elif angle > 90:
                angle -= 180

            return {
                "shoulder_mid": SHO_MID,
                "hip_mid": HIP_MID,
                "torso_mid": TORSO_MID,
                "shoulder_width": shoulder_width,
                "torso_height": torso_height,
                "angle": angle,
                "left_shoulder": L_SHO,
                "right_shoulder": R_SHO
            }

        except Exception as e:
            self.logger.error(f"Error calculating pose metrics: {e}")
            return None

    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        try:
            return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        except Exception:
            return 0.0

    def apply_dress_overlay(self, frame, pose_landmarks, dress_rgba, config):
        """
        ✅ FIXED: Apply dress overlay with proper scaling controls
        - WIDTH_FACTOR: Controls dress width
        - HEIGHT_FACTOR: Controls dress height
        - VERTICAL_OFFSET_FACTOR: Controls vertical position (-1.0 up, 0.0 center, 1.0 down)
        """
        try:
            if not pose_landmarks or not hasattr(pose_landmarks, 'landmark'):
                return frame

            h, w = frame.shape[:2]
            landmarks = pose_landmarks.landmark

            def get_xy(i):
                if i < len(landmarks):
                    lm = landmarks[i]
                    return (int(lm.x * w), int(lm.y * h))
                return None

            # Get shoulder positions
            R_SHO = get_xy(11)  # Right shoulder
            L_SHO = get_xy(12)  # Left shoulder

            if None in [R_SHO, L_SHO]:
                return frame

            # Calculate shoulder metrics
            SHO_MID = ((L_SHO[0] + R_SHO[0]) // 2, (L_SHO[1] + R_SHO[1]) // 2)
            shoulder_width = sqrt((L_SHO[0] - R_SHO[0]) ** 2 + (L_SHO[1] - R_SHO[1]) ** 2)

            # ✅ FIXED: Apply scaling factors properly
            target_w = int(config.WIDTH_FACTOR * shoulder_width)
            target_h = int(config.HEIGHT_FACTOR * shoulder_width)  # Base height on shoulder width for consistency

            # Calculate rotation angle
            angle = degrees(atan2(R_SHO[1] - L_SHO[1], R_SHO[0] - L_SHO[0]))
            if angle < -90:
                angle += 180
            elif angle > 90:
                angle -= 180

            # Scale and rotate dress
            overlay = self.scale_and_rotate_rgba(dress_rgba, target_w, target_h, angle)
            if overlay is None:
                return frame

            oh, ow = overlay.shape[:2]

            # ✅ FIXED: Apply vertical offset properly
            # Default: center dress at shoulders
            base_y = SHO_MID[1]

            # Apply vertical offset as fraction of dress height
            vertical_offset = int(config.VERTICAL_OFFSET_FACTOR * oh)

            dress_center = (
                SHO_MID[0],  # Horizontal center at shoulders
                base_y + vertical_offset  # Apply vertical offset
            )

            # Apply overlay
            frame = self.overlay_rgba_onto_bgr(frame, overlay, dress_center)

            # ✅ DEBUG: Show scaling parameters on screen
            debug_text = f"W:{config.WIDTH_FACTOR:.1f} H:{config.HEIGHT_FACTOR:.1f} V:{config.VERTICAL_OFFSET_FACTOR:.1f}"
            cv2.putText(frame, debug_text, (w - 300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            return frame

        except Exception as e:
            self.logger.error(f"Dress overlay error: {e}")
            return frame
