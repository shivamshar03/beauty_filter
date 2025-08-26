"""
Dress Overlay Filter (no UI)
- Uses MediaPipe Pose to place/scale/rotate a PNG dress (with transparent BG) on the body.
- Controls:
    [  : previous dress
    ]  : next dress
    s  : save current frame to ./output
    q  : quit
Requirements:
    pip install opencv-python mediapipe numpy
Prepare:
    Put your PNGs (with alpha) inside ./assets/dresses
"""

import cv2
import numpy as np
import os
from pathlib import Path
import time
import mediapipe as mp
from math import atan2, degrees, sqrt

# ------------------------- Config -------------------------
ASSETS_DIR = Path("../assets/dresses")
OUTPUT_DIR = Path("../output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Scale factors you can tweak for fit
WIDTH_FACTOR  = 2.1
HEIGHT_FACTOR = 2.2
VERTICAL_OFFSET_FACTOR = 0.35

# Video
CAM_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ------------------------- Utilities -------------------------
def load_dresses(folder: Path):
    exts = {".png"}
    files = sorted([p for p in folder.glob("*") if p.suffix.lower() in exts])
    imgs = []
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)  # BGRA
        if img is None or img.shape[2] != 4:
            print(f"[WARN] Skipping {p.name}: not a valid RGBA PNG.")
            continue
        imgs.append((p.name, img))
    return imgs

def overlay_rgba_onto_bgr(frame_bgr, overlay_rgba, center_xy):
    """
    Alpha blend overlay_rgba (H,W,4) onto frame_bgr (H,W,3) centered at center_xy=(cx,cy).
    Handles clipping when overlay goes out of frame.
    """
    h, w = overlay_rgba.shape[:2]
    cx, cy = int(center_xy[0]), int(center_xy[1])

    x1 = cx - w // 2
    y1 = cy - h // 2
    x2 = x1 + w
    y2 = y1 + h

    fh, fw = frame_bgr.shape[:2]
    # Clip ROI to frame bounds
    x1_clip, y1_clip = max(0, x1), max(0, y1)
    x2_clip, y2_clip = min(fw, x2), min(fh, y2)

    if x1_clip >= x2_clip or y1_clip >= y2_clip:
        return frame_bgr  # nothing to draw

    # Corresponding region in overlay
    ox1, oy1 = x1_clip - x1, y1_clip - y1
    ox2, oy2 = ox1 + (x2_clip - x1_clip), oy1 + (y2_clip - y1_clip)

    roi = frame_bgr[y1_clip:y2_clip, x1_clip:x2_clip]
    overlay_crop = overlay_rgba[oy1:oy2, ox1:ox2]

    overlay_rgb = overlay_crop[..., :3].astype(np.float32)
    alpha = (overlay_crop[..., 3:4].astype(np.float32)) / 255.0  # (H,W,1)

    # Blend
    roi_float = roi.astype(np.float32)
    blended = alpha * overlay_rgb + (1.0 - alpha) * roi_float
    frame_bgr[y1_clip:y2_clip, x1_clip:x2_clip] = blended.astype(np.uint8)
    return frame_bgr

def scale_and_rotate_rgba(img_rgba, target_w, target_h, angle_deg):
    """
    Resize to (target_w, target_h) then rotate around center keeping alpha.
    """
    target_w = max(1, int(target_w))
    target_h = max(1, int(target_h))
    resized = cv2.resize(img_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # ✅ Flip horizontally to correct inversion
    resized = cv2.flip(resized, 1)

    (h, w) = resized.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        resized, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)  # keep transparency
    )
    return rotated

def dist(p1, p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# ------------------------- Main -------------------------
def main():
    dresses = load_dresses(ASSETS_DIR)
    if not dresses:
        print(f"[ERROR] No valid RGBA PNGs found in {ASSETS_DIR}.")
        return

    idx = 0
    print(f"[INFO] Loaded {len(dresses)} dresses. Current: {dresses[idx][0]}")
    print("Controls: [ prev | ] next | s save | q quit")

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Camera read failed.")
                break

            frame = cv2.flip(frame, 1)  # mirror
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark

                # Pixel coordinates of key landmarks
                def get_xy(i):
                    return (int(lm[i].x * w), int(lm[i].y * h))


                R_SHO = get_xy(11)
                L_SHO = get_xy(12)
                R_HIP = get_xy(23)
                L_HIP = get_xy(24)

                # Midpoints
                SHO_MID = ((L_SHO[0] + R_SHO[0]) // 2, (L_SHO[1] + R_SHO[1]) // 2)
                HIP_MID = ((L_HIP[0] + R_HIP[0]) // 2, (L_HIP[1] + R_HIP[1]) // 2)
                TORSO_MID = ((SHO_MID[0] + HIP_MID[0]) // 2, (SHO_MID[1] + HIP_MID[1]) // 2)

                shoulder_width = dist(L_SHO, R_SHO)
                torso_height = dist(SHO_MID, HIP_MID)

                # Compute angle of shoulders (rotation)

                # Compute raw angle
                angle = degrees(atan2(R_SHO[1] - L_SHO[1], R_SHO[0] - L_SHO[0]))

                # Normalize: keep dress upright
                if angle < -90:
                    angle += 180
                elif angle > 90:
                    angle -= 180


                # Target size for dress
                target_w = WIDTH_FACTOR * shoulder_width
                target_h = HEIGHT_FACTOR * torso_height

                # Vertical offset to align top of dress near shoulders
                offset_y = VERTICAL_OFFSET_FACTOR * torso_height
                dress_center = (TORSO_MID[0], int(TORSO_MID[1] + offset_y))

                # Prepare overlay
                name, dress_rgba = dresses[idx]
                overlay = scale_and_rotate_rgba(dress_rgba, target_w, target_h, angle)
                frame = overlay_rgba_onto_bgr(frame, overlay, dress_center)

                # Optional: small HUD
                cv2.putText(frame, f"Dress: {name}  |  [ / ] change, s save, q quit",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Pose not detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 255), 2, cv2.LINE_AA)

            cv2.imshow("Dress Filter (no UI)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = OUTPUT_DIR / f"frame_{int(time.time())}.png"
                cv2.imwrite(str(filename), frame)
                print(f"[INFO] Saved {filename}")
            elif key == ord(']'):
                idx = (idx + 1) % len(dresses)
                print(f"[INFO] Dress -> {dresses[idx][0]}")
            elif key == ord('['):
                idx = (idx - 1) % len(dresses)
                print(f"[INFO] Dress -> {dresses[idx][0]}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
