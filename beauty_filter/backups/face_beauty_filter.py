"""
Streamlit Snapchat‑style PNG dress filter (accurate, landmark‑anchored)

Run:
  pip install streamlit streamlit-webrtc opencv-python mediapipe numpy
  streamlit run app.py

Notes:
- Upload a transparent PNG of a dress/top. Use the 3 anchor sliders to map where
  the garment's left shoulder, right shoulder, and waist/mid‑hip points are in the PNG.
- For accuracy, we use MediaPipe Pose (upper body) + EMA smoothing and an affine
  transform (3-point) to warp the PNG onto the live frame. Optional fine‑tune
  offsets and scale compensate for garment variations.
- Works with webcam via streamlit-webrtc. If your environment blocks camera access,
  use the "Process still photo" tab.
"""

import math
import time
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode


# ---------------------------- UI CONFIG ----------------------------
st.set_page_config(page_title="PNG Dress Filter", page_icon="👗", layout="wide")

st.markdown(
    """
    <style>
    .small-note { opacity: 0.75; font-size: 0.9rem; }
    .tip { background: #f6f7ff; border: 1px solid #e5e7ff; padding: .75rem 1rem; border-radius: .75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("👗 Snapchat‑style Dress Filter (PNG overlay)")
st.caption("Accurate affine warp using shoulders + waist landmarks, with smoothing.")

# ---------------------------- HELPERS ----------------------------
POSE_IDXS = {
    "l_shoulder": 11,
    "r_shoulder": 12,
    "l_hip": 23,
    "r_hip": 24,
}

mp_pose = mp.solutions.pose


def get_pose_landmarks_rgb(rgb: np.ndarray) -> Optional[dict]:
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(rgb)
        if not results.pose_landmarks:
            return None
        h, w = rgb.shape[:2]
        lm = results.pose_landmarks.landmark
        out = {}
        for name, idx in POSE_IDXS.items():
            out[name] = (int(lm[idx].x * w), int(lm[idx].y * h), lm[idx].visibility)
        return out


@dataclass
class EMA:
    alpha: float
    value: Optional[np.ndarray] = None

    def update(self, x: np.ndarray) -> np.ndarray:
        if self.value is None:
            self.value = x.astype(np.float32)
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


# Alpha blend PNG with BGRA support

def alpha_blend(bg_bgr: np.ndarray, fg_bgra: np.ndarray) -> np.ndarray:
    (h, w) = fg_bgra.shape[:2]
    if bg_bgr.shape[0] < h or bg_bgr.shape[1] < w:
        h = min(h, bg_bgr.shape[0])
        w = min(w, bg_bgr.shape[1])
        fg_bgra = fg_bgra[:h, :w]
    b, g, r, a = cv2.split(fg_bgra)
    alpha = a.astype(np.float32) / 255.0
    overlay = cv2.merge((b, g, r)).astype(np.float32)
    bg = bg_bgr[:h, :w].astype(np.float32)
    out = (alpha[..., None] * overlay + (1 - alpha[..., None]) * bg)
    bg_bgr[:h, :w] = np.clip(out, 0, 255).astype(np.uint8)
    return bg_bgr


# Compute affine transform from 3 src points -> 3 dst points

def warp_png_to_points(png_bgra: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray,
                       out_shape: Tuple[int, int]) -> np.ndarray:
    M = cv2.getAffineTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
    warped = cv2.warpAffine(png_bgra, M, (out_shape[1], out_shape[0]), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return warped


# ------------------------- SIDEBAR CONTROLS -------------------------
st.sidebar.header("Garment PNG & Anchors")

uploaded_png = st.sidebar.file_uploader(
    "Upload transparent PNG (dress/top)", type=["png"], help="Use a front‑facing garment with transparency."
)

st.sidebar.markdown("**Define garment anchors (percent of image size):**")
col1, col2 = st.sidebar.columns(2)
with col1:
    ls_x = st.slider("Left shoulder X%", 5, 95, 30)
    rs_x = st.slider("Right shoulder X%", 5, 95, 70)
with col2:
    shoulders_y = st.slider("Shoulders Y%", 0, 100, 13)
    waist_y = st.slider("Waist/Mid‑hip Y%", 0, 100, 55)

fine_scale = st.sidebar.slider("Fine scale", 0.5, 1.8, 1.0, 0.01)

st.sidebar.subheader("Pose offsets (pixels)")
dx = st.sidebar.slider("Horizontal offset", -150, 150, 0)
dy = st.sidebar.slider("Vertical offset", -200, 200, 20)

st.sidebar.markdown(
    """
    <div class="tip">
    <b>Tips for accuracy</b><br>
    • Align anchors to the garment's shoulder seam points and waistline.<br>
    • If dress is long, set waist near the natural waist so scaling stays realistic.<br>
    • Use fine scale for small global adjustments.
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- TABS -----------------------------
live_tab, photo_tab = st.tabs(["🎥 Live webcam", "🖼️ Process still photo"])


# Prepare PNG (lazy load each call)

def load_png() -> Optional[np.ndarray]:
    if uploaded_png is None:
        return None
    file_bytes = np.asarray(bytearray(uploaded_png.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] != 4:
        st.error("PNG must have an alpha channel (RGBA). Export with transparency.")
        return None
    return img


# Compute garment src triangle from sliders

def garment_src_triangle(png: np.ndarray) -> np.ndarray:
    h, w = png.shape[:2]
    p_ls = (w * ls_x / 100.0, h * shoulders_y / 100.0)
    p_rs = (w * rs_x / 100.0, h * shoulders_y / 100.0)
    p_waist = (w * (ls_x + rs_x) / 200.0, h * waist_y / 100.0)
    return np.array([p_ls, p_rs, p_waist], dtype=np.float32)


# --------------------------- LIVE PROCESSOR ---------------------------
class DressProcessor(VideoProcessorBase):
    def __init__(self, png_bgra: Optional[np.ndarray]):
        self.png = png_bgra
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                                 enable_segmentation=False, min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)
        self.ema = EMA(alpha=0.35)  # smoothing for jitter‑free overlay
        self.frame_ts = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if results.pose_landmarks is None or self.png is None:
            return frame

        h, w = img.shape[:2]
        lm = results.pose_landmarks.landmark
        def denorm(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)

        Ls = denorm(POSE_IDXS["l_shoulder"])  # (x,y)
        Rs = denorm(POSE_IDXS["r_shoulder"])
        hips_mid = 0.5 * (denorm(POSE_IDXS["l_hip"]) + denorm(POSE_IDXS["r_hip"]))

        dst = np.stack([Ls, Rs, hips_mid], axis=0)
        dst = self.ema.update(dst)

        # Apply user fine controls
        dst[:, 0] = dst[:, 0] + dx  # x
        dst[:, 1] = dst[:, 1] + dy  # y

        # Optional uniform scaling around shoulder midpoint
        shoulder_mid = 0.5 * (dst[0] + dst[1])
        vecs = dst - shoulder_mid
        dst = shoulder_mid + vecs * fine_scale

        # Build affine warp
        src_tri = garment_src_triangle(self.png)
        warped = warp_png_to_points(self.png, src_tri, dst, (h, w))

        out = alpha_blend(img, warped)
        return out


with live_tab:
    st.write("Enable camera to try the filter live.")
    png_live = load_png()

    rtc_config = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
    })
    webrtc_streamer(
        key="dress-filter",
        mode=WebRtcMode.SENDRECV,  # note: it's SENDRECV, not VIDEO
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: DressProcessor(png_live),
    )

# --------------------------- PHOTO PIPELINE ---------------------------
with photo_tab:
    st.write("Upload a photo, we will overlay the garment using the same anchors.")
    colp1, colp2 = st.columns(2)
    photo = colp1.file_uploader("Photo (JPG/PNG)", type=["jpg", "jpeg", "png"], key="photo")

    if photo is not None:
        np_bytes = np.asarray(bytearray(photo.read()), dtype=np.uint8)
        frame = cv2.imdecode(np_bytes, cv2.IMREAD_UNCHANGED)
        if frame is None:
            st.error("Could not read image.")
        else:
            png = load_png()
            if png is None:
                st.warning("Upload a garment PNG with alpha first.")
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                lms = get_pose_landmarks_rgb(rgb)
                if lms is None:
                    st.error("No person detected. Try a clearer, front‑facing photo.")
                else:
                    h, w = frame.shape[:2]
                    Ls = np.array(lms["l_shoulder"][:2], dtype=np.float32)
                    Rs = np.array(lms["r_shoulder"][:2], dtype=np.float32)
                    hips_mid = 0.5 * (np.array(lms["l_hip"][:2]) + np.array(lms["r_hip"][:2]))
                    dst = np.stack([Ls, Rs, hips_mid], axis=0).astype(np.float32)

                    # Apply same fine controls
                    dst[:, 0] = dst[:, 0] + dx
                    dst[:, 1] = dst[:, 1] + dy
                    shoulder_mid = 0.5 * (dst[0] + dst[1])
                    vecs = dst - shoulder_mid
                    dst = shoulder_mid + vecs * fine_scale

                    src_tri = garment_src_triangle(png)
                    warped = warp_png_to_points(png, src_tri, dst, (h, w))
                    out = alpha_blend(frame.copy(), warped)
                    colp2.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Result", use_column_width=True)

st.markdown(
    """
    <p class="small-note">Issues on Windows with camera? In Chrome, allow camera permission for localhost and ensure no other app is using the webcam. If you see a black feed, try disabling hardware acceleration or run: <code>setx OPENCV_VIDEOIO_PRIORITY_MSMF 0</code> and restart terminal.</p>
    """,
    unsafe_allow_html=True,
)
