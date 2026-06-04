import av
import cv2
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import logging
import urllib.request
import json

# Suppress verbose logging
logging.getLogger("streamlit_webrtc").setLevel(logging.ERROR)
logging.getLogger("aiortc").setLevel(logging.ERROR)
logging.getLogger("aioice").setLevel(logging.ERROR)

MODEL_PATH = "my_model.pt"
CONF_THRESHOLD = 0.50
YOLO_IMGSZ = 320

ACTION_MAP = {
    "one": "Next Slide",
    "peace": "Previous Slide",
    "peace_inverted": "Previous Slide",
    "like": "Start Slide Show",
    "fist": "Lock / Stop Detecting",
    "palm": "Unlock / Resume Detecting"
}

st.set_page_config(page_title="Gesture Recognition Web App", layout="wide")
st.title("🖐️ Gesture Recognition Web App")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Gesture Actions")
    for gesture, action in ACTION_MAP.items():
        st.markdown(f"**{gesture}** → {action}")
    st.markdown("---")
    st.caption("Confidence threshold: 0.50")

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ── ICE Servers (multiple fallbacks for Streamlit Cloud) ──────────────────────
@st.cache_data(ttl=3600)
def get_ice_servers():
    """
    Returns a list of ICE servers.
    Tries multiple free TURN providers with fallback to STUN-only.
    For production, replace with your own Metered.ca or Twilio TURN credentials.
    """
    return [
        # Google STUN
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        # Cloudflare STUN
        {"urls": ["stun:stun.cloudflare.com:3478"]},
        # Free TURN - openrelay (UDP + TCP + TLS)
        {
            "urls": [
                "turn:openrelay.metered.ca:80",
                "turn:openrelay.metered.ca:80?transport=tcp",
            ],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": [
                "turn:openrelay.metered.ca:443",
                "turn:openrelay.metered.ca:443?transport=tcp",
            ],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        # Backup TURN
        {
            "urls": ["turn:relay.webwormhole.io:443?transport=tcp"],
            "username": "user",
            "credential": "pass",
        },
    ]

rtc_configuration = RTCConfiguration({"iceServers": get_ice_servers()})

# ── Video Processor ────────────────────────────────────────────────────────────
class GestureDetector(VideoProcessorBase):
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")

            results = model(
                img,
                imgsz=YOLO_IMGSZ,
                conf=CONF_THRESHOLD,
                verbose=False
            )

            annotated = results[0].plot()
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                best_box = max(boxes, key=lambda b: float(b.conf[0]))
                cls_id = int(best_box.cls[0])
                conf = float(best_box.conf[0])
                label = model.names[cls_id]
                action = ACTION_MAP.get(label, "No Action")

                cv2.putText(
                    annotated,
                    f"Gesture: {label} ({conf:.2f})",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )
                cv2.putText(
                    annotated,
                    f"Action: {action}",
                    (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA
                )
            else:
                cv2.putText(
                    annotated,
                    "No gesture detected",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        except Exception as e:
            # Return original frame unmodified so stream doesn't crash
            img = frame.to_ndarray(format="bgr24")
            cv2.putText(
                img,
                f"Error: {str(e)[:50]}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1
            )
            return av.VideoFrame.from_ndarray(img, format="bgr24")

# ── Compact camera style ───────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stElementContainer"] iframe {
        max-width: 400px !important;
        height: 320px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ── Instructions ───────────────────────────────────────────────────────────────
st.info("📷 Click **START** and allow camera access. If it stops, wait 3 seconds and click START again.")

# ── WebRTC Streamer ────────────────────────────────────────────────────────────
ctx = webrtc_streamer(
    key="gesture-detection",
    video_processor_factory=GestureDetector,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 320, "max": 640},
            "height": {"ideal": 240, "max": 480},
            "frameRate": {"ideal": 15, "max": 24}
        },
        "audio": False
    },
    rtc_configuration=rtc_configuration,
    async_processing=True,
)

# ── Connection status ──────────────────────────────────────────────────────────
if ctx and ctx.state.playing:
    st.success("✅ Camera connected and running!")
elif ctx:
    st.warning("⏳ Connecting... if it stops, click START again.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Deployed on Streamlit Cloud · Camera requires HTTPS (enabled by default on Streamlit Cloud)")
