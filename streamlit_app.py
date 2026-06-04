import av
import cv2
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import logging

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

# ── Sidebar info ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Gesture Actions")
    for gesture, action in ACTION_MAP.items():
        st.markdown(f"**{gesture}** → {action}")
    st.markdown("---")
    st.caption("Confidence threshold: 0.50")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ── Video Processor ───────────────────────────────────────────────────────────
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
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )
                cv2.putText(
                    annotated,
                    f"Action: {action}",
                    (10, 40),
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
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        except Exception as e:
            # On error, return the original frame unmodified so stream doesn't crash
            img = frame.to_ndarray(format="bgr24")
            cv2.putText(
                img,
                f"Error: {str(e)[:40]}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1
            )
            return av.VideoFrame.from_ndarray(img, format="bgr24")

# ── RTC Configuration (STUN + TURN for reliable connections) ──────────────────
rtc_configuration = RTCConfiguration(
    {"iceServers": [
        # Google STUN servers
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        # Free TURN server (helps behind NAT/firewalls)
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        },
        {
            "urls": ["turn:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        },
        {
            "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        },
    ]}
)

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
st.info("📷 Click **START** below, then allow camera access when your browser asks.")

# ── WebRTC Streamer ────────────────────────────────────────────────────────────
webrtc_streamer(
    key="gesture-detection",
    video_processor_factory=GestureDetector,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 320, "max": 640},
            "height": {"ideal": 240, "max": 480},
            "frameRate": {"ideal": 15, "max": 30}
        },
        "audio": False
    },
    rtc_configuration=rtc_configuration,
    async_processing=True,
)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Make sure you're on **HTTPS** or **localhost** for camera access to work.")
