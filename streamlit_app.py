import av
import cv2
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import logging

# Suppress verbose logging
logging.getLogger("streamlit_webrtc").setLevel(logging.ERROR)

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
st.title("Gesture Recognition Web App")

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

class GestureDetector(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model(
            img,
            imgsz=YOLO_IMGSZ,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        annotated = results[0].plot()

        boxes = results[0].boxes

        if len(boxes) > 0:
            best_box = max(boxes, key=lambda b: float(b.conf[0]))
            cls_id = int(best_box.cls[0])
            conf = float(best_box.conf[0])
            label = model.names[cls_id]
            action = ACTION_MAP.get(label, "No Action")

            cv2.putText(
                annotated,
                f"Gesture: {label} ({conf:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
                f"Simulated Action: {action}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

        else:
            cv2.putText(
                annotated,
                "No gesture detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# RTC Configuration with better server support
rtc_configuration = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]}
)

# Webcam feed - smaller size
col = st.columns([1])[0]
with col:
    webrtc_streamer(
        key="gesture-detection",
        video_processor_factory=GestureDetector,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 320},
                "height": {"ideal": 240}
            },
            "audio": False
        },
        rtc_configuration=rtc_configuration,
        async_processing=True,
    )
