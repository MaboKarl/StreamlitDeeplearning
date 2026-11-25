# Hand Gesture-Controlled PowerPoint

A Streamlit application for detecting hand gestures from a webcam and associating them with PowerPoint presentation actions. The project uses a YOLOv4 hand-gesture model and a public dataset of approximately 8,000 images from Roboflow, labeled with Label Studio.

## What it does

The app streams webcam video to a YOLO model, annotates detected gestures, and displays the highest-confidence gesture and its mapped action. The confidence threshold is `0.50`, and inference uses an image size of `320`.

| Detected gesture | Mapped action |
| --- | --- |
| `one` | Next Slide |
| `peace` | Previous Slide |
| `peace_inverted` | Previous Slide |
| `like` | Start Slide Show |
| `fist` | Lock / Stop Detecting |
| `palm` | Unlock / Resume Detecting |

## Requirements

- Python 3.10
- A webcam
- A browser with camera access

## Installation

Create a virtual environment and install the dependencies:

### Windows

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the app

From the repository root:

```bash
streamlit run streamlit_app.py
```

On Windows, if the virtual environment is not activated, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit and allow the browser to use your webcam. A deployed app must use HTTPS for browser camera access.

## Model files

- `my_model.pt` is the checkpoint loaded by the current application.
- `jamod.pt` is included in the repository but is not referenced by the current application code.

The model file must remain at the path expected by `streamlit_app.py`.

## Privacy

The browser sends video frames to the app for processing. If the app is deployed on a server, that server receives the camera stream; only use deployments you trust.

## Dependencies

Dependencies are listed in [`requirements.txt`](requirements.txt). The configured runtime is specified in [`runtime.txt`](runtime.txt).
