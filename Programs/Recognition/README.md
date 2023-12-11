# AppBell-Recognition-Program

# Overview

This repository contains a Python script for face recognition in videos using the FastAPI framework. The script processes a video file, extracts faces from each frame, and performs face recognition to identify individuals. The face recognition is based on the VGG-Face model and supports multiple distance metrics.

# Prerequisites

Before using the script, make sure you have the following:

1. Python installed (version 3.6 or later)
2. DeepFace
3. RetinaFace
4. OpenCV

# Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/MurderousNinja/AppBell-Recognition.git
   cd AppBell-Recognition
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:

   ```bash
   python Recognition.py
   ```

   Follow the prompts to enter your Business ID (bid) and upload a video file for processing.

# Code Description

## Dependencies

- `os`: Operating system interface.
- `shutil`: File operations.
- `concurrent.futures`: Asynchronous execution of tasks.
- `cv2`: OpenCV for image processing.
- `time`: Time-related functions.
- `pickle`: Serialization and deserialization of Python objects.
- `path` (from `os`): Path manipulation.
- `FastAPI`: Web framework for building APIs.
- `UploadFile`, `Request`, `Response`, `File`, `Form` (from `fastapi`): FastAPI components for handling HTTP requests and responses.
- `HTMLResponse`, `JSONResponse`, `RedirectResponse` (from `fastapi.responses`): Response classes.
- `RetinaFace`: Face detection library.
- `ngrok`: Exposes local servers to the internet.
- `DeepFace`: Face recognition library.
- `pandas`: Data manipulation and analysis.

## Configuration

The `Config` class contains various configuration parameters, such as folder names, model names, and detection thresholds.

## Classes

### `VideoManager`

- `extract_faces_from_frame`: Extracts faces from a video frame using RetinaFace.
- `process_video_to_faces`: Processes a video, extracts frames, and calls `extract_faces_from_frame` for face extraction.

### `FaceRecog`

- `get_id_data`: Performs face recognition for a given image and Business ID (bid).
- `sort_ids`: Appends face recognition results to a global list.
- `get_ids_from_faces`: Processes a directory of face images using multiple threads and retrieves face IDs.

### `Cleanup`

- `clean`: Deletes specified directories and their contents.

## Functions

- `find`: Performs face recognition for a given image and database path.
- `process_video`: Processes an uploaded video file, performs face recognition, and returns results.

# Note

Make sure to handle sensitive information securely, as the script deals with face recognition and business-related IDs. Additionally, follow any legal and ethical considerations when using face recognition technology.
