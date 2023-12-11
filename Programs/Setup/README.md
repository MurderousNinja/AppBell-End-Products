# AppBell-Setup-Program

# Overview

This Python script is designed to record a video from a webcam, extract faces using the RetinaFace algorithm, represent those faces using the DeepFace library, and store the representations in a Pickle file. The script allows for parallel processing of video frames to improve efficiency.

# Prerequisites

- Python 3.x
- OpenCV
- NumPy
- RetinaFace
- DeepFace
- tqdm

# Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the script:**
   ```bash
   python Setup.py
   ```

   Follow the prompts to enter your ID and Business ID.

# Code Description

## Dependencies

- `os`: Operating system module for file and directory operations.
- `pickle`: Serialization module for storing and retrieving Python objects.
- `shutil`: File operations module for moving and deleting directories.
- `time`: Time-related functions.
- `concurrent.futures.ThreadPoolExecutor`: Library for parallel execution of tasks.
- `numpy`: Numerical computing library.
- `cv2`: OpenCV library for computer vision tasks.
- `tqdm`: Progress bar library.
- `RetinaFace`: Face detection library.
- `DeepFace`: Deep learning-based face recognition library.

## Configuration

- `FACE_MAX_WORKERS`: Maximum number of workers for face extraction parallelization.
- `PKL_MAX_WORKERS`: Maximum number of workers for Pickle file creation parallelization.
- `DIRECTORY`: Current working directory.
- `RECORDED_FOLDER`: Folder to store recorded video frames.
- `FACE_FOLDER`: Folder to store extracted faces.
- `PROCESSING_FOLDER`: Folder for temporary storage during face extraction.
- `FACE_THRESHOLD`: Confidence threshold for face extraction.
- `FPS`: Frames per second for video recording.
- `CAMERA_INDEX`: Index of the webcam to use.
- `MODEL_NAME`: Name of the face recognition model (e.g., "VGG-Face").
- `PKL_FOLDER`: Folder to store Pickle files.
- `folders`: List of folders to be created if they don't exist.

## Classes

### `WebcamRecorder`

- `__init__(self, camera_index)`: Constructor to initialize the WebcamRecorder object.
- `get_frame(self)`: Capture and return a frame from the webcam.
- `generate_filename(self, count)`: Generate a filename for a given frame count.
- `record_video(self)`: Record video from the webcam and save frames.

### `FaceExtractor`

- `process_frame(self, frame, id)`: Extract faces from a frame using RetinaFace.
- `video_to_faces_parallel(self, frames, id)`: Process video frames in parallel to extract faces.

### `Cleanup`

- `clean()`: Clean up recorded and processing directories.
- `move()`: Move extracted faces from the processing directory to the data directory.

### `Deep`

- `represent(img_path, model_name, enforce_detection, detector_backend, align, normalization)`: Extract facial representations using DeepFace.
- `Create_PKL(self, model_name, distance_metric, enforce_detection, detector_backend, align, normalization, silent, bid)`: Create a Pickle file containing facial representations.

## Functions

- `processing_video(id, bid)`: Main function to coordinate video recording, face extraction, representation creation, and cleanup.

# Note

Make sure to install the required packages mentioned in the "Prerequisites" section before running the script. Additionally, configure the constants and parameters in the script according to your preferences.
