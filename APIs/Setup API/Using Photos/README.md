# AppBell-Setup-API

## Introduction
This FastAPI application is designed for real-time face recognition and representation using a webcam. It captures video frames from the webcam, detects faces using RetinaFace, and then represents the faces using the VGG-Face model. The processed faces are stored as embeddings in a pickle (PKL) file for future reference.

## Prerequisites

* Python 3.x
* DeepFace
* Open-CV
* RetinaFace
* FastAPI
* Ngrok Account and AuthToken
* Camera Access

## Useage

### Clone the repository:

```bash
git clone https://github.com/MurderousNinja/AppBell-Setup-Photos-API
cd AppBell-Setup-Photos-API
```

### Install required packages using:

```bash
pip install -r requirements.txt
```

### Run the FastAPI application:

```bash
uvicorn RecogAPI:app --reload
```

### Access API

Access the API at http://127.0.0.1:8000.

# Code Description

## Dependencies

### Standard Library Imports:

* **os**: Provides a way of interacting with the operating system, like reading or writing to the file system.
* **pickle**: Used for serializing and deserializing Python objects.
* **shutil**: Allows file operations, such as moving or deleting directories.
* **time**: Provides time-related functions.
* **concurrent.futures.ThreadPoolExecutor**: Allows concurrent execution of functions using threads.
* **numpy**: A powerful library for numerical operations in Python.

### Image Processing Libraries:

* **cv2 (OpenCV)**: A popular computer vision library for image and video processing.

### File and Path Handling:

* **path (from os)**: Allows handling of file paths.

### Progress Tracking:

* **tqdm**: Provides a progress bar for loops, making it easier to track the progress of operations.

### Web Framework:

* **FastAPI**: A modern, fast web framework for building APIs with Python.
* **FastAPI.responses**: Classes for generating different types of HTTP responses.
* **uvicorn**: A lightweight ASGI server to run FastAPI applications.

### Face Detection and Recognition Libraries:

* **RetinaFace**: A face detection library.
* **deepface**: A deep learning-based face recognition library.

### Tunneling and Networking:

* **pyngrok**: Integrates Ngrok tunneling service with Python for exposing a local server to the internet.

## Configuration

The code includes some global variables with constant values for various configuration parameters. These parameters include folder names, model names, distance metrics, and other settings. These variables ensure that required folders exist and provides a centralized way to manage configuration parameters.

## Classes and functions

### WebcamRecorder Class

* **__init__(self, camera_index=0)**: Initializes the WebcamRecorder object with a specified camera index. Raises an exception if the webcam cannot be opened.
* **get_frame(self)**: Captures a frame from the webcam and returns it.
* **generate_filename(self, count)**: Generates a filename for the recorded frames based on the count.
* **record_video(self)**: Records video frames from the webcam, saves them to the 'Recorded' folder, and displays the frames. Stops recording when 'q' is pressed or a certain number of frames are recorded.

### FaceExtractor Class

* **process_frame(self, frame, id)**: Processes each frame to extract faces using RetinaFace. Saves the largest face to a processed frames folder.
* **video_to_faces_parallel(self, frames, id)**: Processes video frames in parallel to extract faces using RetinaFace.

### Cleanup Class

* **clean()**: Cleans up recorded and processed directories by deleting them and their contents.

### Deep Class

* **represent**: Represents faces in an image using the specified deep face recognition model. Returns embeddings and facial area information.
* **Create_PKL**: Creates a pickle file with face representations from processed images.

## Endpoints

### Home Page Endpoint

* URL: /
* Method: GET
* Description: Displays an HTML form for user input, prompting the user to enter an ID and business ID.

### Processing Video Endpoint

* URL: /processing_video
* Method: GET
* Parameters:
* id: User ID
* bid: Business ID
* Description: Initiates the video recording and processing based on user input. This includes recording frames, extracting faces, creating face representations, and cleaning up.

### Post Face Representation Endpoint

* URL: /postfacerep
* Method: GET
* Description: Displays an HTML page indicating that face representation is done.

# Notes
* The system uses ngrok to create a public URL. Make sure to set your ngrok auth token.
* Adjust the configuration constants in the script as needed, such as the camera index, face detection threshold, and model name for face recognition.
* The application cleans up recorded and processed frames automatically after face representation.
