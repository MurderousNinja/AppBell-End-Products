# Face Recognition API using FastAPI and DeepFace

## Overview

This repository contains a FastAPI-based API for face recognition in videos. The API utilizes various dependencies, including face detection (RetinaFace), face recognition (DeepFace), and ngrok for creating a public tunnel.

## 

### Prerequisites

* Python 3.x
* FastAPI
* DeepFace
* RetinaFace
* Ngrok

## Useage

### Clone the repository:

```bash
git clone https://github.com/MurderousNinja/AppBell-Face-Recog-API
cd AppBell-Face-Recog-API
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

***

# Code Description

## Dependencies

### Inbuilt Dependencies

* **os**: Operating system module for interaction.
* **shutil**: Utility functions for file operations.
* **concurrent.futures**: Provides a high-level interface for asynchronously executing functions.
* **time**: Module for handling time-related tasks.
* **pickle**: Serialization and deserialization of Python objects.
* **path**: Module for working with file paths.
* **pandas**: Library for data manipulation and analysis.

### External Dependencies

* **cv2** (OpenCV): Library for computer vision tasks, used for image processing.
* **FastAPI**: Modern, fast (high-performance), web framework for building APIs with Python.
* **FastAPI.UploadFile**, Request, File, Form: Components from FastAPI for handling file uploads and form data.
* **FastAPI.JSONResponse**: Class from FastAPI for creating JSON responses.
* **starlette.requests**: Request handling module from the Starlette framework.
* **uvicorn**: ASGI server to run the FastAPI application.
* **ngrok**: Provides a public URL for the local FastAPI application.

### Public API Dependency

* **pyngrok**: Python wrapper for ngrok, used for creating a public URL.

### Face Extraction Dependency

* **RetinaFace**: Face detection library for extracting faces from images.

### Face Recognition Dependencies

* **DeepFace**: Face recognition library that wraps various deep learning models.

## Configuration

The code includes a Config class with constant values for various configuration parameters. These parameters include folder names, model names, distance metrics, and other settings. The class ensures that required folders exist and provides a centralized way to manage configuration parameters.

## Classes
### 1. Config
This class manages configuration parameters and ensures the existence of required folders.

### **2. VideoManager**
The VideoManager class handles video processing tasks, including frame extraction and face extraction using RetinaFace. It uses multithreading for parallel face extraction from video frames.

### **3. FaceRecog**
The FaceRecog class performs face recognition using the find function, extracts identification information, and manages the global list of identification data. It also provides methods for getting IDs from faces and sorting identification data.

### **4. Cleanup**
The Cleanup class contains a static method clean to delete specified directories and their contents. It is used for cleanup operations after processing.

## Functions
### **1. find**
The find function performs face recognition using the DeepFace library. It compares faces in an input image with representations stored in a database and returns a list of DataFrames containing identification information.

### **2. extract_faces_from_frame**
The extract_faces_from_frame function extracts faces from a video frame using RetinaFace and saves them as separate images.

### **3. process_video_to_faces**
The process_video_to_faces function processes a video file, extracting faces from each frame and saving them to the specified folders.

### **4. get_id_data**
The get_id_data function performs face recognition on a single image and returns the identified person's ID.

### **5. sort_ids**
The sort_ids function appends new identification data to the global list.

### **6. get_ids_from_faces**
The get_ids_from_faces function performs face recognition on a directory of faces, returning identified IDs and the stored identification data.

### **7. clean**
The clean function deletes specified directories and their contents for cleanup.

### **8. process_video**
The process_video function handles the endpoint for processing video files. It saves the uploaded video, extracts faces, performs face recognition, and returns the results as a JSON response.

## Endpoints
### **1. /process_video**
The /process_video endpoint allows users to upload a video file for face recognition. It returns JSON containing identified IDs, stored identification data, and counts of each ID.

# Note

* Ensure that the required dependencies are installed before running the code.
* The code uses ngrok for creating a public URL, so an active internet connection is necessary.
* The process_video endpoint accepts a video file upload with a unique bid (Batch ID).
* Identification results, including IDs and counts, are returned as JSON.
* Temporary video files and extracted images are cleaned up after processing.
* Feel free to customize the code according to your specific use case and requirements.
