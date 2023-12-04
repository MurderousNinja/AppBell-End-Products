# AppBell-Setup-API

## Introduction
This FastAPI application is designed for real-time face recognition and representation using a webcam. It captures video frames from the webcam, detects faces using RetinaFace, and then represents the faces using the VGG-Face model. The processed faces are stored as embeddings in a pickle (PKL) file for future reference.

# Code Description

## Classes

### 1. Webcam Recorder
The WebcamRecorder class handles video recording from the webcam. It captures frames, saves them to the "Recorded" folder, and displays the frames in a window named 'Webcam Feed'.

### 2. Face Extractor
The FaceExtractor class uses RetinaFace to extract faces from recorded frames. It processes frames in parallel, selects the largest face from each frame, and saves it to the "Processing" folder.

### 3. Deep Face Representation
The Deep class handles deep face representation using a specified model (default is VGG-Face). It creates a PKL file containing face representations from the processed images.

### 4. Cleanup
The Cleanup class provides a method to clean up recorded and processed directories, freeing up space after face extraction and representation.

### 5. FastAPI Web Application
The FastAPI web application defines routes for user interaction:

Home Route: Provides an HTML form to input user ID and business ID.
Processing Video Route: Initiates video recording, face extraction, and deep face representation based on user input.
Post-Processing Page Route: Displays a confirmation message after face representation is complete.

# Notes
* The system uses ngrok to create a public URL. Make sure to set your ngrok auth token.
* Adjust the configuration constants in the script as needed, such as the camera index, face detection threshold, and model name for face recognition.
* The application cleans up recorded and processed frames automatically after face representation.
