# Standard Library Imports
import os
import pickle
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

# Numeric and Scientific Libraries
import numpy as np

# Image Processing Libraries
import cv2

# File and Path Handling
from os import path

# Progress Tracking
from tqdm import tqdm

# Web Framework
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse

# Face Detection Libraries
from retinaface import RetinaFace

# Face Recognition Libraries
from deepface import DeepFace
import deepface.commons.functions as functions

# Tunneling and Networking
from pyngrok import ngrok

# Web Server
import uvicorn

# Set your Ngrok auth token here
ngrok.set_auth_token("2VQ3N6dnV7bMo5PhJskNw6KOynM_43X2AJPHbCN42wdjt29jb")

# Constants

# Maximum number of workers for face processing and PKL creation
FACE_MAX_WORKERS = 14
PKL_MAX_WORKERS = 12

# Get the current working directory
DIRECTORY = os.getcwd()

# Directories for different stages of the process
RECORDED_FOLDER = "Recorded"       # Directory for recorded frames
FACE_FOLDER = "Data"               # Directory for processed faces
PROCESSING_FOLDER = "Processing"   # Temporary directory for processing frames
PKL_FOLDER = "PKL"                 # Directory for storing PKL files
UPLOADS_FOLDER = "Uploads"         # Directory for uploaded videos

# Face detection threshold
FACE_THRESHOLD = 0.99

# Frames per second for video recording
FPS = 30

# Camera index (0 for default camera)
CAMERA_INDEX = 0

# Model name for face recognition
MODEL_NAME = "VGG-Face"

# List of folders to be created if they don't exist
folders = [FACE_FOLDER, RECORDED_FOLDER, PROCESSING_FOLDER, PKL_FOLDER]

# Create output directories if they don't exist
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Create a FastAPI instance
app = FastAPI()

# Define a class for handling webcam recording
class WebcamRecorder:

    def __init__(self, camera_index=0):
        # Initialize the webcam recorder with the specified camera index
        self.camera_index = camera_index
        self.capture = cv2.VideoCapture(self.camera_index)
        
        # Check if the webcam is successfully opened
        if not self.capture.isOpened():
            raise Exception("Could not open the webcam.")

    def get_frame(self):
        # Capture a frame from the webcam
        ret, frame = self.capture.read()
        
        # Check if the frame is successfully captured
        if not ret:
            raise Exception("Could not read a frame from the webcam.")
        return frame

    def generate_filename(self, count):
        # Generate a filename for the recorded frames
        return f'Frame {count:02d}.jpg'

    def record_video(self):
        # Record video frames from the webcam and save them to the 'Recorded' folder
        count = -1
        frames = []
        
        # Continuously capture frames until a certain condition is met
        while True:
            count += 1
            filename = os.path.join(RECORDED_FOLDER, self.generate_filename(count))
            frame = self.get_frame()
            frames.append(frame)
            
            # Save the frame as an image file
            cv2.imwrite(filename, frame)
            print(filename)
            
            # Display the frame in a window named 'Webcam Feed'
            cv2.imshow('Webcam Feed', frame)
            
            # Break the loop if 'q' is pressed or a certain number of frames are recorded
            if cv2.waitKey(1000 // FPS) & 0xFF == ord('q') or count > 98:
                break

        # Release the webcam capture and close the display window
        self.capture.release()
        cv2.destroyAllWindows()

        return frames

# Define a class for face extraction using RetinaFace
class FaceExtractor:
    def process_frame(self, frame, id):
        # Process each frame to extract faces using RetinaFace
        try:
            # Extract the frame count from the file path
            count = frame.split('\\')[1].split(' ')[1].split('.')[0]
            
            # Use RetinaFace to extract faces from the frame
            faces = RetinaFace.extract_faces(frame, threshold=FACE_THRESHOLD, model=None, allow_upscaling=True)

            # Check if any faces are extracted
            if not faces:
                print(f"No faces extracted in Frame {count}.")
                return None

            # Create a directory for processed frames if it doesn't exist
            path0 = os.path.join(DIRECTORY, PROCESSING_FOLDER)
            if not os.path.exists(path0):
                os.makedirs(path0)

            # Find the index of the largest face based on image size
            image_sizes = [np.prod(image.shape) for image in faces]
            largest_index = np.argmax(image_sizes)
            
            # Generate a unique filename for the processed face
            filename2 = f"user.{id}.{count}.jpg"
            result = os.path.join(path0, filename2)
            
            # Save the largest face as an image file
            print(f"user.{id}.{count}.jpg written")
            cv2.imwrite(result, faces[largest_index])

            return count

        except Exception as e:
            print(str(e))
            return None

    def video_to_faces_parallel(self, frames, id):
        # Process video frames in parallel to extract faces
        with ThreadPoolExecutor(max_workers=FACE_MAX_WORKERS) as executor:
            futures = [executor.submit(self.process_frame, frame, id) for frame in frames]
            
            # Wait for all threads to complete
            for future in futures:
                future.result()

# Define a class for cleaning up directories
class Cleanup:
    @staticmethod
    def clean():
        # Clean up recorded and processed directories
        directories = (os.path.join(DIRECTORY, a) for a in [RECORDED_FOLDER, PROCESSING_FOLDER])
        
        try:
            # Iterate through the directories and delete them along with their contents
            for directory_path in directories:
                shutil.rmtree(directory_path)
                print(f'Directory "{directory_path}" and its contents have been successfully deleted.')
        except Exception as e:
            # Handle exceptions if any error occurs during cleanup
            print(f'An error occurred: {e}')

# Define a class for deep face representation using a specified model
class Deep:
    @staticmethod
    def represent(img_path, model_name="VGG-Face", enforce_detection=True, detector_backend="retinaface", align=True,
                  normalization="base"):
        # Represent faces in an image using the specified model
        resp_objs = []

        # Build the specified deep face recognition model
        model = DeepFace.build_model(model_name)

        # Extract faces from the image using the specified detector backend
        target_size = functions.find_target_size(model_name=model_name)
        if detector_backend != "skip":
            img_objs = functions.extract_faces(img=img_path, target_size=target_size, detector_backend=detector_backend,
                                               grayscale=False, enforce_detection=enforce_detection, align=align)
        else:
            if isinstance(img_path, str):
                img = functions.load_image(img_path)
            elif type(img_path).__module__ == np.__name__:
                img = img_path.copy()
            else:
                raise ValueError(f"unexpected type for img_path - {type(img_path)}")
            if len(img.shape) == 4:
                img = img[0]
            if len(img.shape) == 3:
                img = cv2.resize(img, target_size)
                img = np.expand_dims(img, axis=0)
            img_region = [0, 0, img.shape[1], img.shape[0]]
            img_objs = [(img, img_region, 0)]

        # Iterate over extracted faces, normalize input, and predict embeddings
        for img, region, _ in img_objs:
            img = functions.normalize_input(img=img, normalization=normalization)
            if "keras" in str(type(model)):
                embedding = model.predict(img, verbose=0)[0].tolist()
            else:
                embedding = model.predict(img)[0].tolist()
            resp_obj = {}
            resp_obj["embedding"] = embedding
            resp_obj["facial_area"] = region
            resp_objs.append(resp_obj)

        return resp_objs

    def Create_PKL(self, model_name="VGG-Face", distance_metric="cosine", enforce_detection=False, detector_backend="opencv",
                   align=True, normalization="base", silent=False, bid=0000):
        # Create a pickle file with face representations from processed images
        tic = time.time()
        db_path = PROCESSING_FOLDER
        file_name = f"representations_{model_name}.pkl"
        file_name = file_name.replace("-", "_").lower()
        pkl = f"{PKL_FOLDER}/{bid}_{file_name}"

        # Check if the specified database path exists
        if os.path.isdir(db_path) is not True:
            raise ValueError("Passed db_path does not exist!")

        # Get the list of employee images for representation
        target_size = functions.find_target_size(model_name=model_name)
        employees = []
        for r, _, f in os.walk(db_path):
            for file in f:
                if ((".jpg" in file.lower()) or (".jpeg" in file.lower()) or (".png" in file.lower())):
                    exact_path = r + "/" + file
                    employees.append(exact_path)

        # Check if there are images in the specified folder
        if len(employees) == 0:
            raise ValueError("There is no image in ", db_path, " folder! Validate .jpg or .png files exist in this path.")

        # List to store face representations
        representations = []

        # Loop through the employee images, extract faces, and create representations
        pbar = tqdm(range(0, len(employees)), desc="Finding representations", disable=silent)
        for index in pbar:
            employee = employees[index]
            img_objs = functions.extract_faces(img=employee, target_size=target_size, detector_backend=detector_backend,
                                               grayscale=False, enforce_detection=enforce_detection, align=align)
            for img_content, _, _ in img_objs:
                # Represent each face and store in the representations list
                embedding_obj = self.represent(img_path=img_content, model_name=model_name,
                                               enforce_detection=enforce_detection, detector_backend="skip", align=align,
                                               normalization=normalization)
                img_representation = embedding_obj[0]["embedding"]
                instance = []
                instance.append(employee)
                instance.append(img_representation)
                representations.append(instance)

        # Check if a previous representation file exists, and create a backup
        if path.exists(pkl):
            with open(pkl, "rb") as f:
                representations_present = pickle.load(f)
                representations.extend(representations_present)
            backup_pkl = pkl.split('.')[0] + "_Backup.pkl"
            os.rename(pkl, backup_pkl)

        # Save the new representations to the pickle file
        with open(pkl, "wb") as f:
            pickle.dump(representations, f)

        # Print a completion message if not in silent mode
        if not silent:
            print(f"Representations stored in {pkl} file."
                  + "Please delete this file when you add new identities in your database.")

# Define the home route, which provides an HTML form to initiate the video processing
@app.get("/", response_class=HTMLResponse)
async def home():
    # HTML content for the home page with a form for user input
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prerecording</title>
    </head>
    <body>
        <div class="center-text">
            <h1>Welcome to the FastAPI API</h1>
        </div>
            
        <!-- Form for entering user ID and business ID -->
        <form method="GET" action="/processing_video">
            <label for="id">Enter an ID:</label>
            <input type="text" id="id" name="id" required>
            
            <label for="bid">Enter your Business ID:</label>
            <input type="text" id="bid" name="bid" required>
            
            <!-- Submit button to initiate video recording and processing -->
            <button type="submit">Let's start recording and processing!</button>
        </form>
    </body>
    </html>
    """
    # Return the HTML content as a response
    return HTMLResponse(content=html_content)

# Define the route for processing video and extracting faces
@app.get("/processing_video")
async def processing_video(id: str, bid: str):
    try:
        # Initialize objects for webcam recording, face extraction, and deep face representation
        recorder = WebcamRecorder(camera_index=CAMERA_INDEX)
        face_extractor = FaceExtractor()
        deep_processor = Deep()

        # Record the video and get frames using the webcam
        recorder.record_video()

        # Create a list of frame file paths in the 'Recorded' folder
        frames = list(os.path.join(RECORDED_FOLDER, a) for a in os.listdir(RECORDED_FOLDER))

        # Process frames in parallel to extract faces using RetinaFace
        face_extractor.video_to_faces_parallel(frames, id)

        # Create a pickle file with face representations using the specified model
        deep_processor.Create_PKL(bid=bid)

        # Clean up recorded and processed frames to free up space
        Cleanup.clean()

        # Redirect to the post-processing page to view the results
        return RedirectResponse("/postfacerep", status_code=200)

    except Exception as e:
        # Handle any errors that may occur during the video processing
        return {"error": str(e)}

# Define the post-processing page route
@app.get("/postfacerep", response_class=HTMLResponse)
async def postrep():
    # HTML content for the post-processing page
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Extraction Done!</title>
    </head>
    <body>
        <div class="center-text">
            <h1>Face Representation Done</h1>
        </div>
    </body>
    </html>
    """

    # Clean up recorded and processed frames

    # Return the HTML content as the response to be displayed on the post-processing page
    return HTMLResponse(content=html_content)

# Entry point for the application
if __name__ == "__main__":
    # Use ngrok to create a public URL for the FastAPI application
    public_url = ngrok.connect(port=8000)
    
    # Print the ngrok tunnel information for user reference
    print(" * ngrok tunnel \"{}\" -> http://127.0.0.1:{}/".format(public_url, 8000))
    
    # Run the FastAPI application using uvicorn
    # The FastAPI app is located in the "SetupAPI" module, and it is named "app"
    uvicorn.run("SetupPhotos:app", port=8000, reload=True)
