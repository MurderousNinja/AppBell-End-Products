# Standard Library
import os
import pickle
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

# Numerical and Scientific Libraries
import numpy as np

# Image Processing Libraries
import cv2

# File and Path Utilities
from os import path

# Progress Bar
from tqdm import tqdm

# FastAPI and HTTP-Related Libraries
from fastapi import FastAPI, HTTPException, UploadFile, File

# Face Detection and Recognition Libraries
from retinaface import RetinaFace
from deepface import DeepFace
import deepface.commons.functions as functions

# Ngrok for Creating Public URLs
from pyngrok import ngrok

# 3rd Party Dependencies (Additional Packages)
import uvicorn

# Package Dependencies (Specific Module from deepface.commons)
from deepface.commons import functions

# Set your Ngrok auth token here
ngrok.set_auth_token("YOUR_TOKEN_HERE")

# Constants

# Maximum number of workers for face processing in parallel
FACE_MAX_WORKERS = 14

# Maximum number of workers for pickle file creation
PKL_MAX_WORKERS = 12

# Get the current working directory
DIRECTORY = os.getcwd()

#Setting a high threshold so that we are certain of a face being there.
FACE_THRESHOLD = 0.99

# Folder names for different stages of processing
RECORDED_FOLDER = "Recorded"       # Folder for storing recorded video frames
FACE_FOLDER = "Data"                # Folder for storing processed face images
PROCESSING_FOLDER = "Processing"    # Folder for temporarily storing processed images
PKL_FOLDER = "PKL"                  # Folder for storing pickle files
UPLOADS_FOLDER = "Uploads"          # Folder for uploading video files

# List of folders to be created
folders = [FACE_FOLDER, RECORDED_FOLDER, PROCESSING_FOLDER, PKL_FOLDER]

# Create output directories if they don't exist
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

app = FastAPI()

class VideoManager:

    def generate_filename(self, count):
        """
        Generate a filename for the recorded frames.

        Parameters:
        - count (int): The frame count.

        Returns:
        - str: The generated filename.
        """
        return f'Frame {count:02d}.jpg'

    def process_video(self, videoname):
        """
        Process a video file to extract frames and save them to the 'Recorded' folder.

        Parameters:
        - videoname (str): The path to the video file.

        Returns:
        - None
        """
        # Open the video file for reading
        video = cv2.VideoCapture(videoname)
        
        # Get frames per second and total number of frames in the video
        fps = video.get(cv2.CAP_PROP_FPS)
        frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        
        # Calculate the total time duration of the video in seconds
        seconds = round(frames / fps)
        
        # Calculate the time separation between frames for even distribution
        separation = seconds / 100
        
        sec = 0
        count = 0
        
        # Loop through the video and extract frames at regular intervals
        while sec < seconds:
            # Set the current position in milliseconds
            t_msec = int(1000 * sec)
            video.set(cv2.CAP_PROP_POS_MSEC, t_msec)
            
            # Read the frame
            ret, frame = video.read()
            
            # Generate the filename for the frame
            filename1 = os.path.join(RECORDED_FOLDER, self.generate_filename(count))
            
            # Save the frame to the 'Recorded' folder
            cv2.imwrite(filename1, frame)
            
            # Update counters
            count += 1
            sec += separation
        
        # Release the video capture object
        video.release()

class FaceExtractor:
    def process_frame(self, frame, id, bid):
        """
        Process a single frame to extract faces using RetinaFace.

        Parameters:
        - frame (str): The path to the frame image.
        - id (str): User ID.
        - bid (str): Business ID.

        Returns:
        - count (str): The frame count for reference.
        """
        try:
            # Extract the frame count from the filename
            count = frame.split('\\')[1].split(' ')[1].split('.')[0]

            # Use RetinaFace to extract faces from the frame
            faces = RetinaFace.extract_faces(frame, threshold=FACE_THRESHOLD, model=None, allow_upscaling=True)

            # Check if any faces are extracted
            if not faces:
                print(f"No faces extracted in Frame {count}.")
                return None

            # Create a processing folder if it doesn't exist
            path0 = os.path.join(DIRECTORY, PROCESSING_FOLDER)
            if not os.path.exists(path0):
                os.makedirs(path0)

            # Find the largest face in terms of image size
            image_sizes = [np.prod(image.shape) for image in faces]
            largest_index = np.argmax(image_sizes)

            # Generate a filename for the extracted face
            filename2 = f"user.{id}.{count}.jpg"
            result = os.path.join(path0, filename2)
            print(f"user.{id}.{count}.jpg written")

            # Save the largest face to the processing folder
            cv2.imwrite(result, faces[largest_index])

            return count

        except Exception as e:
            print(str(e))
            return None

    def video_to_faces_parallel(self, frames, id):
        """
        Process video frames in parallel to extract faces.

        Parameters:
        - frames (list): List of paths to video frames.
        - id (str): User ID.

        Returns:
        - None
        """
        with ThreadPoolExecutor(max_workers=FACE_MAX_WORKERS) as executor:
            # Submit face extraction tasks for each frame in parallel
            futures = [executor.submit(self.process_frame, frame, id) for frame in frames]
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()

class Cleanup:
    @staticmethod
    def clean():
        """
        Clean up the recorded and processed directories.

        Removes the entire contents of the 'Recorded' and 'Processing' directories.

        Returns:
        - None
        """
        directory_path1 = os.path.join(DIRECTORY, RECORDED_FOLDER)
        directory_path2 = os.path.join(DIRECTORY, PROCESSING_FOLDER)
        try:
            # Remove the 'Recorded' directory and its contents
            shutil.rmtree(directory_path1)
            print(f'Directory "{directory_path1}" and its contents have been successfully deleted.')

            # Remove the 'Processing' directory and its contents
            shutil.rmtree(directory_path2)
            print(f'Directory "{directory_path2}" and its contents have been successfully deleted.')
        except Exception as e:
            print(f'An error occurred: {e}')

    @staticmethod
    def move():
        """
        Move processed images from the 'Processing' folder to the 'Data' folder.

        If a file with the same name already exists in the 'Data' folder, it is replaced.

        Returns:
        - None
        """
        old_fold = os.path.join(DIRECTORY, PROCESSING_FOLDER)
        new_fold = os.path.join(DIRECTORY, FACE_FOLDER)

        # Create the 'Data' folder if it doesn't exist
        if not os.path.exists(FACE_FOLDER):
            os.makedirs(FACE_FOLDER)

        # Iterate through files in the 'Processing' folder
        for filename in os.listdir(old_fold):
            old_name = os.path.join(old_fold, filename)
            new_name = os.path.join(new_fold, filename)

            # If the file doesn't exist in 'Data', move it
            if not os.path.exists(new_name):
                os.rename(old_name, new_name)
            else:
                # Remove the existing file in 'Data', then move the new one
                os.remove(new_name)
                os.rename(old_name, new_name)

        print("Moved!!")

class Deep:
    @staticmethod
    def represent(img_path, model_name="VGG-Face", enforce_detection=True, detector_backend="retinaface", align=True,
                  normalization="base"):
        """
        Extract facial embeddings using a specified deep learning model.

        Args:
        - img_path (str): Path to the image file.
        - model_name (str): Name of the deep learning model to use (default is "VGG-Face").
        - enforce_detection (bool): Whether to enforce face detection (default is True).
        - detector_backend (str): Backend for face detection (default is "retinaface").
        - align (bool): Whether to align faces during processing (default is True).
        - normalization (str): Type of normalization to apply (default is "base").

        Returns:
        - resp_objs (list): List of dictionaries containing facial embeddings and facial area information.
        """
        resp_objs = []

        # Build the specified deep learning model
        model = DeepFace.build_model(model_name)

        target_size = functions.find_target_size(model_name=model_name)

        # Extract faces from the image
        if detector_backend != "skip":
            img_objs = functions.extract_faces(img=img_path, target_size=target_size, detector_backend=detector_backend,
                                               grayscale=False, enforce_detection=enforce_detection, align=align)
        else:
            # If face detection is skipped, create an image object
            if isinstance(img_path, str):
                img = functions.load_image(img_path)
            elif type(img_path).__module__ == np.__name__:
                img = img_path.copy()
            else:
                raise ValueError(f"unexpected type for img_path - {type(img_path)}")

            # Handle different dimensions of the image
            if len(img.shape) == 4:
                img = img[0]
            if len(img.shape) == 3:
                img = cv2.resize(img, target_size)
                img = np.expand_dims(img, axis=0)

            img_region = [0, 0, img.shape[1], img.shape[0]]
            img_objs = [(img, img_region, 0)]

        # Iterate through image objects, normalize input, and predict embeddings
        for img, region, _ in img_objs:
            img = functions.normalize_input(img=img, normalization=normalization)

            # Predict facial embeddings using the model
            if "keras" in str(type(model)):
                embedding = model.predict(img, verbose=0)[0].tolist()
            else:
                embedding = model.predict(img)[0].tolist()

            # Create a dictionary with embedding and facial area information
            resp_obj = {"embedding": embedding, "facial_area": region}
            resp_objs.append(resp_obj)

        return resp_objs

    def Create_PKL(self, model_name="VGG-Face", distance_metric="cosine", enforce_detection=False,
                   detector_backend="opencv", align=True, normalization="base", silent=False, bid=0000):
        """
        Create a pickle file with face representations from processed images.

        Args:
        - model_name (str): Name of the deep learning model to use (default is "VGG-Face").
        - distance_metric (str): Distance metric for face verification (default is "cosine").
        - enforce_detection (bool): Whether to enforce face detection (default is False).
        - detector_backend (str): Backend for face detection (default is "opencv").
        - align (bool): Whether to align faces during processing (default is True).
        - normalization (str): Type of normalization to apply (default is "base").
        - silent (bool): Whether to disable progress bar display (default is False).
        - bid (int): Business ID for creating a unique pickle file (default is 0000).

        Returns:
        - None
        """
        tic = time.time()
        db_path = PROCESSING_FOLDER
        file_name = f"representations_{model_name}.pkl"
        file_name = file_name.replace("-", "_").lower()
        pkl = f"{PKL_FOLDER}/{bid}_{file_name}"

        # Check if the specified directory exists
        if os.path.isdir(db_path) is not True:
            raise ValueError("Passed db_path does not exist!")

        target_size = functions.find_target_size(model_name=model_name)
        employees = []

        # Iterate through files in the 'Processing' folder
        for r, _, f in os.walk(db_path):
            for file in f:
                if ((".jpg" in file.lower()) or (".jpeg" in file.lower()) or (".png" in file.lower())):
                    exact_path = r + "/" + file
                    employees.append(exact_path)

        # Raise an error if no images are found in the 'Processing' folder
        if len(employees) == 0:
            raise ValueError("There is no image in ", db_path, " folder! Validate .jpg or .png files exist in this path.")

        representations = []

        # Iterate through images and create representations
        pbar = tqdm(range(0, len(employees)), desc="Finding representations", disable=silent)
        for index in pbar:
            employee = employees[index]
            img_objs = functions.extract_faces(img=employee, target_size=target_size, detector_backend=detector_backend,
                                               grayscale=False, enforce_detection=enforce_detection, align=align)
            for img_content, _, _ in img_objs:
                embedding_obj = self.represent(img_path=img_content, model_name=model_name,
                                               enforce_detection=enforce_detection, detector_backend="skip", align=align,
                                               normalization=normalization)
                img_representation = embedding_obj[0]["embedding"]
                instance = [employee, img_representation]
                representations.append(instance)

        # If the pickle file already exists, load existing representations and extend the list
        if path.exists(pkl):
            with open(pkl, "rb") as f:
                representations_present = pickle.load(f)
                representations.extend(representations_present)
            backup_pkl = pkl.split('.')[0] + "_Backup.pkl"
            os.rename(pkl, backup_pkl)

        # Save the representations to the pickle file
        with open(pkl, "wb") as f:
            pickle.dump(representations, f)

        # Print a message about the storage location
        if not silent:
            print(f"Representations stored in {pkl} file."
                  + "Please delete this file when you add new identities in your database.")

@app.get("/processing_video")
async def processing_video(id: str, bid: str, video_file: UploadFile = File(...)):
    try:
        # Check if the file has an MP4 extension
        if not video_file.filename.lower().endswith(".mp4"):
            raise HTTPException(status_code=400, detail="Only MP4 files are allowed")

        # Define the file path where the video will be saved
        file_path = os.path.join(UPLOADS_FOLDER, f"{bid}_user_{id}.mp4")
        
        # Save the uploaded video to the specified path
        with open(file_path, "wb") as file:
            file.write(video_file.file.read())
        
        # Initialize the Class Objects for video recording, face extraction, and deep processing
        recorder = VideoManager()
        face_extractor = FaceExtractor()
        deep_processor = Deep()

        # Record the video and get frames
        recorder.process_video(filename=file_path)

        # Get the list of recorded frames
        frames = list(os.path.join(RECORDED_FOLDER, a) for a in os.listdir(RECORDED_FOLDER))

        # Process frames in parallel to extract faces
        face_extractor.video_to_faces_parallel(frames=frames, id=id, bid=bid)

        # Create a pickle file with face representations
        deep_processor.Create_PKL(bid=bid)

        # Move and clean up recorded and processed frames
        Cleanup.move()
        Cleanup.clean()

        # Return a success message
        return "Done!"

    except Exception as e:
        # Return an error message if an exception occurs
        return {"error": str(e)}

if __name__ == "__main__":
    # Use ngrok to create a public URL for the FastAPI application on port 8000
    public_url = ngrok.connect(8000, "http")
    print(" * ngrok tunnel \"{}\" -> http://127.0.0.1:{}/".format(public_url, 8000))
    
    # Run the FastAPI application using uvicorn on port 8000 with auto-reload
    uvicorn.run("SetupVideo:app", port=8000, reload=True)