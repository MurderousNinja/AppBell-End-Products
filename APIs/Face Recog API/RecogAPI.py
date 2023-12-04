#Inbuilt Dependencies
import os
import shutil
import concurrent.futures
import cv2
import time
import pickle
from os import path
import pandas as pd

#External Dependencies

#API Dependencies
from fastapi import FastAPI, UploadFile, Request, File, Form
from fastapi.responses import JSONResponse
from starlette.requests import Request
import uvicorn

#Public API Dependency
from pyngrok import ngrok

#Face Extraction Dependency
from retinaface import RetinaFace

# Face Recognition Dependencies
from deepface import DeepFace
from deepface.commons import functions, distance as dst

# Initiation of API through FastAPI
app = FastAPI()

# Create a custom ngrok configuration
config = {
    "web_addr": "http://127.0.0.1:3119/",
    "addr": "http://127.0.0.1:3119",  # Set the ngrok tunnel address
    "region": "us",
    "http_tunnel": True,  # Add this to specify an HTTP tunnel
}

# Function taken and modified from DeepFace
def find(img_path, db_path, model_name="VGG-Face", distance_metric="euclidean", enforce_detection=True, detector_backend="retinaface", align=True, normalization="base", silent=False, bid=0000):
    try:
        # Measure the start time
        tic = time.time()

        # Check if the database path exists
        if not os.path.isdir(db_path):
            raise ValueError("Passed db_path does not exist!")

        # Determine the target size for the model
        target_size = functions.find_target_size(model_name=model_name)

        # Create a file name for storing representations based on bid and model name
        file_name = f"{bid}_representations_{model_name}.pkl"
        file_name = file_name.replace("-", "_").lower()

        # Check if representations were previously stored
        if path.exists(db_path + "/" + file_name):
            if not silent:
                print(
                    f"WARNING: Representations for images in {db_path} folder were previously stored"
                    + f" in {file_name}. If you added new instances after the creation, then please "
                    + "delete this file and call find function again. It will create it again."
                )

            # Load representations from the file
            with open(f"{db_path}/{file_name}", "rb") as f:
                representations = pickle.load(f)

            if not silent:
                print("There are ", len(representations), " representations found in ", file_name)

        # Create a DataFrame from the representations
        df = pd.DataFrame(representations, columns=["identity", f"{model_name}_representation"])

        # Extract faces from the input image
        target_objs = functions.extract_faces(img=img_path, target_size=target_size, detector_backend=detector_backend, grayscale=False, enforce_detection=enforce_detection, align=align)

        # Initialize an empty list to store result DataFrames for each face in the input image
        resp_obj = []

        # Iterate over each face in the input image
        for target_img, target_region, _ in target_objs:
            # Extract the representation of the target face using the specified model
            target_embedding_obj = DeepFace.represent(img_path=target_img, model_name=model_name, enforce_detection=enforce_detection, detector_backend="skip", align=align, normalization=normalization)
            target_representation = target_embedding_obj[0]["embedding"]

            # Create a copy of the DataFrame for filtering
            result_df = df.copy()

            # Add source face region information to the DataFrame
            result_df["source_x"] = target_region["x"]
            result_df["source_y"] = target_region["y"]
            result_df["source_w"] = target_region["w"]
            result_df["source_h"] = target_region["h"]

            # Calculate distances between the target face and all database faces
            distances = []
            for index, instance in df.iterrows():
                source_representation = instance[f"{model_name}_representation"]

                # Choose the appropriate distance metric
                if distance_metric == "cosine":
                    distance = dst.findCosineDistance(source_representation, target_representation)
                elif distance_metric == "euclidean":
                    distance = dst.findEuclideanDistance(source_representation, target_representation)
                elif distance_metric == "euclidean_l2":
                    distance = dst.findEuclideanDistance(dst.l2_normalize(source_representation), dst.l2_normalize(target_representation))
                else:
                    raise ValueError(f"invalid distance metric passes - {distance_metric}")

                distances.append(distance)

            # Add distances to the DataFrame
            result_df[f"{model_name}_{distance_metric}"] = distances

            # Find and apply a distance threshold
            threshold = dst.findThreshold(model_name, distance_metric)
            result_df = result_df.drop(columns=[f"{model_name}_representation"])
            result_df = result_df[result_df[f"{model_name}_{distance_metric}"] <= threshold]
            result_df = result_df.sort_values(by=[f"{model_name}_{distance_metric}"], ascending=True).reset_index(drop=True)

            # Append the result DataFrame to the response object list
            resp_obj.append(result_df)

        # Measure the end time
        toc = time.time()

        # Print the duration of the find function if not silent
        if not silent:
            print("find function lasts ", toc - tic, " seconds")

        # Return the list of result DataFrames
        return resp_obj
    except Exception as e:
        print(e)

# The Config class is a container for various configuration parameters.
# Each parameter is a constant value that can be accessed using Config.PARAMETER_NAME.

# Explanation of parameters:
# - FRAME_FOLDER: Folder to store video frames
# - FACE_FOLDER: Folder to store extracted faces
# - MODEL_NAME: Name of the face recognition model
# - DISTANCE_METRIC: Distance metric used in face recognition
# - DETECTOR_BACKEND: Backend used for face detection
# - DIRECTORY_FOLDER: Current working directory
# - VIDEO_NAME: Name of the video file
# - UPLOADS_FOLDER: Folder to store uploaded files
# - PKL_FOLDER: Folder to store pickle files
# - INTERVAL: Time interval for some operation (e.g., in seconds)
# - PORT: Port number for server communication

# FOLDERS is a list containing the names of folders that need to be created.
# A loop iterates through each folder name, and if the folder doesn't exist, it creates it using os.makedirs().

# Overall, this class provides a centralized way to manage configuration parameters and ensures that the required folders exist.

class Config:
    # Define constant values for configuration parameters
    FRAME_FOLDER = "Video_Frames"
    FACE_FOLDER = "Faces"
    MODEL_NAME = "VGG-Face"
    DISTANCE_METRIC = "euclidean"
    DETECTOR_BACKEND = "retinaface"
    DIRECTORY_FOLDER = os.getcwd()
    VIDEO_NAME = "Clip.mp4"
    UPLOADS_FOLDER = "Uploads"
    PKL_FOLDER = "PKL"
    INTERVAL = float(1.0)
    PORT = 8000

    # Making a list of folder names.
    FOLDERS = [FRAME_FOLDER, FACE_FOLDER, PKL_FOLDER, UPLOADS_FOLDER]

    # Creating the folders if they don't exist
    for folder in FOLDERS:
        if not os.path.exists(folder):
            os.makedirs(folder)

class VideoManager:
    @staticmethod
    def extract_faces_from_frame(frame, counter):
        try:
            # Create a filename for the frame and save it
            filename1 = f"{Config.FRAME_FOLDER}/Image {counter}.jpg"
            print(filename1)
            cv2.imwrite(filename1, frame)

            # Extract faces from the saved frame using RetinaFace
            faces = RetinaFace.extract_faces(filename1, threshold=0.99, model=None, allow_upscaling=True)

            # Check if any faces are extracted
            if not faces:
                print(f"No faces extracted in Frame {counter}.")
                return None

            # Save each extracted face as a separate image
            for idx, face in enumerate(faces):
                idxx = idx + 1
                filename2 = f"Frame.{counter}.Face.{idxx}.jpg"
                print(filename2)
                result = os.path.join(Config.FACE_FOLDER, filename2)
                cv2.imwrite(result, face)

            return counter

        except Exception as e:
            print(f"Error processing Frame {counter}: {str(e)}")
            return None

    @staticmethod
    def process_video_to_faces(video_path):
        # Open the video file
        video = cv2.VideoCapture(video_path)

        # Create the frame folder if it doesn't exist
        if not os.path.exists(Config.FRAME_FOLDER):
            os.makedirs(Config.FRAME_FOLDER)

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        seconds = round(frames / fps)

        sec = 0.0
        counter = 0
        global_counter = 0
        interval = Config.INTERVAL

        # Use ThreadPoolExecutor for parallel face extraction
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []

            # Iterate through video frames at specified intervals
            while sec < seconds:
                t_msec = int(1000 * sec)
                video.set(cv2.CAP_PROP_POS_MSEC, t_msec)
                ret, frame = video.read()

                if ret:
                    # Submit face extraction task to the ThreadPoolExecutor
                    futures.append(executor.submit(VideoManager.extract_faces_from_frame, frame.copy(), counter))
                    sec += interval
                    counter += 1
                    global_counter += 1
                else:
                    sec += interval
                    global_counter += 1

            # Wait for all face extraction tasks to complete
            concurrent.futures.wait(futures)

        # Release the video and close all windows
        video.release()
        cv2.destroyAllWindows()

class FaceRecog:
    # Initialize a global list to store identification data
    global idds
    idds = []

    def get_id_data(self, img_path, bid):
        # Print a message indicating the image being processed
        print(f"Studying image: {img_path}")

        # Perform face recognition using the 'find' function
        finds = find(img_path, db_path=os.path.join(Config.DIRECTORY_FOLDER, Config.PKL_FOLDER), model_name=Config.MODEL_NAME, distance_metric=Config.DISTANCE_METRIC, detector_backend=Config.DETECTOR_BACKEND, enforce_detection=False, align=True, normalization="base", silent=True, bid=bid)

        # Extract identification information from the result
        ids = (img_path, finds[0]['identity'][0].split('/')[1].split('.')[1])

        # Sort and store the identification data
        self.sort_ids(ids)

        # Return the identified person's ID
        return finds[0]['identity'][0].split('/')[1].split('.')[1]

    @staticmethod
    def sort_ids(ids):
        # Access the global idds list and append the new identification data
        global idds
        idds.append(ids)

    def get_ids_from_faces(self, face_dir, bid):
        ids = []
        faces = [os.path.join(face_dir, a) for a in os.listdir(face_dir)]

        # Use ThreadPoolExecutor for parallel face recognition
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []

            # Submit face recognition tasks for each face in the directory
            for face in faces:
                future = executor.submit(FaceRecog.get_id_data, self, face, bid)
                futures.append(future)

            # Wait for all face recognition tasks to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    id = future.result()
                    ids.append(id)
                except Exception as e:
                    print(f"An error occurred: {str(e)}")

        # Access the global idds list and return the identified IDs along with stored identification data
        global idds
        return ids, idds

class Cleanup:
    @staticmethod
    def clean():
        # Define directories to be cleaned
        trash = [os.path.join(Config.DIRECTORY_FOLDER, a) for a in [Config.FRAME_FOLDER, Config.FACE_FOLDER]]

        try:
            # Iterate through directories and delete them along with their contents
            for directory_path in trash:
                shutil.rmtree(directory_path)
                print(f'Directory "{directory_path}" and its contents have been successfully deleted.')
        except Exception as e:
            # Handle exceptions and print an error message
            print(f'An error occurred: {e}')

# Explanation of the Cleanup class and clean method:
# The Cleanup class provides a static method 'clean' to delete specified directories and their contents.
# The 'trash' list contains the paths of directories to be cleaned, such as frames and faces folders.
# The code then attempts to delete each directory using shutil.rmtree(), which removes the entire directory and its contents.
# If an exception occurs during the cleanup process, an error message is printed.
# The class can be used to perform cleanup operations in a convenient and centralized manner.

class Checkup:
    def check_PKL(self):
        files = os.listdir(os.path.join(Config.DIRECTORY_FOLDER, Config.PKL_FOLDER))
        
        if len(files) == 0:
            return "PKL not found."
        
        for a in files:
            if "PKL" in a:
                return "PKL not found."
            else:
                return 123

def process_video(video_file: UploadFile, bid: int):
    try:
        # Save the uploaded video to a temporary directory
        video_path = os.path.join(Config.UPLOADS_FOLDER, Config.VIDEO_NAME)

        # Cleanup or delete the temporary video file (if it already exists)
        if os.path.exists(video_path):
            os.remove(video_path)

        # Write the contents of the uploaded video file to the temporary file
        with open(video_path, "wb") as f:
            f.write(video_file.file.read())

        # Initialize VideoManager and FaceRecog instances
        video_manager = VideoManager()
        face_recog = FaceRecog()
        check = Checkup()

        #Checking if PKL file exists. If it does not, we cannot proceed.
        if "PKL not found" in check.check_PKL():
            temp = {
                "Error" : "PKL file not found"
            }
            return JSONResponse(content=temp)

        # Process the video to extract faces
        video_manager.process_video_to_faces(video_path)

        # Get IDs from the faces and the stored identification data
        ids, idds = face_recog.get_ids_from_faces(Config.FACE_FOLDER, bid)

        # Create an empty dictionary to store the counts
        count_dict = {}

        # Loop through the list of IDs and update the counts in the dictionary
        for num in ids:
            if num in count_dict:
                count_dict[num] += 1
            else:
                count_dict[num] = 1

        # Create a dictionary to return as the JSON response
        response_dict = {
            "ids": ids,
            "idds": idds,
            "counts": count_dict,
        }

        # Cleanup or delete the temporary video file
        if os.path.exists(video_path):
            os.remove(video_path)

        # Return a JSON response with the processed data
        return JSONResponse(content=response_dict)
    except Exception as e:
        # Handle exceptions and return an error response
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Define an endpoint to process video files
@app.post("/process_video")
async def process_video_endpoint(bid: int = Form(...), video_file: UploadFile = File(...)):
    return process_video(video_file, bid)

# Define a middleware to remove ngrok browser warning
@app.middleware("http")
async def remove_ngrok_warning(request: Request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response

# Main block to run the FastAPI application with ngrok tunneling
if __name__ == "__main__":
    # Connect to ngrok and get the public URL
    public_url = ngrok.connect(Config.PORT, "http")
    print(" * ngrok tunnel \"{}\" -> http://127.0.0.1:{}/".format(public_url, Config.PORT))
    
    # Run the FastAPI application using uvicorn
    uvicorn.run("RecogAPI:app", port=Config.PORT, reload=True)