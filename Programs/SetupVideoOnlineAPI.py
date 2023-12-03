import os
import pickle
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from os import path
from tqdm import tqdm
from fastapi import FastAPI, HTTPException, UploadFile, File
from retinaface import RetinaFace
from deepface import DeepFace
import deepface.commons.functions as functions
from pyngrok import ngrok

# 3rd party dependencies

# package dependencies
from deepface.commons import functions

import uvicorn

# Set your Ngrok auth token here
ngrok.set_auth_token("2VQ3N6dnV7bMo5PhJskNw6KOynM_43X2AJPHbCN42wdjt29jb")

# Constants
FACE_MAX_WORKERS = 14
PKL_MAX_WORKERS = 12
DIRECTORY = os.getcwd()
RECORDED_FOLDER = "Recorded"
FACE_FOLDER = "Data"
PROCESSING_FOLDER = "Processing"
FACE_THRESHOLD = 0.99
FPS = 30
CAMERA_INDEX = 0
MODEL_NAME = "VGG-Face"
PKL_FOLDER = "PKL"
UPLOADS_FOLDER = "Uploads"

folders = [FACE_FOLDER, RECORDED_FOLDER, PROCESSING_FOLDER, PKL_FOLDER]

# Create output directories if they don't exist
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

app = FastAPI()

class VideoManager:

    def generate_filename(self, count):
        return f'Frame {count:02d}.jpg'

    def process_video(self, videoname):
        
        video = cv2.VideoCapture(videoname)
        fps = video.get(cv2.CAP_PROP_FPS)
        frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        seconds = round(frames / fps)
        
        seperation = seconds/100
        
        sec = 0
        count = 0
        while sec < seconds:
            t_msec = int(1000 * sec)
            video.set(cv2.CAP_PROP_POS_MSEC, t_msec)
            ret, frame = video.read()
            filename1 = os.path.join(RECORDED_FOLDER, self.generate_filename(count))
            cv2.imwrite(filename1, frame)
            count += 1
            sec += seperation
        
        video.release() 

# Define a class for face extraction
class FaceExtractor:
    def process_frame(self, frame, id, bid):
        try:
            count = frame.split('\\')[1].split(' ')[1].split('.')[0]

            # Use RetinaFace to extract faces from the frame
            faces = RetinaFace.extract_faces(frame, threshold=FACE_THRESHOLD, model=None, allow_upscaling=True)

            if not faces:
                print(f"No faces extracted in Frame {count}.")
                return None

            path0 = os.path.join(DIRECTORY, PROCESSING_FOLDER)
            if not os.path.exists(path0):
                os.makedirs(path0)

            image_sizes = [np.prod(image.shape) for image in faces]
            largest_index = np.argmax(image_sizes)
            filename2 = f"user.{id}.{count}.jpg"
            result = os.path.join(path0, filename2)
            print(f"user.{id}.{count}.jpg written")
            cv2.imwrite(result, faces[largest_index])

            return count

        except Exception as e:
            print(str(e))
            return None

    def video_to_faces_parallel(self, frames, id):
        with ThreadPoolExecutor(max_workers=FACE_MAX_WORKERS) as executor:
            futures = [executor.submit(self.process_frame, frame, id) for frame in frames]
            for future in futures:
                future.result()

class Cleanup:
    @staticmethod
    def clean():
        directory_path1 = os.path.join(DIRECTORY, RECORDED_FOLDER)
        directory_path2 = os.path.join(DIRECTORY, PROCESSING_FOLDER)
        try:
            shutil.rmtree(directory_path1)
            print(f'Directory "{directory_path1}" and its contents have been successfully deleted.')
            shutil.rmtree(directory_path2)
            print(f'Directory "{directory_path2}" and its contents have been successfully deleted.')
        except Exception as e:
            print(f'An error occurred: {e}')

    @staticmethod
    def move():
        old_fold = os.path.join(DIRECTORY, PROCESSING_FOLDER)
        new_fold = os.path.join(DIRECTORY, FACE_FOLDER)

        if not os.path.exists(FACE_FOLDER):
            os.makedirs(FACE_FOLDER)

        for filename in os.listdir(old_fold):
            old_name = os.path.join(old_fold, filename)
            new_name = os.path.join(new_fold, filename)
            if not os.path.exists(new_name):
                os.rename(old_name, new_name)
            else:
                os.remove(new_name)
                os.rename(old_name, new_name)

        print("Moved!!")

class Deep:
    @staticmethod
    def represent(img_path, model_name="VGG-Face", enforce_detection=True, detector_backend="retinaface", align=True,
                  normalization="base"):
        resp_objs = []

        model = DeepFace.build_model(model_name)

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
        tic = time.time()
        db_path = PROCESSING_FOLDER
        file_name = f"representations_{model_name}.pkl"
        file_name = file_name.replace("-", "_").lower()
        pkl = f"{PKL_FOLDER}/{bid}_{file_name}"
        if os.path.isdir(db_path) is not True:
            raise ValueError("Passed db_path does not exist!")
        target_size = functions.find_target_size(model_name=model_name)
        employees = []
        for r, _, f in os.walk(db_path):
            for file in f:
                if ((".jpg" in file.lower()) or (".jpeg" in file.lower()) or (".png" in file.lower())):
                    exact_path = r + "/" + file
                    employees.append(exact_path)
        if len(employees) == 0:
            raise ValueError("There is no image in ", db_path, " folder! Validate .jpg or .png files exist in this path.")
        representations = []
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
                instance = []
                instance.append(employee)
                instance.append(img_representation)
                representations.append(instance)
        if path.exists(pkl):
            with open(pkl, "rb") as f:
                representations_present = pickle.load(f)
                representations.extend(representations_present)
            backup_pkl = pkl.split('.')[0] + "_Backup.pkl"
            os.rename(pkl, backup_pkl)
        with open(pkl, "wb") as f:
            pickle.dump(representations, f)
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
        
        # Initialize the Class Object
        recorder = VideoManager()
        face_extractor = FaceExtractor()
        deep_processor = Deep()

        # Record the video and get frames
        recorder.process_video(filename = file_path)

        frames = list(os.path.join(RECORDED_FOLDER, a) for a in os.listdir(RECORDED_FOLDER))

        # Process frames in parallel to extract faces
        face_extractor.video_to_faces_parallel(frames = frames, id = id, bid = bid)

        deep_processor.Create_PKL(bid=bid)

        # Moving and cleaning
        Cleanup.move()
        Cleanup.clean()

        return "Done!"

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    public_url = ngrok.connect(8000, "http")
    print(" * ngrok tunnel \"{}\" -> http://127.0.0.1:{}/".format(public_url, 8000))
    uvicorn.run("SetupAPIVideo:app", port=8000, reload=True)