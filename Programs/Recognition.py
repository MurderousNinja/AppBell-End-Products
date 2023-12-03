import os
import shutil
import concurrent.futures
import cv2
import time
import pickle
from os import path

from fastapi import FastAPI, UploadFile, Request, Response, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from retinaface import RetinaFace
from starlette.requests import Request
from pyngrok import ngrok
from deepface import DeepFace

# 3rd party dependencies
import pandas as pd

# package dependencies
from deepface.commons import functions, distance as dst

def find(img_path, db_path, model_name="VGG-Face", distance_metric="euclidean", enforce_detection=True, detector_backend="retinaface", align=True, normalization="base", silent=False, bid = 0000):

    tic = time.time()

    # -------------------------------
    if os.path.isdir(db_path) is not True:
        raise ValueError("Passed db_path does not exist!")

    target_size = functions.find_target_size(model_name=model_name)

    # ---------------------------------------

    file_name = f"{bid}_representations_{model_name}.pkl"
    file_name = file_name.replace("-", "_").lower()

    if path.exists(db_path + "/" + file_name):

        if not silent:
            print(
                f"WARNING: Representations for images in {db_path} folder were previously stored"
                + f" in {file_name}. If you added new instances after the creation, then please "
                + "delete this file and call find function again. It will create it again."
           )

        with open(f"{db_path}/{file_name}", "rb") as f:
            representations = pickle.load(f)

        if not silent:
            print("There are ", len(representations), " representations found in ", file_name)
    
    
    df = pd.DataFrame(representations, columns=["identity", f"{model_name}_representation"])

    # img path might have more than once face
    target_objs = functions.extract_faces(img=img_path, target_size=target_size, detector_backend=detector_backend, grayscale=False, enforce_detection=enforce_detection, align=align)

    resp_obj = []

    for target_img, target_region, _ in target_objs:
        target_embedding_obj = DeepFace.represent(img_path=target_img, model_name=model_name, enforce_detection=enforce_detection, detector_backend="skip", align=align, normalization=normalization)

        target_representation = target_embedding_obj[0]["embedding"]

        result_df = df.copy()  # df will be filtered in each img
        result_df["source_x"] = target_region["x"]
        result_df["source_y"] = target_region["y"]
        result_df["source_w"] = target_region["w"]
        result_df["source_h"] = target_region["h"]

        distances = []
        for index, instance in df.iterrows():
            source_representation = instance[f"{model_name}_representation"]

            if distance_metric == "cosine":
                distance = dst.findCosineDistance(source_representation, target_representation)
            elif distance_metric == "euclidean":
                distance = dst.findEuclideanDistance(source_representation, target_representation)
            elif distance_metric == "euclidean_l2":
                distance = dst.findEuclideanDistance(dst.l2_normalize(source_representation), dst.l2_normalize(target_representation))
            else:
                raise ValueError(f"invalid distance metric passes - {distance_metric}")

            distances.append(distance)

            # ---------------------------

        result_df[f"{model_name}_{distance_metric}"] = distances

        threshold = dst.findThreshold(model_name, distance_metric)
        result_df = result_df.drop(columns=[f"{model_name}_representation"])
        result_df = result_df[result_df[f"{model_name}_{distance_metric}"] <= threshold]
        result_df = result_df.sort_values(by=[f"{model_name}_{distance_metric}"], ascending=True).reset_index(drop=True)

        resp_obj.append(result_df)

    # -----------------------------------

    toc = time.time()

    if not silent:
        print("find function lasts ", toc - tic, " seconds")

    return resp_obj

class Config:
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

    # Create folders if they don't exist
    FOLDERS = [FRAME_FOLDER, FACE_FOLDER, PKL_FOLDER, UPLOADS_FOLDER]

    for folder in FOLDERS:
        if not os.path.exists(folder):
            os.makedirs(folder)

class VideoManager:
    @staticmethod
    def extract_faces_from_frame(frame, counter):
        try:
            filename1 = f"{Config.FRAME_FOLDER}/Image {counter}.jpg"
            print(filename1)
            cv2.imwrite(filename1, frame)

            faces = RetinaFace.extract_faces(filename1, threshold=0.99, model=None, allow_upscaling=True)

            if not faces:
                print(f"No faces extracted in Frame {counter}.")
                return None

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
        video = cv2.VideoCapture(video_path)

        if not os.path.exists(Config.FRAME_FOLDER):
            os.makedirs(Config.FRAME_FOLDER)

        fps = video.get(cv2.CAP_PROP_FPS)
        frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        seconds = round(frames / fps)

        sec = 0.0
        counter = 0
        global_counter = 0
        interval = Config.INTERVAL

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []

            while sec < seconds:
                t_msec = int(1000 * sec)
                video.set(cv2.CAP_PROP_POS_MSEC, t_msec)
                ret, frame = video.read()

                if ret:
                    futures.append(executor.submit(VideoManager.extract_faces_from_frame, frame.copy(), counter))
                    sec += interval
                    counter += 1
                    global_counter += 1
                else:
                    sec += interval
                    global_counter += 1

            concurrent.futures.wait(futures)

        video.release()
        cv2.destroyAllWindows()

class FaceRecog:
    global idds
    idds = []
    
    def get_id_data(self, img_path, bid):
        print(f"Studing image: {img_path}")

        finds = find(img_path, db_path=os.path.join(Config.DIRECTORY_FOLDER, Config.PKL_FOLDER), model_name=Config.MODEL_NAME, distance_metric=Config.DISTANCE_METRIC, detector_backend=Config.DETECTOR_BACKEND, enforce_detection=False, align=True, normalization="base", silent=True, bid = bid)

        ids = (img_path, finds[0]['identity'][0].split('/')[1].split('.')[1])

        self.sort_ids(ids)

        return finds[0]['identity'][0].split('/')[1].split('.')[1]
    
    @staticmethod
    def sort_ids(ids):
        global idds
        idds.append(ids)

    def get_ids_from_faces(self, face_dir, bid):
        ids = []
        faces = [os.path.join(face_dir, a) for a in os.listdir(face_dir)]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            
            for face in faces:
                future = executor.submit(FaceRecog.get_id_data, self, face, bid)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    id = future.result()
                    ids.append(id)
                    # print(id)
                except Exception as e:
                    print(f"An error occurred: {str(e)}")

        global idds

        return ids, idds

class Cleanup:
    @staticmethod
    def clean():

        trash = (os.path.join(Config.DIRECTORY_FOLDER,a) for a in [Config.FRAME_FOLDER, Config.FACE_FOLDER,]) # Config.UPLOADS_FOLDER])


        try:
            for directory_path in trash:
                shutil.rmtree(directory_path)
                print(f'Directory "{directory_path}" and its contents have been successfully deleted.')
        except Exception as e:
            print(f'An error occurred: {e}')

def process_video(video_file: UploadFile, bid: int):
    # Save the uploaded video to a temporary directory
    video_path = os.path.join(Config.UPLOADS_FOLDER, Config.VIDEO_NAME)

    # Cleanup or delete the temporary video file (if needed)
    if os.path.exists(video_path):
        os.remove(video_path)

    with open(video_path, "wb") as f:
        f.write(video_file.file.read())

    fac = VideoManager()
    idd = FaceRecog()

    fac.process_video_to_faces(os.path.join(Config.UPLOADS_FOLDER, Config.VIDEO_NAME))

    ids, idds = idd.get_ids_from_faces(Config.FACE_FOLDER, bid)

    # Create an empty dictionary to store the counts
    count_dict = {}

    # Loop through the list and update the counts in the dictionary
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

    # Cleanup or delete the temporary video file (if needed)
    if os.path.exists(video_path):
        os.remove(video_path)

    return response_dict

if __name__ == "__main__":
    bid = input("Enter your Buisness ID: ")
    process_video(bid)
