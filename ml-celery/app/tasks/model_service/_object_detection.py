from email.mime import image
import logging
import re
from zipfile import ZipFile
from celery import shared_task
from sympy import false, use
from tqdm import tqdm
from mq_main import redis
from time import perf_counter
import gdown
from .image_classify.autogluon_trainer import AutogluonTrainer
import uuid
from autogluon.multimodal import MultiModalPredictor
import joblib
from settings.config import TEMP_DIR


from utils.dataset_utils import (
    find_latest_model,
    split_data,
    create_csv,
    remove_folders_except,
    create_folder,
    download_dataset,
    download_dataset,
)
import os
from pathlib import Path


def train(task_id: str, request: dict):
    print("task_id:", task_id)
    print("request:", request)
    print("Object Detection Training request received")
    start = perf_counter()
    # request["training_argument"]["ag_fit_args"]["time_limit"] = request["training_time"]
    try:
        user_dataset_path = (
            f"{TEMP_DIR}/{request['userEmail']}/{request['projectName']}/dataset"
        )
        user_model_path = f"{TEMP_DIR}/{request['userEmail']}/{request['projectName']}/trained_models/{request['runName']}/{uuid.uuid4()}"
        # if os.path.exists(user_dataset_path) == False:
        #     user_dataset_path = download_dataset(
        #         user_dataset_path,
        #         True,
        #         request["dataset_url"],
        #         request["dataset_download_method"],
        #    )

        print("downloading dataset")
        user_dataset_path = download_dataset(
            user_dataset_path,
            True,
            request["dataset_url"],
            request["dataset_download_method"],
        )
        download_end = perf_counter()
        print("Download dataset successfully ", user_dataset_path)

        #! temporary, train and val should be split, file name should be changed
        data_dir = Path(user_dataset_path)
        train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
        test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")

        preset = "medium_quality"

        # training job của mình sẽ chạy ở đây
        predictor = MultiModalPredictor(
            problem_type="object_detection",
            sample_data_path=train_path,
            path=user_model_path,
        )
        logging.basicConfig(level=logging.DEBUG)

        predictor.fit(
            train_path,
            time_limit=request["training_time"],
            presets=preset,
            save_path=user_model_path,
        )

        print("Training model successfully")

        metrics = predictor.evaluate(test_path)
        print("Evaluate model successfully")

        end = perf_counter()
        return {
            "metrics": metrics,
            "training_evaluation_time": end - start,
            "model_download_time": download_end - start,
            "saved_model_path": user_model_path,
        }

    except Exception as e:
        print(e)
        # raise HTTPException(status_code=500, detail=f"Error in downloading or extracting folder: {str(e)}")
    finally:
        # if os.path.exists(temp_dataset_path):
        #    os.remove(temp_dataset_path)
        return {}


memory = joblib.Memory(
    f"{TEMP_DIR}", verbose=0, mmap_mode="r", bytes_limit=1024 * 1024 * 1024 * 100
)


@memory.cache
def load_model_from_path(model_path: str) -> MultiModalPredictor:
    return MultiModalPredictor.load(model_path)


async def load_model(
    user_name: str, project_name: str, run_name: str
) -> MultiModalPredictor:
    model_path = find_latest_model(
        f"{TEMP_DIR}/{user_name}/{project_name}/trained_models/{run_name}"
    )
    return load_model_from_path(model_path)


async def predict(task_id: str, request: dict):
    userEmail = request["userEmail"]
    projectName = request["projectName"]
    runName = request["runName"]
    image = request["image"]
    print(userEmail)
    print("Run Name:", runName)
    try:
        # write the image to a temporary file
        temp_image_path = f"{TEMP_DIR}/{userEmail}/{projectName}/temp.jpg"
        os.makedirs(Path(temp_image_path).parent, exist_ok=True)
        with open(temp_image_path, "wb") as buffer:
            buffer.write(await image.read())

        start_load = perf_counter()
        # TODO : Load model with any path
        model = await load_model(userEmail, projectName, runName)
        load_time = perf_counter() - start_load
        inference_start = perf_counter()
        predictions = model.predict(temp_image_path, realtime=True, save_results=True)
        proba: float = 0.98

        return {
            "status": "success",
            "message": "Prediction completed",
            "load_time": load_time,
            "proba": proba,
            "inference_time": perf_counter() - inference_start,
            "predictions": str(predictions),
        }
    except Exception as e:
        print(e)
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)