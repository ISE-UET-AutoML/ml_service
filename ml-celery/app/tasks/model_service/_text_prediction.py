from email.mime import image
import logging
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

from utils import get_storage_client
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
    temp_dataset_path = ""
    print("task_id:", task_id)
    print("request:", request)
    print("Text Prediction Training request received")
    start = perf_counter()
    # request["training_argument"]["ag_fit_args"]["time_limit"] = request["training_time"]
    # request["training_argument"]["ag_fit_args"]["presets"] = request["presets"]
    try:
        user_dataset_path = (
            f"{TEMP_DIR}/{request['userEmail']}/{request['projectName']}/dataset/"
        )
        os.makedirs(user_dataset_path, exist_ok=True)
        user_model_path = f"{TEMP_DIR}/{request['userEmail']}/{request['projectName']}/trained_models/{request['runName']}/{task_id}"

        if request["dataset_download_method"] == "gdrive":
            user_dataset_path = download_dataset(
                user_dataset_path,
                True,
                request["dataset_url"],
                request["dataset_download_method"],
            )
        elif request["dataset_download_method"] == "gcloud":
            print(
                request["gcloud_dataset_bucketname"],
                request["gcloud_dataset_directory"],
            )
            get_storage_client().download_folder(
                request["gcloud_dataset_bucketname"],
                request["gcloud_dataset_directory"],
                user_dataset_path,
            )
            user_dataset_path = (
                f"{user_dataset_path}/{request['gcloud_dataset_directory']}"
            )

        train_path = f"{user_dataset_path}/data.csv"

        presets = request["presets"]

        # training job của mình sẽ chạy ở đây
        predictor = MultiModalPredictor(
            sample_data_path=train_path,
            path=user_model_path,
        )
        logging.basicConfig(level=logging.DEBUG)

        print("created predictor")

        predictor.fit(
            train_path,
            time_limit=request["training_time"],
            presets=presets,
            save_path=user_model_path,
        )

        print(predictor.eval_metric)

        print("Training model successfully")

        end = perf_counter()

        return {
            "validation_accuracy": acc,
            "training_evaluation_time": end - start,
            "saved_model_path": user_model_path,
        }

    except Exception as e:
        print(e)
        # raise HTTPException(status_code=500, detail=f"Error in downloading or extracting folder: {str(e)}")
    # finally:
    # if os.path.exists(temp_dataset_path):
    #    os.remove(temp_dataset_path)
    # return {}
