from email.mime import image
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

from utils import get_storage_client
from utils.dataset_utils import (
    find_latest_model,
    split_data,
    create_csv,
    remove_folders_except,
    create_folder,
    download_dataset,
)
from utils.train_utils import find_in_current_dir
import os
from pathlib import Path


def train(task_id: str, request: dict):
    print("task_id:", task_id)
    print("request:", request)
    print("MultiModal Training request received")
    start = perf_counter()
    request["training_argument"]["ag_fit_args"]["time_limit"] = request["training_time"]
    request["training_argument"]["ag_fit_args"]["presets"] = request["presets"]
    try:
        user_dataset_path = (
            f"{TEMP_DIR}/{request['userEmail']}/{request['projectName']}/dataset/"
        )
        os.makedirs(user_dataset_path, exist_ok=True)
        user_model_path = f"{TEMP_DIR}/{request['userEmail']}/{request['projectName']}/trained_models/{request['runName']}/{task_id}"

        # TODO: download dataset in this function
        user_dataset_path = download_dataset(
            user_dataset_path, True, request, request["dataset_download_method"]
        )

        train_path = find_in_current_dir(
            "train", user_dataset_path, is_pattern=True, extension=".csv"
        )
        val_path = find_in_current_dir(
            "val", user_dataset_path, is_pattern=True, extension=".csv"
        )
        if val_path == train_path:
            val_path = None
        test_path = find_in_current_dir(
            "test", user_dataset_path, is_pattern=True, extension=".csv"
        )
        train_path = f"{user_dataset_path}/{train_path}"
        if val_path is not None:
            val_path = f"{user_dataset_path}/{val_path}"
        test_path = f"{user_dataset_path}/{test_path}"

        presets = request["presets"]

        # # training job của mình sẽ chạy ở đây
        predictor = MultiModalPredictor(
            label=request["label_column"],
            sample_data_path=train_path,
            path=user_model_path,
        )

        predictor.fit(
            train_data=train_path,
            tuning_data=val_path,
            time_limit=request["training_time"],
            presets=presets,
            save_path=user_model_path,
        )

        metrics = predictor.evaluate(test_path)
        # print("Training model successfully")

        end = perf_counter()

        return {
            "metrics": metrics,
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
