from cProfile import label
from pathlib import Path
from urllib import request
from celery import shared_task
import uuid
import time
from .tabular.autogluon_trainer import AutogluonTrainer
from mq_main import redis
from time import perf_counter
import os
import uuid
import joblib
from settings.config import TEMP_DIR
import gdown

from utils.dataset_utils import (
    download_dataset2,
    find_latest_model,
    split_data,
    create_csv,
    remove_folders_except,
    create_folder,
    download_dataset,
)

from autogluon.tabular import TabularPredictor

def train(task_id: str, request: dict):
    print("Tabular Training request received")
    temp_dataset_path = ""
    start = perf_counter()
    print("Training request received")
    # request["training_argument"]["ag_fit_args"]["time_limit"] = request["training_time"]
    # request["training_argument"]["ag_fit_args"]["presets"] = request["presets"]
    try:
        user_dataset_path = (
            f"{TEMP_DIR}/{request['userEmail']}/{request['projectName']}/dataset/"
        )
        os.makedirs(user_dataset_path, exist_ok=True)
        user_model_path = f"{TEMP_DIR}/{request['userEmail']}/{request['projectName']}/trained_models/{request['runName']}/{task_id}"

        # TODO: download dataset in this function
        train_df = download_dataset2(
            user_dataset_path,
            request["dataset_url"],
            request["dataset_download_method"],
        )
        # train_df, test_df = train_test_split(train_df, test_size=0.2)

        presets = request["presets"]

        # # training job của mình sẽ chạy ở đây
        predictor = TabularPredictor(
            label="label",
            path=user_model_path,
        )

        predictor.fit(
            train_data=train_df,
            # tuning_data=val_data,
            time_limit=request["training_time"],
            presets=presets,
            save_path=user_model_path,
        )

        # metrics = predictor.evaluate(test_data, metrics=request["metrics"])
        # print("Training model successfully")

        end = perf_counter()

        return {
            "metrics": "temp_metrics",
            "training_evaluation_time": end - start,
            "saved_model_path": user_model_path,
        }

    except Exception as e:
        print(e)
        # raise HTTPException(status_code=500, detail=f"Error in downloading or extracting folder: {str(e)}")
    finally:
        if os.path.exists(temp_dataset_path):
            os.remove(temp_dataset_path)
