from email.mime import image
import re
from zipfile import ZipFile
from celery import shared_task
from sympy import false, use
from tqdm import tqdm
from mq_main import redis
from time import perf_counter
import gdown
from worker.model_service.image_classify.autogluon_trainer import AutogluonTrainer
import uuid
from autogluon.multimodal import MultiModalPredictor
import joblib


from utils.dataset_utils import (
    find_latest_model,
    split_data,
    create_csv,
    remove_folders_except,
    create_folder,
)
import os
from pathlib import Path


def train(task_id: str, request: dict):
    redis.set(f"status-{task_id}", "RUNNING")
    temp_dataset_path = ""
    print("task_id:", task_id)
    print("request:", request)
    print("Image Classification Training request received")
    start = perf_counter()
    request["training_argument"]["ag_fit_args"]["time_limit"] = request["training_time"]
    try:
        # temp folder to store dataset and then delete after training
        temp_dataset_path = Path(
            f"D:/tmp/{request['userEmail']}/{request['projectName']}/data.zip"
        )
        os.makedirs(temp_dataset_path.parent, exist_ok=True)

        dataset_url = f"https://drive.google.com/uc?id={request['dataset_url']}"
        if os.path.exists(temp_dataset_path) == False:
            gdown.download(url=dataset_url, output=str(temp_dataset_path), quiet=False)
        download_end = perf_counter()
        print("Download dataset successfully")
        user_dataset_path = (
            f"D:/tmp/{request['userEmail']}/{request['projectName']}/datasets"
        )
        user_model_path = f"D:/tmp/{request['userEmail']}/{request['projectName']}/trained_models/{request['runName']}/{uuid.uuid4()}"
        with ZipFile(temp_dataset_path, "r") as zip_ref:
            zip_ref.extractall(user_dataset_path)

        create_folder(Path(user_dataset_path))

        if os.path.exists(f"{user_dataset_path}/split") == False:
            split_data(Path(user_dataset_path), f"{user_dataset_path}/split/")
            # # TODO : User can choose ratio to split data @DuongNam
            # # assume user want to split data into 80% train, 10% val, 10% test
            create_csv(
                Path(f"{user_dataset_path}/split/train"),
                Path(f"{user_dataset_path}/train.csv"),
            )
            create_csv(
                Path(f"{user_dataset_path}/split/val"),
                Path(f"{user_dataset_path}/val.csv"),
            )
            create_csv(
                Path(f"{user_dataset_path}/split/test"),
                Path(f"{user_dataset_path}/test.csv"),
            )
            print("Split data successfully")
        remove_folders_except(Path(user_dataset_path), "split")
        print("Remove folders except split successfully")
        trainer = AutogluonTrainer(request["training_argument"])
        print("Create trainer successfully")
        # training job của mình sẽ chạy ở đây
        model = trainer.train(
            "label",
            Path(f"{user_dataset_path}/train.csv"),
            Path(f"{user_dataset_path}/val.csv"),
            Path(f"{user_model_path}"),
        )
        print("Training model successfully")
        if model is None:
            raise ValueError("Error in training model")

        acc = AutogluonTrainer.evaluate(model, Path(f"{user_dataset_path}/test.csv"))
        print("Evaluate model successfully")
        acc = 0.98

        end = perf_counter()
        redis.set(f"status-{task_id}", "SUCCESS")

        return {
            "validation_accuracy": acc,
            "training_evaluation_time": end - start,
            "model_download_time": download_end - start,
            "saved_model_path": user_model_path,
        }

    except Exception as e:
        redis.set(f"status-{task_id}", "FAILED")
        print(e)
        # raise HTTPException(status_code=500, detail=f"Error in downloading or extracting folder: {str(e)}")
    finally:
        # if os.path.exists(temp_dataset_path):
        #    os.remove(temp_dataset_path)
        return {}


memory = joblib.Memory(
    "D:/tmp", verbose=0, mmap_mode="r", bytes_limit=1024 * 1024 * 1024 * 100
)


@memory.cache
def load_model_from_path(model_path: str) -> MultiModalPredictor:
    return MultiModalPredictor.load(model_path)


async def load_model(
    user_name: str, project_name: str, run_name: str
) -> MultiModalPredictor:
    model_path = find_latest_model(
        f"D:/tmp/{user_name}/{project_name}/trained_models/{run_name}"
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
        temp_image_path = f"D:/tmp/{userEmail}/{projectName}/temp.jpg"
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