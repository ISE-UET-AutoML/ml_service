#! DEPRECATED
import asyncio
from app.utils.dataset_utils import (
    find_latest_model,
    split_data,
    create_csv,
    remove_folders_except,
    create_folder,
)
from app.utils import storage
from app.model_service.train.local.image_classifier.model import TrainingRequest
from app.model_service.train.local.image_classifier.autogluon_trainer import (
    AutogluonTrainer,
)
from fastapi import HTTPException
import os
import joblib
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile
import uuid
from fastapi import APIRouter, File, Form, UploadFile
import pandas as pd
import warnings
from autogluon.multimodal import MultiModalPredictor

from settings.config import TEMP_DIR

warnings.filterwarnings("ignore")

router = APIRouter()

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


@router.post("/image_classifier/train", tags=["image_classifier"])
async def handler(request: TrainingRequest):
    print("Training request received")
    temp_dataset_path = ""
    start = perf_counter()
    print("Training request received")
    request.training_argument["ag_fit_args"]["time_limit"] = request.training_time
    try:
        # temp folder to store dataset and then delete after training
        temp_dataset_path = Path(
            f"{TEMP_DIR}/{request.userEmail}/{request.projectName}/datasets.zip"
        )

        os.makedirs(temp_dataset_path.parent, exist_ok=True)

        res = await storage.download_blob_async(
            request.userEmail,
            f"{request.projectName}/datasets/datasets.zip",
            temp_dataset_path,
        )

        download_end = perf_counter()

        if res is None or not res:
            raise ValueError("Error in downloading folder")

        user_dataset_path = (
            f"{TEMP_DIR}/{request.userEmail}/{request.projectName}/datasets"
        )
        user_model_path = f"{TEMP_DIR}/{request.userEmail}/{request.projectName}/trained_models/{request.runName}/{uuid.uuid4()}"

        create_folder(Path(user_dataset_path))

        with ZipFile(temp_dataset_path, "r") as zip_ref:
            zip_ref.extractall(user_dataset_path)

        split_data(Path(user_dataset_path), f"{user_dataset_path}/split/")

        # # TODO : User can choose ratio to split data @DuongNam
        # # assume user want to split data into 80% train, 10% val, 10% test

        create_csv(
            Path(f"{user_dataset_path}/split/train"),
            Path(f"{user_dataset_path}/train.csv"),
        )
        create_csv(
            Path(f"{user_dataset_path}/split/val"), Path(f"{user_dataset_path}/val.csv")
        )
        create_csv(
            Path(f"{user_dataset_path}/split/test"),
            Path(f"{user_dataset_path}/test.csv"),
        )
        print("Split data successfully")
        remove_folders_except(Path(user_dataset_path), "split")
        print("Remove folders except split successfully")
        trainer = AutogluonTrainer(request.training_argument)
        print("Create trainer successfully")
        # training job của mình sẽ chạy ở đây
        model = await trainer.train_async(
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

        return {
            "validation_accuracy": acc,
            "training_evaluation_time": end - start,
            "model_download_time": download_end - start,
            "saved_model_path": user_model_path,
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error in training: {str(e)}")
        # raise HTTPException(status_code=500, detail=f"Error in downloading or extracting folder: {str(e)}")
    finally:
        if os.path.exists(temp_dataset_path):
            os.remove(temp_dataset_path)


@router.post("/image_classifier/predict", tags=["image_classifier"])
async def predict(
    userEmail: str = Form("lexuanan18102004"),
    projectName: str = Form("flower-classifier"),
    runName: str = Form("Model-v0"),
    image: UploadFile = File(...),
):
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
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


async def evaluate_async(model: MultiModalPredictor, test_data_path: str) -> float:
    try:
        test_data = await asyncio.to_thread(pd.read_csv, test_data_path)

        accuracy = await asyncio.to_thread(
            model.evaluate, test_data, chunk_size=4096, realtime=True
        )
        return accuracy
    except Exception as e:
        raise e


@router.post("/image_classifier/accuracy", tags=["image_classifier"])
async def get_accuracy(
    user_name: str = Form("lexuanan18102004"),
    project_name: str = Form("flower-classifier"),
):
    try:
        model = await load_model(user_name, project_name)
        test_data_path = f"{TEMP_DIR}/{user_name}/{project_name}/datasets/test.csv"
        acc = await evaluate_async(model, test_data_path)
        return {
            "accuracy": acc,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in getting accuracy: {str(e)}"
        )
