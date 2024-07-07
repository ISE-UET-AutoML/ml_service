from email.mime import image
import re
from zipfile import ZipFile
from celery import shared_task
from fastapi import APIRouter, File, Form, UploadFile
from sympy import false, use
from time import perf_counter
from autogluon.multimodal import MultiModalPredictor
import joblib
from settings.config import TEMP_DIR


from utils.dataset_utils import (
    find_latest_model,
    split_data,
    create_csv,
    remove_folders_except,
    create_folder,
)
import os
from pathlib import Path

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


@router.post(
    "/image_classification/temp_predict",
    tags=["image_classification"],
    description="Only use in dev and testing, not for production",
)
async def predict(
    userEmail: str = Form("test-automl"),
    projectName: str = Form("4-animal"),
    runName: str = Form("ISE"),
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
        print(e)
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
    #! unusable, cant jsonify UploadFile
    # task_id: str = celery_client.send_task(
    #     "model_service.image_classify.predict",
    #     kwargs={
    #         "request": {
    #             "userEmail": userEmail,
    #             "projectName": projectName,
    #             "runName": runName,
    #             "image": image,
    #         },
    #     },
    #     queue="ml_celery",
    # )

    # return {
    #     "task_id": task_id,
    #     "send_status": "SUCCESS",
    # }
