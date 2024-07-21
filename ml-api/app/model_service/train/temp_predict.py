from email.mime import image
from io import StringIO
import re
from zipfile import ZipFile
from fastapi import APIRouter, File, Form, UploadFile
import pandas
from sympy import false, use
from time import perf_counter
from autogluon.multimodal import MultiModalPredictor
from autogluon.tabular import TabularPredictor
import joblib
from settings.config import TEMP_DIR

import numpy as np

from utils.dataset_utils import (
    find_latest_model,
    find_latest_tabular_model,
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


@memory.cache
def load_tabular_model_from_path(model_path: str) -> TabularPredictor:
    #! tabular predictor load model from the folder containing the model
    return TabularPredictor.load(os.path.dirname(model_path))


async def load_model(
    user_name: str, project_name: str, run_name: str
) -> MultiModalPredictor:
    model_path = find_latest_model(
        f"{TEMP_DIR}/{user_name}/{project_name}/trained_models/{run_name}"
    )
    print("model path: ", model_path)
    return load_model_from_path(model_path)


async def load_tabular_model(
    user_name: str, project_name: str, run_name: str
) -> TabularPredictor:
    model_path = find_latest_tabular_model(
        f"{TEMP_DIR}/{user_name}/{project_name}/trained_models/{run_name}"
    )
    print("model path: ", model_path)
    return load_tabular_model_from_path(model_path)


@router.post(
    "/image_classification/temp_predict",
    tags=["image_classification"],
    description="Only use in dev and testing, not for production",
)
async def img_predict(
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
        try:
            proba: pandas.DataFrame = model.predict_proba(
                temp_image_path, as_pandas=True, as_multiclass=True
            )
        except Exception as e:
            return {
                "status": "success",
                "message": "Prediction completed",
                "load_time": load_time,
                "proba": "Not a classification problem",
                "inference_time": perf_counter() - inference_start,
                "predictions": predictions.to_csv(),
            }

        return {
            "status": "success",
            "message": "Prediction completed",
            "load_time": load_time,
            "proba": proba.to_csv(),
            "inference_time": perf_counter() - inference_start,
            "predictions": str(predictions),
        }
    except Exception as e:
        print(e)
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


@router.post(
    "tabular_classification/temp_predict",
    tags=["tabular_classification"],
    description="Only use in dev and testing, not for production",
)
async def tab_predict(
    userEmail: str = Form("test-automl"),
    projectName: str = Form("titanic"),
    runName: str = Form("ISE"),
    csv_file: UploadFile = File(...),
):
    print(userEmail)
    print(runName)
    try:
        df = pandas.read_csv(csv_file.file)
        start_load = perf_counter()
        # TODO : Load model with any path

        print("Loading model")
        model = await load_tabular_model(userEmail, projectName, runName)
        print("Model loaded")

        print(df.head())
        load_time = perf_counter() - start_load
        inference_start = perf_counter()
        predictions = model.predict(df, as_pandas=True)
        try:
            proba: pandas.DataFrame = model.predict_proba(
                df, as_pandas=True, as_multiclass=True
            )
        except Exception as e:
            return {
                "status": "success",
                "message": "Prediction completed",
                "load_time": load_time,
                "proba": "Not a classification problem",
                "inference_time": perf_counter() - inference_start,
                "predictions": predictions.to_csv(),
            }
        return {
            "status": "success",
            "message": "Prediction completed",
            "load_time": load_time,
            "proba": proba.to_csv(),
            "inference_time": perf_counter() - inference_start,
            "predictions": predictions.to_csv(),
        }
    except Exception as e:
        print(e)


@router.post(
    "/object_detection/temp_predict",
    tags=["object_detection"],
    description="Only use in dev and testing, not for production",
)
async def obj_predict(
    userEmail: str = Form("test-automl"),
    projectName: str = Form("tiny-motobike"),
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
        predictions: pandas.DataFrame = model.predict(temp_image_path, realtime=True)

        print(predictions)

        return {
            "status": "success",
            "message": "Prediction completed",
            "load_time": load_time,
            "proba": "Not a classification problem",
            "inference_time": perf_counter() - inference_start,
            "predictions": predictions.to_csv(),
        }
    except Exception as e:
        print(e)
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


@router.post(
    "/image_segmentation/temp_predict",
    tags=["image_segmentation"],
    description="Only use in dev and testing, not for production",
)
async def img_seg_predict(
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
        predictions = model.predict({"Unnamed: 0": [0], "image": [temp_image_path]})

        # temp_result_path = f"{TEMP_DIR}/{userEmail}/{projectName}/temp_predict.txt"
        np.set_printoptions(threshold=np.inf)

        # print(predictions)
        return {
            "status": "success",
            "message": "Prediction completed",
            "load_time": load_time,
            "proba": "Not a classification problem",
            "inference_time": perf_counter() - inference_start,
            "predictions": str(predictions[0].astype(int)),
        }
    except Exception as e:
        print(e)
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
