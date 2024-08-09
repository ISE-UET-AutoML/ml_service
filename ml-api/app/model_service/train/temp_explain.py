import base64
from email.mime import image
from io import StringIO
import re
from typing import Optional, Union
from urllib import response
from zipfile import ZipFile
from fastapi import APIRouter, File, Form, UploadFile, Body
import pandas
from sympy import false, use
from time import perf_counter
from autogluon.multimodal import MultiModalPredictor
from autogluon.tabular import TabularPredictor
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from .explainers.ImageExplainer import ImageExplainer
from .explainers.TextExplainer import TextExplainer
import joblib
from settings.config import TEMP_DIR
import shutil
import numpy as np
from time import time
from .ExplainRequest import TextExplainRequest

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


@memory.cache
def load_timeseries_model_from_path(model_path: str) -> TimeSeriesPredictor:
    #! tabular predictor load model from the folder containing the model
    return TimeSeriesPredictor.load(os.path.dirname(model_path))


async def load_model(
    user_name: str, project_name: str, run_name: str
) -> MultiModalPredictor:
    model_path = find_latest_model(
        f"{TEMP_DIR}/{user_name}/{project_name}/trained_models/{run_name}"
    )
    print("model path: ", model_path)
    return load_model_from_path(model_path)


async def load_timeseries_model(
    user_name: str, project_name: str, run_name: str
) -> TimeSeriesPredictor:
    model_path = find_latest_tabular_model(
        f"{TEMP_DIR}/{user_name}/{project_name}/trained_models/{run_name}"
    )
    print("model path: ", model_path)
    return load_timeseries_model_from_path(model_path)


async def load_tabular_model(
    user_name: str, project_name: str, run_name: str
) -> TabularPredictor:
    model_path = find_latest_tabular_model(
        f"{TEMP_DIR}/{user_name}/{project_name}/trained_models/{run_name}"
    )
    print("model path: ", model_path)
    return load_tabular_model_from_path(model_path)



# image explain
@router.post(
    "/image_classification/explain",
    tags=["image_classification"],
    description="Only use in dev and testing, not for production",
)
async def img_explain(
    userEmail: str = Form("test-automl"),
    projectName: str = Form("4-animal"),
    runName: str = Form("ISE"),
    image: UploadFile = File(...),
):
    # print(userEmail)
    # print(projectName)
    # print("Run Name:", runName)
    try:
        # write the image to a temporary file
        temp_image_path = f"{TEMP_DIR}/{userEmail}/{projectName}/temp.jpg"
        temp_explain_image_path = f"{TEMP_DIR}/{userEmail}/{projectName}/explain.jpg"
        temp_directory_path = f"{TEMP_DIR}/{userEmail}/{projectName}/temp"

        os.makedirs(Path(temp_image_path).parent, exist_ok=True)
        os.makedirs(Path(temp_explain_image_path).parent, exist_ok=True)
        os.makedirs(temp_directory_path, exist_ok=True)

        with open(temp_image_path, "wb") as buffer:
            buffer.write(await image.read())

        start_load = perf_counter()
        # TODO : Load model with any path
        model = await load_model(userEmail, projectName, runName)
        load_time = perf_counter() - start_load
        inference_start = perf_counter()

        explainer = ImageExplainer("lime", model, temp_directory_path, num_samples=100, batch_size=50, class_names=[label for label in model.class_labels])
        try:
            explainer.explain(temp_image_path, temp_explain_image_path)
            # TODO: change return format, base64 string usually very slow
            with open(temp_explain_image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
        except Exception as e:
            print(e)

        return {
            "status": "success",
            "message": "Explanation completed",
            "load_time": load_time,
            "inference_time": perf_counter() - inference_start,
            "explain_image": encoded_image,
        }
    except Exception as e:
        print(e)
    finally:
        print("Cleaning up")
        if os.path.exists(temp_directory_path):
            shutil.rmtree(temp_directory_path)



# text explain
@router.post(
    "/text_prediction/explain",
    tags=["text_prediction"],
    description="Only use in dev and testing, not for production",
)
async def text_explain(
    userEmail: str = Form(...),
    projectName: str = Form(...),
    runName: str = Form("ISE"),
    text: str = Form(...),
):

    # print(request.userEmail)
    # print(request.projectName)
    # print("Run Name:", request.runName)
    # print(request.text)

    start_time = time()
    try:
        start_load = perf_counter()
        # TODO : Load model with any path
        model = await load_model(userEmail, projectName, runName)
        load_time = perf_counter() - start_load
        inference_start = perf_counter()

        explainer = TextExplainer("shap", model, class_names=[label for label in model.class_labels])
        try:
            explain_html = explainer.explain(text)
        except Exception as e:
            print(e)

        return {
            "status": "success",
            "message": "Explanation completed",
            "load_time": load_time,
            "inference_time": perf_counter() - inference_start,
            "explain_html": explain_html,
        }
    except Exception as e:
        print(e)
    finally:
        print(f"Eplapsed time: {time() - start_time}")
        pass