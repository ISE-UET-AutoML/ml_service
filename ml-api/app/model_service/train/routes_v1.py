import re
import uuid
from fastapi import APIRouter, File, Form, UploadFile
import redis
from settings.config import celery_client, redis_client
from helpers import time as time_helper
from .TrainRequest import ImageClassifyTrainRequest, TabularTrainRequest

router = APIRouter()


@router.post("/tabular_classification", tags=["tabular_classification"])
async def train_tabular_classification(request: TabularTrainRequest):
    # this can also train tabular regression, but might change in the future
    print("Tabular Classification Training request received")

    time = time_helper.now_utc()
    task_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_OID,
            request.userEmail
            + "_"
            + request.projectName
            + "_"
            + request.runName
            + "_"
            + str(time),
        )
    )
    redis_client.set(task_id, "PENDING")

    celery_client.send_task(
        "model_service.tabular_classify.train",
        kwargs={
            "task_id": task_id,
            "request": request.dict(),
        },
        queue="ml_celery",
    )
    return {
        "task_id": task_id,
        "send_status": "SUCCESS",
    }


@router.post("/image_classification", tags=["image_classification"])
async def train_image_classification(request: ImageClassifyTrainRequest):
    print("Image Classification Training request received")

    time = time_helper.now_utc()
    task_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_OID,
            request.userEmail
            + "_"
            + request.projectName
            + "_"
            + request.runName
            + "_"
            + str(time),
        )
    )
    redis_client.set(task_id, "PENDING")

    celery_client.send_task(
        "model_service.image_classify.train",
        kwargs={
            "task_id": task_id,
            "request": request.dict(),
        },
        queue="ml_celery",
    )
    return {
        "task_id": task_id,
        "send_status": "SUCCESS",
    }


@router.post("/image_classification/predict", tags=["image_classification"])
async def predict(
    userEmail: str = Form("lexuanan18102004"),
    projectName: str = Form("flower-classifier"),
    runName: str = Form("Model-v0"),
    image: UploadFile = File(...),
):
    time = time_helper.now_utc()
    task_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_OID,
            userEmail + "_" + projectName + "_" + runName + "_" + str(time),
        )
    )
    redis_client.set(f"status-{task_id}", "PENDING")

    celery_client.send_task(
        "model_service.image_classify.predict",
        kwargs={
            "task_id": task_id,
            "request": {
                "userEmail": userEmail,
                "projectName": projectName,
                "runName": runName,
                "image": image,
            },
        },
        queue="ml_celery",
    )
    return {
        "task_id": task_id,
        "send_status": "SUCCESS",
    }
