import re
import uuid
from fastapi import APIRouter, File, Form, UploadFile
import redis
from settings.config import celery_client, redis_client
from .temp_predict import router as temp_predict_router
from helpers import time as time_helper
from .TrainRequest import ImageClassifyTrainRequest, TabularTrainRequest

router = APIRouter()
router.include_router(temp_predict_router, prefix="")


@router.post(
    "/tabular_classification",
    tags=["tabular_classification"],
    description="Train tabular classification model, can also be used for tabular regression",
)
async def train_tabular_classification(request: TabularTrainRequest):
    # this can also train tabular regression, but might change in the future
    print("Tabular Classification Training request received")

    task_id = celery_client.send_task(
        "model_service.tabular_classify.train",
        kwargs={
            "request": request.dict(),
        },
        queue="ml_celery",
    )
    return {
        "task_id": str(task_id),
        "send_status": "SUCCESS",
    }


@router.post("/image_classification", tags=["image_classification"])
async def train_image_classification(request: ImageClassifyTrainRequest):
    print("Image Classification Training request received")

    task_id = celery_client.send_task(
        "model_service.image_classify.train",
        kwargs={
            "request": request.dict(),
        },
        queue="ml_celery",
    )

    return {
        "task_id": str(task_id),
        "send_status": "SUCCESS",
    }
