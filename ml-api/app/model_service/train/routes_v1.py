import re
import uuid
from fastapi import APIRouter, File, Form, UploadFile
import redis
from settings.config import celery_client, redis_client
from .temp_predict import router as temp_predict_router
from .celery_predict import router as celery_predict_router
from helpers import time as time_helper
from .TrainRequest import (
    GenericMultiModalTrainRequest,
    ImageClassifyTrainRequest,
    ImageSegmentationTrainRequest,
    NamedEntityRecognitionTrainRequest,
    ObjectDetectionTrainRequest,
    TTSemanticMatchingTrainRequest,
    TabularTrainRequest,
    TimeSeriesTrainRequest,
    TrainRequest,
)

router = APIRouter()
router.include_router(temp_predict_router, prefix="")
router.include_router(celery_predict_router, prefix="")


@router.post(
    "/tabular_classification",
    tags=["tabular_classification"],
    description="Train tabular classification model, can also be used for tabular regression\nfaster and better performance multimodal",
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


@router.post("/object_detection", tags=["object_detection"])
async def train_object_detection(request: ObjectDetectionTrainRequest):
    print("Object Detection Training request received")

    task_id = celery_client.send_task(
        "model_service.object_detection.train",
        kwargs={
            "request": request.dict(),
        },
        queue="ml_celery",
    )

    return {
        "task_id": str(task_id),
        "send_status": "SUCCESS",
    }


@router.post(
    "/generic_multimodal_prediction",
    tags=["generic_multimodal"],
)
async def train_generic_mm_prediction(request: GenericMultiModalTrainRequest):
    print("Generic Multimodal Prediction Training request received")

    task_id = celery_client.send_task(
        "model_service.generic_mm_prediction.train",
        kwargs={
            "request": request.dict(),
        },
        queue="ml_celery",
    )

    return {
        "task_id": str(task_id),
        "send_status": "SUCCESS",
    }


@router.post(
    "/image_segmentation",
    tags=["image_segmentation"],
)
async def train_image_segmentation(request: ImageSegmentationTrainRequest):
    print("Image Segmentation Training request received")

    task_id = celery_client.send_task(
        "model_service.image_segmentation.train",
        kwargs={
            "request": request.dict(),
        },
        queue="ml_celery",
    )

    return {
        "task_id": str(task_id),
        "send_status": "SUCCESS",
    }


@router.post(
    "/named_entity_recognition",
    tags=["named_entity_recognition"],
)
async def train_named_entity_recognition(request: NamedEntityRecognitionTrainRequest):
    print("Named Entity Recognition Training request received")

    task_id = celery_client.send_task(
        "model_service.named_entity_recognition.train",
        kwargs={
            "request": request.dict(),
        },
        queue="ml_celery",
    )

    return {
        "task_id": str(task_id),
        "send_status": "SUCCESS",
    }


@router.post(
    "/text_text_semantic_matching",
    tags=["semantic_matching"],
)
def train_text_text_semantic_matching(request: TTSemanticMatchingTrainRequest):
    print("Text Text Semantic Matching Training request received")

    task_id = celery_client.send_task(
        "model_service.text_text_semantic_matching.train",
        kwargs={
            "request": request.dict(),
        },
        queue="ml_celery",
    )

    return {
        "task_id": str(task_id),
        "send_status": "SUCCESS",
    }


@router.post(
    "/img_img_semantic_matching",
    tags=["semantic_matching"],
)
def train_img_semantic_matching(request: TTSemanticMatchingTrainRequest):
    print("Text Text Semantic Matching Training request received")

    task_id = celery_client.send_task(
        "model_service.img_img_semantic_matching.train",
        kwargs={
            "request": request.dict(),
        },
        queue="ml_celery",
    )

    return {
        "task_id": str(task_id),
        "send_status": "SUCCESS",
    }


@router.post(
    "/time_series",
    tags=["time_series"],
)
def train_time_series(request: TimeSeriesTrainRequest):
    print("Time Series Training request received")

    task_id = celery_client.send_task(
        "model_service.time_series.train",
        kwargs={
            "request": request.dict(),
        },
        queue="ml_celery",
    )

    return {
        "task_id": str(task_id),
        "send_status": "SUCCESS",
    }
