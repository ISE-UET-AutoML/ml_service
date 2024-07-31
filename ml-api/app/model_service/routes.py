import os
from pathlib import Path
from re import sub
import subprocess
import celery
from fastapi import APIRouter, Form
import requests
from torch import tensor
from .train.routes_v1 import router as train_router
from .train.routes_v2 import router as train_router_v2
from .autogluon_zeroshot.routes import router as zeroshot_router
from settings.config import (
    celery_client,
    redis_client,
    BACKEND_HOST,
    ACCESS_TOKEN,
    REFRESH_TOKEN,
)

from celery.result import AsyncResult


router = APIRouter()
router.include_router(train_router, prefix="/train")
router.include_router(train_router_v2, prefix="/train")
router.include_router(zeroshot_router, prefix="/zeroshot")


@router.get("/cel_test")
def test_celery():
    task_id = celery_client.send_task(
        "time_test", kwargs={"request": {}}, queue="ml_celery"
    )
    print(task_id)
    return {"task_id": str(task_id)}


@router.get("/status/{task_id}")
def status(task_id):
    result = celery_client.AsyncResult(task_id)
    return {"status": result.status}


@router.get("/result/{task_id}")
def result(task_id):
    result = celery_client.AsyncResult(task_id)
    status = result.status
    if status == "SUCCESS":
        return {"status": status, "result": result.get()}
    return {"status": status, "result": ""}


from requests.auth import HTTPBasicAuth


@router.post("/getdataset")
def get_be_dataset(
    projectID: str = Form(...),
):
    # ${BACKENDHOST}/projects/${projectID}/datasets
    dataset_url = f"{BACKEND_HOST}/projects/{projectID}/datasets"

    temp = requests.get(dataset_url, cookies={"accessToken": ACCESS_TOKEN})

    print(temp.text)
