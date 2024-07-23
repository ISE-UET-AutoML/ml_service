from unittest import result
import celery
from fastapi import APIRouter
from .train.routes_v1 import router as train_router
from .autogluon_zeroshot.routes import router as zeroshot_router
from settings.config import celery_client, redis_client
from celery.result import AsyncResult

router = APIRouter()
router.include_router(train_router, prefix="/train")
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
