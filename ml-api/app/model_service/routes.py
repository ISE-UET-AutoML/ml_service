from fastapi import APIRouter
from .train.routes_v1 import router as train_router
from settings.config import redis_client

router = APIRouter()
router.include_router(train_router, prefix="/train", tags=["train"])

@router.get("/status/{task_id}")
def status(task_id):
    return redis_client.get(task_id)