from fastapi import APIRouter
from .train.routes_v1 import router as train_router

router = APIRouter()
router.include_router(train_router, prefix="/train", tags=["train"])