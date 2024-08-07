from pathlib import Path
from fastapi import APIRouter

from settings.config import label_studio_client

from .project_route import router as project_router

router = APIRouter()

router.include_router(project_router, prefix="/projects", tags=["projects"])
