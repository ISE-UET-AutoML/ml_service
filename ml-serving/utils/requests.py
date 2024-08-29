from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Annotated
from bentoml.validators import ContentType


class DeployRequest(BaseModel):
    userEmail: str = Field(description="User Email")
    projectName: str = Field(description="Project name")
    runName: str = Field(description="Run name")


class ImagePredictionRequest(BaseModel):
    userEmail: str = Field(description="User Email")
    projectName: str = Field(description="Project name")
    runName: str = Field(description="Run name")
    images: List[str] = Field(description="Image file paths")


class ImageExplainRequest(BaseModel):
    userEmail: str = Field(description="User Email")
    projectName: str = Field(description="Project name")
    runName: str = Field(description="Run name")
    image: str = Field(description="Image file path")
