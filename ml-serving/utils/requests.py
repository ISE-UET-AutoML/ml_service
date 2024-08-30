from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Annotated
from bentoml.validators import ContentType


class DeployRequest(BaseModel):
    userEmail: str = Field(description="User Email")
    projectName: str = Field(description="Project name")
    runName: str = Field(description="Run name")
    task: str = Field(description="task to be performed", default="image_classification")


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
    image_explained_path: str = Field(description="path to store the explained image")
    method: str = Field(description="Method to explain the image", default="lime")
