from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Annotated
from bentoml.validators import ContentType



class BaseRequest(BaseModel):
    userEmail: str = Field(description="User Email")
    projectName: str = Field(description="Project name")
    runName: str = Field(description="Run name")

class DeployRequest(BaseRequest):
    task: str = Field(description="task to be performed", default="image_classification")

# IMAGE CLASSIFICATION
class ImagePredictionRequest(BaseRequest):
    images: List[str] = Field(description="Image file paths")

class ImageExplainRequest(BaseRequest):
    image: str = Field(description="Image file path")
    image_explained_path: str = Field(description="path to store the explained image")
    method: str = Field(description="Method to explain the image", default="lime")


# TEXT CLASSIFICATION
class TextPredictionRequest(BaseRequest):
    text_file_path: str = Field(description="Text file path", default="text.csv")
    text_col: List[str] = Field(description="text column in the text file")

class TextExplainRequest(BaseRequest):
    text: str = Field(description="text to explained")
    method: str = Field(description="Method to explain the text", default="lime")