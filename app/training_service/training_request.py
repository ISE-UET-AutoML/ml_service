from fastapi import FastAPI, File, UploadFile
from nptyping import Byte
from pydantic import BaseModel
from pydantic.fields import Field

from app.training_service.image_classifier.available_checkpoint import Timm_Checkpoint


class TrainingRequest(BaseModel):
    # Mỗi user là 1 bucket riêng trong GCS
    userEmail: str = Field(default="test-automl-bucket1", title="userEmail without @")
    projectName: str = Field(default="titanic", title="Project name")
    training_time: int = Field(default=60, title="Training Time")
    runName: str = Field(default="ISE", title="Run name")
    training_argument: dict = Field(default =
    {
        "data_args": {},
        "ag_model_args": {        },
        "ag_fit_args": {        }
    }, title="Training arguments.")