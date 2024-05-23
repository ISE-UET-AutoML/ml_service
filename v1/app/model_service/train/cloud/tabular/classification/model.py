from fastapi import FastAPI, File, UploadFile
from nptyping import Byte
from pydantic import BaseModel
from pydantic.fields import Field

from app.model_service.train.training_request import TrainingRequest

class TabularTrainingRequest(TrainingRequest):
    # Mỗi user là 1 bucket riêng trong GCS
    label_column: str = Field(default="label")
    training_argument: dict = Field(default =
    {
        "ag_fit_args": {
            "time_limit": 60,
        }
    }, title="Training arguments.")