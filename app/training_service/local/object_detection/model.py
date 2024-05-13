from fastapi import FastAPI, File, UploadFile
from nptyping import Byte
from pydantic import BaseModel
from pydantic.fields import Field

from app.training_service.training_request import TrainingRequest

class TrainingRequest(TrainingRequest):
    # Mỗi user là 1 bucket riêng trong GCS
    training_argument: dict = Field(default =
    {
        "data_args": {},
        "ag_model_args": {},
        "ag_fit_args": {
            "time_limit": 60,
        }
    }, title="Training arguments.")