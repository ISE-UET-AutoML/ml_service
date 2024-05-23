from fastapi import FastAPI, File, UploadFile
from nptyping import Byte
from pydantic import BaseModel
from pydantic.fields import Field



class TabularTrainingRequest(BaseModel):
    # Mỗi user là 1 bucket riêng trong GCS
    userEmail: str = Field(default="lexuanan18102004", title="userEmail but loai bo @ to get the bucket => Have to be unique")
    projectName: str = Field(default="flower-classifier", title="Project name")
    training_time: int = Field(default=60, title="Training Time")
    runName: str = Field(default="Namdeptraiqua", title="Run name")
    training_argument: dict = Field(default =
    {
        "data_args": {},
        "ag_fit_args": {
            "time_limit": 60,
        }
    }, title="Training arguments.")