from fastapi import FastAPI, File, UploadFile
from nptyping import Byte
from pydantic import BaseModel
from pydantic.fields import Field

from app.image_classifier.available_checkpoint import Timm_Checkpoint


class TrainingRequest(BaseModel):
    # Mỗi user là 1 bucket riêng trong GCS
    userEmail: str = Field(default="lexuanan18102004", title="userEmail but loai bo @ to get the bucket => Have to be unique")
    projectName: str = Field(default="flower-classifier", title="Project name")
    training_time: int = Field(default=60, title="Training Time")
    runName: str = Field(default="Namdeptraiqua", title="Run name")
    training_argument: dict = Field(default =
    {
        "data_args": {},
        "ag_model_args": {
            "pretrained": True,
            "hyperparameters": {
                "model.timm_image.checkpoint_name": Timm_Checkpoint.swin_small_patch4_window7_224,
            }
        },
        "ag_fit_args": {
            "time_limit": 60,
            "hyperparameters": {
                "env.per_gpu_batch_size": 4,
                "env.batch_size": 4
            }
        }
    }, title="Training arguments.")