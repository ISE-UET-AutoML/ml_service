from fastapi import FastAPI, File, UploadFile
from nptyping import Byte
from pydantic import BaseModel
from pydantic.fields import Field

from available_checkpoint import Timm_Checkpoint


class TrainingRequest(BaseModel):
    # Mỗi user là 1 bucket riêng trong GCS
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