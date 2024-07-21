from typing import Optional
from pydantic import BaseModel, Field


class MlResult(BaseModel):
    task_id: str
    status: dict = None
    time: dict = None
    error: Optional[str] = None


class TrainRequest(BaseModel):
    userEmail: str = Field(default="test-automl", title="userEmail without @")
    projectName: str = Field(default="titanic", title="Project name")
    training_time: int = Field(default=60, title="Training Time")
    runName: str = Field(default="ISE", title="Run name")
    presets: str = Field(default="medium_quality", title="Presets")
    dataset_url: str = Field(
        default="1yIkh7Wvu4Lk1o6gVIuyXTb3l9zwNXOCE", title="Dataset URL"
    )
    gcloud_dataset_bucketname: str = Field(title="gcloud dataset bucketname")
    gcloud_dataset_directory: str = Field(title="gcloud dataset directory")
    dataset_download_method: str = Field(
        default="gdrive",
        title="Dataset download method",
        description="if 'gdrive', dataset url is an id\n",
    )
    label_column: str = Field(default="label")
    training_argument: dict = Field(
        default={
            # "data_args": {},
            # "ag_model_args": {},
            "ag_fit_args": {
                "time_limit": 60,
            },
        },
        title="Training arguments.",
    )


class TabularTrainRequest(TrainRequest):
    presets: str | list[str] = Field(default="good_quality", title="Presets")
    label_column: str = Field(
        default="Survived", description="Survived for titanic dataset"
    )


from dataclasses import dataclass


@dataclass
class Timm_Checkpoint:
    swin_small_patch4_window7_224: str = "swin_small_patch4_window7_224"
    swin_base_patch4_window7_224: str = "swin_base_patch4_window7_224"
    swin_large_patch4_window7_224: str = "swin_large_patch4_window7_224"
    swin_large_patch4_window12_384: str = "swin_large_patch4_window12_384"


class ImageClassifyTrainRequest(TrainRequest):
    training_argument: dict = Field(
        default={
            "data_args": {},
            "ag_model_args": {
                "pretrained": True,
                "hyperparameters": {
                    "model.timm_image.checkpoint_name": Timm_Checkpoint.swin_small_patch4_window7_224,
                },
            },
            "ag_fit_args": {
                "time_limit": 60,
                "hyperparameters": {"env.per_gpu_batch_size": 4, "env.batch_size": 4},
            },
        },
        title="Training arguments.",
    )


class ObjectDetectionTrainRequest(TrainRequest):
    projectName: str = Field(default="mini-motobike", title="Project name")


class ImageSegmentationTrainRequest(TrainRequest):
    image_cols: list[str] = Field(
        default=["image", "label"],
        title="Image column",
        description="List of image columns, used to expand image path",
    )
