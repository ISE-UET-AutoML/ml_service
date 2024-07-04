
from typing import Optional
from pydantic import BaseModel, Field

class MlResult(BaseModel):
    task_id: str
    status: dict = None
    time: dict = None
    error: Optional[str] = None

class TrainRequest(BaseModel):
    userEmail: str = Field(default="test-automl-bucket1", title="userEmail without @")
    projectName: str = Field(default="titanic", title="Project name")
    training_time: int = Field(default=60, title="Training Time")
    runName: str = Field(default="ISE", title="Run name")
    dataset_url: str = Field(default="", title="Dataset URL")
    training_argument: dict = Field(default =
    {
        "data_args": {            
            "time_limit": 60,
        },
        "ag_model_args": {        },
        "ag_fit_args": {        }
    }, title="Training arguments.")

class TabularTrainRequest(TrainRequest):
    label_column: str = Field(default="label")

    
class ImageClassifyTrainRequest(TrainRequest):
    label_column: str = Field(default="label")
