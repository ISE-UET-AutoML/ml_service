from pydantic import BaseModel



class CloudTrainRequest(BaseModel):
    dataset_url: str
    dataset_label_url: str
    training_time: int = 60
    presets: str = "high_quality"
    