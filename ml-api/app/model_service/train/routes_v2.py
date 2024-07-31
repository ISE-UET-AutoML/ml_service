from fastapi import APIRouter
from .TrainRequest import SimpleTrainRequest
from settings.config import celery_client

router = APIRouter()


@router.post("/v2/image_classification", tags=["v2"])
async def train_image_classification(request: SimpleTrainRequest):
    payload = {
        "userEmail": request.userEmail,
        "projectName": request.projectId,
        "training_time": request.training_time,
        "runName": request.runName,
        "presets": request.presets,
        "dataset_url": "string",
        "gcloud_dataset_bucketname": "string",
        "gcloud_dataset_directory": "string",
        "dataset_download_method": "backend",
        "training_argument": {
            "data_args": {},
            "ag_model_args": {
                "pretrained": True,
                "hyperparameters": {
                    "model.timm_image.checkpoint_name": "swin_small_patch4_window7_224"
                },
            },
            "ag_fit_args": {
                "time_limit": 60,
                "hyperparameters": {"env.per_gpu_batch_size": 4, "env.batch_size": 4},
            },
        },
        "label_column": "label",
    }

    task_id = celery_client.send_task(
        "model_service.image_classify.train",
        kwargs={
            "request": payload,
        },
        queue="ml_celery",
    )

    return {
        "task_id": str(task_id),
        "send_status": "SUCCESS",
    }
