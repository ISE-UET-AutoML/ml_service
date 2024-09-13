from fastapi import APIRouter
from .TrainRequest import SimpleTrainRequest
from settings.config import BACKEND_HOST, celery_client

router = APIRouter()


@router.post("/v2/image_classification", tags=["v2"])
async def train_image_classification(request: SimpleTrainRequest):
    payload = {
        "userEmail": request.userEmail,
        "projectName": request.projectId,
        "training_time": request.training_time,
        "runName": request.runName,
        "presets": request.presets,
        "dataset_url": request.ds_projectId,
        "dataset_download_method": "data_service",
        "training_argument": {
            "data_args": {},
            "ag_model_args": {
                "pretrained": True,
                "hyperparameters": {
                    "model.timm_image.checkpoint_name": "swin_small_patch4_window7_224"
                },
            },
            "ag_fit_args": {
                "time_limit": 600,
                "hyperparameters": {
                    "env.precision": "bf16-mixed",
                    "env.per_gpu_batch_size": 4,
                    "env.batch_size": 4,
                    "optimization.efficient_finetune": "lora",
                    "optimization.log_every_n_steps": 2,
                    "model.timm_image.checkpoint_name": "swin_small_patch4_window7_224",
                },
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


@router.post("/v2/text_prediction", tags=["v2"])
async def train_text_prediction(request: SimpleTrainRequest):
    # TODO fix dataset
    payload = {
        "userEmail": request.userEmail,
        "projectName": request.projectId,
        "training_time": request.training_time,
        "runName": request.runName,
        "presets": request.presets,
        "dataset_url": request.ds_projectId,
        "dataset_download_method": "data_service",
        "label_column": "label",
        "training_argument": {
            "ag_fit_args": {
                "hyperparameters": {
                    "env.per_gpu_batch_size": 4,
                    "env.batch_size": 4,
                    "optimization.efficient_finetune": "lora",
                    "optimization.log_every_n_steps": 24,
                },
            },
        },
        "problem_type": None,
        "image_cols": [],
        "metrics": ["acc", "f1", "precision", "recall", "roc_auc"],
    }

    task_id = celery_client.send_task(
        "model_service.text_prediction.train",
        kwargs={
            "request": payload,
        },
        queue="ml_celery",
    )

    return {
        "task_id": str(task_id),
        "send_status": "SUCCESS",
    }
