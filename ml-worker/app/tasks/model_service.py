from pathlib import Path
from celery import shared_task
import uuid
import time
from worker.model_service.tabular.autogluon_trainer import AutogluonTrainer

@shared_task(
    name='model_service.train', # it actually is 'model_service.tabular_classifier.train'
)
def train( task_id: str, request: dict):
    """_summary_: Train model
    Args:
        task_id (str): _description_

    Returns:
        _type_: _description_
    """
    # print('train_task called')
    print('task_id:', task_id)
    print('request:', request)
    # time.sleep(5)
    # print('train_task done')
    print("Tabular Training request received")
    temp_dataset_path = ""
    print("Training request received")
    request['training_argument']['ag_fit_args']['time_limit'] = request['training_time']
    try:
        # temp folder to store dataset and then delete after training
        temp_dataset_path = Path(f"D:/tmp/{request['userEmail']}/{request['projectName']}")

        #user_dataset_path = f"D:/tmp/{request.userEmail}/{request.projectName}/datasets"
        user_model_path = f"D:/tmp/ \
            {request['userEmail']}/ \
            {request['projectName']}/trained_models/ \
            {request['runName']}/{uuid.uuid4()}"

        train_path = f"{temp_dataset_path}/datasets.csv"

        
        trainer = AutogluonTrainer(request['training_argument'])
        #trainer = AutogluonTrainer()

        target=request['label_column']
        print("Create trainer successfully")

        model  = trainer.train(target, train_path, None, user_model_path)

        if model is None:
            raise ValueError("Error in training model")
        print("Training model successfully")

        return {
            "validation_accuracy": 0, #!
            "saved_model_path": user_model_path,
            "status": "success",
        }

    except Exception as e:
        print(e)
        return {
            "status": "failed",
            "error": str(e),
        }

