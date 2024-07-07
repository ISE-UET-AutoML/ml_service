from cProfile import label
from pathlib import Path
from urllib import request
from celery import shared_task
import uuid
import time
from worker.model_service.tabular.autogluon_trainer import AutogluonTrainer
from mq_main import redis
from time import perf_counter
import os
import uuid
import joblib
from settings.config import TEMP_DIR



async def train(request: dict):
    print("Tabular Training request received")
    temp_dataset_path = ""
    start = perf_counter()
    print("Training request received")
    request["training_argument"]["ag_fit_args"]["time_limit"] = request["training_time"]
    try:
        # temp folder to store dataset and then delete after training
        temp_dataset_path = Path(
            f"{TEMP_DIR}/{request['userEmail']}/{request['projectName']}"
        )

        os.makedirs(temp_dataset_path.parent, exist_ok=True)

        print("make temp folder successfully")
        # res = await storage.download_blob_async(
        #     request["userEmail"],
        #     f"{request["projectName"]}/datasets/datasets.csv",
        #     f"{temp_dataset_path}/datasets.csv",
        # )

        download_end = perf_counter()
        print("download dataset successfully")

        # if res is None or not res:
        #     raise ValueError("Error in downloading data")

        # user_dataset_path = f"{TEMP_DIR}/{request.userEmail}/{request.projectName}/datasets"
        user_model_path = f"{TEMP_DIR}/{request['userEmail']}/{request['projectName']}/trained_models/{request['runName']}/{uuid.uuid4()}"

        # create_folder(Path(user_dataset_path))

        # with ZipFile(temp_dataset_path, 'r') as zip_ref:
        #     zip_ref.extractall(user_dataset_path)

        # split_data(Path(user_dataset_path), f"{user_dataset_path}/split/")

        # # # TODO : User can choose ratio to split data @DuongNam
        # # # assume user want to split data into 80% train, 10% val, 10% test

        # create_csv(Path(f"{user_dataset_path}/split/train"),
        #            Path(f"{user_dataset_path}/train.csv"))
        # create_csv(Path(f"{user_dataset_path}/split/val"),
        #            Path(f"{user_dataset_path}/val.csv"))
        # create_csv(Path(f"{user_dataset_path}/split/test"),
        #            Path(f"{user_dataset_path}/test.csv"))
        train_path = f"{temp_dataset_path}/datasets.csv"
        # test_path=os.path.dirname(__file__)+"/titanic/test.csv"
        # print("Split data successfully")
        # remove_folders_except(Path(user_dataset_path), "split")
        # print("Remove folders except split successfully")

        trainer = AutogluonTrainer(request["training_argument"])
        # trainer = AutogluonTrainer()

        #! target can change to any column in the dataset
        # TODO: target column should be selected by user
        target = request["label_column"]
        print("Create trainer successfully")

        model = trainer.train(target, train_path, None, user_model_path)

        if model is None:
            raise ValueError("Error in training model")
        print("Training model successfully")

        end = perf_counter()

        return {
            "validation_accuracy": 0,  #!
            "training_evaluation_time": end - start,
            "model_download_time": download_end - start,
            "saved_model_path": user_model_path,
        }

    except Exception as e:
        print(e)
        # raise HTTPException(status_code=500, detail=f"Error in downloading or extracting folder: {str(e)}")
    # finally:
    # if os.path.exists(temp_dataset_path):
    # os.remove(temp_dataset_path)


def predict(task_id: str, request: dict):
    print("Tabular Predict request received")
    try:
        model_path = request["model_path"]
        test_path = request["test_path"]
        trainer = AutogluonTrainer()
        model = trainer.load_model(model_path)
        result = trainer.predict(model, test_path)
        return result
    except Exception as e:
        print(e)
        return {"error": str(e)}
