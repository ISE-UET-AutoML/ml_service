from time import perf_counter
from autogluon.multimodal import MultiModalPredictor
import joblib
from settings.config import TEMP_DIR

from utils.dataset_utils import (
    find_latest_model,
    split_data,
    create_csv,
    remove_folders_except,
    create_folder,
    download_dataset,
)
from utils.train_utils import find_in_current_dir
import os
from pathlib import Path
import pandas as pd


def train(task_id: str, request: dict):
    print("task_id:", task_id)
    print("request:", request)
    print("MultiModal Training request received")
    start = perf_counter()
    request["training_argument"]["ag_fit_args"]["time_limit"] = request["training_time"]
    request["training_argument"]["ag_fit_args"]["presets"] = request["presets"]
    try:
        user_dataset_path = (
            f"{TEMP_DIR}/{request['userEmail']}/{request['projectName']}/dataset/"
        )
        os.makedirs(user_dataset_path, exist_ok=True)
        user_model_path = f"{TEMP_DIR}/{request['userEmail']}/{request['projectName']}/trained_models/{request['runName']}/{task_id}"

        # TODO: download dataset in this function
        user_dataset_path = download_dataset(
            user_dataset_path, True, request, request["dataset_download_method"]
        )

        # get data path
        train_path = find_in_current_dir(
            "train", user_dataset_path, is_pattern=True, extension=".csv"
        )
        # val_path = find_in_current_dir(
        #     "val", user_dataset_path, is_pattern=True, extension=".csv"
        # )
        # if val_path == train_path:
        #     val_path = None
        # test_path = find_in_current_dir(
        #     "test", user_dataset_path, is_pattern=True, extension=".csv"
        # )
        train_path = f"{user_dataset_path}{train_path}"
        # if val_path is not None:
        #     val_path = f"{user_dataset_path}/{val_path}"
        # test_path = f"{user_dataset_path}/{test_path}"

        # expanding image path
        train_data = pd.read_csv(train_path)
        # val_data = None
        # if val_path is not None:
        #     val_data = pd.read_csv(val_path)
        # test_data = pd.read_csv(test_path)

        for img_col in request["image_cols"]:
            train_data[img_col] = train_data[img_col].apply(
                lambda x: f"{user_dataset_path}/{x}"
            )
            # if val_path is not None:
            #     val_data[img_col] = val_data[img_col].apply(
            #         lambda x: f"{user_dataset_path}/{x}"
            #     )
            # test_data[img_col] = test_data[img_col].apply(
            #     lambda x: f"{user_dataset_path}/{x}"
            # )

        presets = request["presets"]

        # # training job của mình sẽ chạy ở đây
        predictor = MultiModalPredictor(
            label=request["label_column"],
            problem_type=request["problem_type"],
            path=user_model_path,
        )

        predictor.fit(
            train_data=train_data,
            # tuning_data=val_data,
            time_limit=request["training_time"],
            presets=presets,
            save_path=user_model_path,
            hyperparameters=request["training_argument"]["ag_fit_args"][
                "hyperparameters"
            ],
        )

        # metrics = predictor.evaluate(test_data, metrics=request["metrics"])
        # print("Training model successfully")

        end = perf_counter()

        return {
            "metrics": "temp_metrics",
            "training_evaluation_time": end - start,
            "saved_model_path": user_model_path,
        }

    except Exception as e:
        print(e)
        # raise HTTPException(status_code=500, detail=f"Error in downloading or extracting folder: {str(e)}")
    # finally:
    # if os.path.exists(temp_dataset_path):
    #    os.remove(temp_dataset_path)
    # return {}


memory = joblib.Memory(
    f"{TEMP_DIR}", verbose=0, mmap_mode="r", bytes_limit=1024 * 1024 * 1024 * 100
)


@memory.cache
def load_model_from_path(model_path: str) -> MultiModalPredictor:
    return MultiModalPredictor.load(model_path)


async def load_model(
    user_name: str, project_name: str, run_name: str
) -> MultiModalPredictor:
    model_path = find_latest_model(
        f"{TEMP_DIR}/{user_name}/{project_name}/trained_models/{run_name}"
    )
    print("model path: ", model_path)
    return load_model_from_path(model_path)


async def predict(task_id: str, request: dict):
    print("task_id:", task_id)
    print("request:", request)
    print("MultiModal Prediction request received")
    start = perf_counter()
    try:
        user_dataset_path = f"{TEMP_DIR}/{request['userEmail']}/{request['projectName']}/dataset/predict/"
        os.makedirs(user_dataset_path, exist_ok=True)

        user_dataset_path = download_dataset(
            user_dataset_path,
            True,
            request["dataset"],
            request["dataset"]["dataset_download_method"],
        )
        predict_file = request["predict_file"]
        if predict_file == None:
            predict_file = find_in_current_dir(
                "", user_dataset_path, is_pattern=True, extension=".csv"
            )
        predict_file = f"{user_dataset_path}/{predict_file}"

        predict_data = pd.read_csv(predict_file)
        for img_col in request["image_cols"]:
            predict_data[img_col] = predict_data[img_col].apply(
                lambda x: f"{user_dataset_path}/{x}"
            )

        model = await load_model(
            request["userEmail"], request["projectName"], request["runName"]
        )
        load_end = perf_counter()

        if request["evaluate"] == True:
            metrics = model.evaluate(predict_data)
            end = perf_counter()
            return {
                "status": "success",
                "message": "Evaluation completed",
                "load_time": load_end - start,
                "evaluation_time": end - load_end,
                "metrics": str(metrics),
            }
        else:
            predictions = model.predict(predict_data)
            try:
                proba = model.predict_proba(predict_data)
                proba = proba.to_csv()
            except Exception as e:
                proba = "None"
            end = perf_counter()
            return {
                "status": "success",
                "message": "Prediction completed",
                "load_time": load_end - start,
                "inference_time": end - load_end,
                "proba": proba,
                "predictions": predictions.to_csv(),
            }
    except Exception as e:
        print(e)
        return {"status": "error", "message": str(e)}
    finally:
        pass
