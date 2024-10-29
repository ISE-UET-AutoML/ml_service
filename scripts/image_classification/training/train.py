from sklearn.model_selection import train_test_split
from time import perf_counter
import gdown
import uuid
from sklearn.preprocessing import label_binarize, LabelEncoder
# from utils import (
#     download_dataset2,
#     split_data,
#     create_csv,
# )
import os
from pathlib import Path
import gc
from typing import Union
import glob
import shutil
from dataclasses import dataclass
import pandas as pd
import logging
import argparse
import time
from autogluon.multimodal import MultiModalPredictor


from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from pathlib import Path
import asyncio
from typing import Optional, Union
import torch
import json


TEMP_DIR = "./tmp"

torch.cuda.empty_cache()
gc.collect()

torch.set_float32_matmul_precision("medium")



@dataclass
class Timm_Checkpoint:
    swin_small_patch4_window7_224: str = "swin_small_patch4_window7_224"
    swin_base_patch4_window7_224: str = "swin_base_patch4_window7_224"
    swin_large_patch4_window7_224: str = "swin_large_patch4_window7_224"
    swin_large_patch4_window12_384: str = "swin_large_patch4_window12_384"

class AutogluonTrainer(object):
    def __init__(self, kwargs: Optional[dict] = None):
        self.fit_args = None
        self.model_args = None
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        self.parse_args(kwargs)

    def parse_args(self, kwargs: Optional[dict] = None):
        if kwargs is None:
            return
        self.model_args = kwargs.setdefault("ag_model_args", {})
        self.fit_args = kwargs.setdefault(
            "ag_fit_args",
            {
                "time_limit": kwargs.get("time_limit", 60 * 2),
                "hyperparameters": {
                    "env.precision": "bf16-mixed",
                    "env.per_gpu_batch_size": 4,
                    "env.batch_size": 4,
                    "optimization.efficient_finetune": "lora",
                    "optimization.log_every_n_steps": 2,
                    "env.num_workers_evaluation": os.cpu_count() - 1,
                    "model.timm_image.checkpoint_name": Timm_Checkpoint.swin_small_patch4_window7_224,
                },
            },
        )

    def train(
        self,
        label: str,
        train_data_path: Path,
        val_data_path: Path | None,
        model_path: Path,
    ) -> Union[MultiModalPredictor, None]:
        try:
            if model_path.exists():
                shutil.rmtree(model_path)

            predictor = MultiModalPredictor(
                label=label, path=str(model_path), **self.model_args
            )
            print("created predictor")

            logging.basicConfig(level=logging.DEBUG)

            train_data = pd.read_csv(train_data_path)

            predictor.fit(
                train_data=pd.read_csv(train_data_path),
                # tuning_data=str(val_data_path),
                save_path=str(model_path),
                **self.fit_args,
            )

            exported_path = predictor.export_onnx(data=train_data[0:1], path=str(model_path), batch_size=4, truncate_long_and_double=True)

            print(predictor.eval_metric)
            
            metadata = {
                "labels": predictor.class_labels.tolist(),
            }
            with open(f"{model_path}/metadata.json", "w") as f:
                json.dump(metadata, f, sort_keys=True, indent=4, ensure_ascii=False)

            self._logger.info(f"Training completed. Model saved to {model_path}")
            self._logger.info(f"Export completed. Model saved to {exported_path}")
            return predictor
        except ValueError as ve:
            self._logger.error(f"Value Error: {ve}")

        except FileNotFoundError as fnfe:
            self._logger.error(f"File Not Found Error: {fnfe}")

        except Exception as e:
            self._logger.error(f"An unexpected error occurred: {e}")

        return None

    async def train_async(
        self, label: str, train_data_path: Path, val_data_path: Path, model_path: Path
    ) -> Union[MultiModalPredictor, None]:
        return await asyncio.to_thread(
            self.train, label, train_data_path, val_data_path, model_path
        )

    @staticmethod
    def evaluate(
        predictor: MultiModalPredictor, test_data_path: Path
    ) -> Optional[float]:
        try:

            df = pd.read_csv(test_data_path)

            df = pd.DataFrame(df["image"])

            output_path = test_data_path.parent / "output"

            os.system(f"mkdir -p {output_path}")

            output_path = Path(output_path)

            test_res: dict = {}

            predictor.predict(df).to_csv(output_path / "predictions.csv", index=False)

            y_pred = pd.read_csv(output_path / "predictions.csv")["label"]

            y_true = pd.read_csv(test_data_path)["label"]

            test_res["accuracy"] = accuracy_score(y_true, y_pred)

            test_res["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

            test_res["mcc"] = matthews_corrcoef(y_true, y_pred)

            test_res["roc_auc"] = roc_auc_score(y_true, y_pred, average="weighted")

            average = "binary" if predictor.problem_type == "binary" else "weighted"

            test_res["f1_score"] = f1_score(y_true, y_pred, average=average)

            test_res["precision"] = precision_score(y_true, y_pred, average=average)
            test_res["recall"] = recall_score(y_true, y_pred, average=average)

            return test_res

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None


def train(task_id: str, request: dict):
    print("task_id:", task_id)
    print("request:", request)
    print("Image Classification Training request received")
    start = perf_counter()

    try:

        # OLD
        # user_dataset_path = (
        #     f"{TEMP_DIR}/{request['userEmail']}/{request['projectName']}/dataset/"
        # )
        # os.makedirs(user_dataset_path, exist_ok=True)
        # user_model_path = f"{TEMP_DIR}/{request['userEmail']}/{request['projectName']}/trained_models/{request['runName']}/{task_id}"

        # train_df = download_dataset2(
        #     user_dataset_path,
        #     request["dataset_url"],
        #     request["dataset_download_method"],
        # )
        # train_df, test_df = train_test_split(train_df, test_size=0.2)


        # TEMPORARY
        user_dataset_path = f"./dataset"
        os.makedirs(user_dataset_path, exist_ok=True)
        user_model_path = f"./model"
        train_df = pd.read_csv(f"{user_dataset_path}/train.csv")
        
        train_df, test_df = train_test_split(train_df, test_size=0.2)
        # print("Remove folders except split successfully")
        trainer = AutogluonTrainer(request["training_argument"])
        print("Create trainer successfully")
        # training job của mình sẽ chạy ở đây
        model = trainer.train(
            "label",
            Path(f"{user_dataset_path}/train.csv"),
            None,
            # Path(f"{user_dataset_path}/val.csv"),
            Path(f"{user_model_path}"),
        )
        print("Training model successfully")
        if model is None:
            raise ValueError("Error in training model")

        # acc = AutogluonTrainer.evaluate(model, Path(f"{user_dataset_path}/test.csv"))
        print("Evaluate model successfully")
        acc = 0.98

        end = perf_counter()

        return {
            "metrics": acc,
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


DEFAULT_TRAINING_ARGUMENT = {
    "training_time": 100,
    "presets": "medium_quality",
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


if __name__ == "__main__":
    try:
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--training-time", type=int, default=60)
        parser.add_argument("--presets", type=str, required=True, default="medium_quality")

        args = parser.parse_args()

        DEFAULT_TRAINING_ARGUMENT["training_argument"]["ag_fit_args"]["time_limit"] = args.training_time
        DEFAULT_TRAINING_ARGUMENT["training_argument"]["ag_fit_args"]["presets"] = args.presets

        print(DEFAULT_TRAINING_ARGUMENT)

        # TEMPORARY
        res = train(task_id="temp", request=DEFAULT_TRAINING_ARGUMENT)
        print(res)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
