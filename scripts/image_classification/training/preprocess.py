import os
from pathlib import Path
import uuid
import pandas
from typing import Union
import gdown
import shutil
import requests
import json
import argparse




#  query DATA_SERVICE to get the labels and reorganize unzipped image files into folders

def preprocess_image_dataset(
    dataset_dir: str,
    dataset_url: str,
    method: str,
    format: str | None = None,
) -> pandas.DataFrame:
    """
    Download dataset

    Args:
        dataset_dir: local folder where the dataset is going to be stored, should be **/dataset/
        url:
        method: where to download dataset from
        format: for future use
    """
    #! Data set may come in many format, this function should be changed to handle all cases

    os.makedirs(dataset_dir, exist_ok=True)
    if method != "data_service":
        raise ValueError(f"Method {method} not supported")

    dataset_url = f"{os.environ['DATASERVICE_URL']}/ls/task/{dataset_url}/export_for_training"

    res = requests.get(dataset_url)
    if res.status_code >= 300:
        raise ValueError("Error in downloading dataset")
    data = res.json()["data"]


    for data_point in data:

        for k, v in data_point.items():
            if k.startswith("data-IMG-"):
                name = uuid.uuid4().hex
                label = data_point["label"]
                os.makedirs(f"{dataset_dir}/{label}", exist_ok=True)
                image_filename = v.split("/")[-1]
                shutil.move(f"{dataset_dir}/{image_filename}", f"{dataset_dir}/{label}/{image_filename}")
                data_point[k] = f"{dataset_dir}/{label}/{image_filename}"

    df = pandas.DataFrame.from_dict(data)
    df.to_csv(f"{dataset_dir}/train.csv", index=False)

    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, default="./dataset")
    parser.add_argument("--dataset-url", type=str, required=True)

    args = parser.parse_args()

    df = preprocess_image_dataset(args.data_path, args.dataset_url, "data_service")
    print("preprocess done!")