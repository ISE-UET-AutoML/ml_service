#!/bin/bash

# Exit immediately if a command exits with a non-zero status

# Variables (replace these with your actual bucket names and paths)
LOCAL_DATA_DIR="./data"           # Directory to store the downloaded data
LOCAL_MODEL_DIR="./model"         # Directory where trained model is stored
ZIP_FILE="trained_model.zip"      # Name of the zip file to upload


mkdir $1

mkdir ./$1/dataset

wget -O ./$1/dataset/training_data.zip $2 # download dataset through presigned url


unzip ./$1/dataset/training_data.zip -d ./$1/dataset


# dataset_label_url to extract label mapping

python3 preprocess.py --dataset-label-url $3 --local-data-path "./$1/dataset" 



python3 train.py --training-time $4 --presets $5 --task-id $1 > ./$1/training_logs.txt 2>&1


# add project_id here
zip ./$1/$6 ./$1/model/metadata.json ./$1/model/model.onnx


python3 postprocess.py --model-path ./$1/$6 --saved-model-url "$7" --fit-history-url "$8"

# https://8089-01jc9yseqvq1ad9j1x1eexbpcg.cloudspaces.litng.ai/temp_data