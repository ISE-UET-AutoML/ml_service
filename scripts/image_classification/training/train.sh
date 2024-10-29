#!/bin/bash

# Exit immediately if a command exits with a non-zero status

# Variables (replace these with your actual bucket names and paths)
LOCAL_DATA_DIR="./data"           # Directory to store the downloaded data
LOCAL_MODEL_DIR="./model"         # Directory where trained model is stored
ZIP_FILE="trained_model.zip"      # Name of the zip file to upload


mkdir dataset

wget -O ./dataset/training_data.zip $1


unzip dataset/training_data.zip -d ./dataset

python3 preprocess.py --data-path "./dataset" 



python3 train.py --training-time $2 --presets $3 --dataset-url $4


# add project_id here
zip $5.zip model/metadata.json model/model.onnx


