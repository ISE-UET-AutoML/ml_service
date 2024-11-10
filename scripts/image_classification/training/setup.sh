#!/bin/bash

# Exit immediately if a command exits with a non-zero status

export DATASERVICE_URL=$1
export BACKEND_URL=$2

echo "Updating package lists and installing Python 3.10..."
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    wget \
    unzip \
    nano \
    zip \
    libgl1-mesa-glx  # For CV packages requiring OpenGL

# Setting up python3.10
sudo add-apt-repository -y ppa:deadsnakes/ppa

sudo apt update -y

echo "Installing Python 3.10..."
sudo apt install -y python3.10 python3.10-venv python3.10-dev

echo "Setting Python 3.10 as the default python3..."
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

echo "Verifying default Python version..."
python3 --version

# Pip
sudo apt install -y python3.10-distutils  # Required to support pip installation
curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.10

# update
pip install --upgrade pip setuptools wheel

echo "Installing Python dependencies from scratch..."

pip install -r requirements.txt


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







