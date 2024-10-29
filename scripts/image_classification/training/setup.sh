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




