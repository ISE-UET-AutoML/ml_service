from cgi import test
from email.mime import image
import re
from zipfile import ZipFile
from celery import shared_task
from sklearn.model_selection import train_test_split
from sympy import false, use
from tqdm import tqdm
from mq_main import redis
from time import perf_counter
import gdown
from .image_classify.autogluon_trainer import AutogluonTrainer
import uuid
from autogluon.multimodal import MultiModalPredictor
import joblib
from settings.config import TEMP_DIR
from utils.aws import create_presigned_url, create_presigned_post, get_training_script_url
from utils.ssh_utils import generate_ssh_key_pair, attach_ssh_key_to_instance, connect_with_retries
import os
from pathlib import Path
import requests
from settings.config import CLOUD_INSTANCE_SERVICE_URL
import paramiko

def train(task_id: str, request: dict):
    print("task_id:", task_id)
    print("Cloud Training request received")
    start = perf_counter()
    # send the metadata including dataset_url, training_metadata: training_time and presets, train_script_url
    try:
        # defines necessary urls
        model_bucket_path = f"{request['projectName']}/{task_id}/trained_model.zip"
        saved_model_url = create_presigned_post(model_bucket_path)
        # dataset_url = request["dataset_url"]
        dataset_url = create_presigned_url(request["dataset_url"])
        dataset_label_url = request["dataset_label_url"]
        train_script_url = get_training_script_url(request['task'])
        
        # create cloud instance
        
        # instance_payload = {
        #     "task": request["task"],
        #     "training_time": request["training_time"],
        #     "presets": request["presets"],
        # }
        
        # instance_info = requests.post(F"{CLOUD_INSTANCE_SERVICE_URL}/create_instance", json=instance_payload).json()
        # response include instance_id, instance_ip, instance_port
        default_instance_id = 13636552
        instance_info = requests.get(f"{CLOUD_INSTANCE_SERVICE_URL}/instances/{default_instance_id}").json()
        print(instance_info)
        # return {"status": "success", "instance_info": instance_info}
        
        train_response = execute_training_process(saved_model_url, dataset_url, dataset_label_url, train_script_url, request["training_time"], request["presets"], task_id, instance_info)
        
        print(train_response)
        
        return {"status": "success", "instance_info": train_response}
        
        # shutdown_response = requests.post(F"{CLOUD_INSTANCE_SERVICE_URL}/shutdown_instance", json={"instance_id": instance_info["id"]}).json()
        
        # print(shutdown_response)
        
        end = perf_counter()
        
        return {
            "training_evaluation_time": end - start,
            "saved_model_path": saved_model_url,
        }

    except Exception as e:
        print(e)
        return {"error": str(e)}


def execute_training_process(saved_model_url, dataset_url, dataset_label_url, train_script_url, training_time, presets, task_id, instance_info: dict):
    # Define your connection details (get from instance service)
    ip = instance_info["public_ip"]
    port = instance_info["ssh_port"]
    username = "root"

    # Initialize the SSH client
    ssh_client = paramiko.SSHClient()

    # Automatically add the server's host key (be cautious about this in production)
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    private_key_path = generate_ssh_key_pair(task_id)
    
    # Attach the public key to the instance
    attach_response = attach_ssh_key_to_instance(task_id, instance_info["id"])
    print(attach_response)
    # Connect to the remote server with custom port
    try:
        ssh_client = connect_with_retries(ip, port, username, private_key_path, max_retries=10, delay=5)
    except Exception as e:
        return {"status": "error", "error": str(e)}
    
    # screen, nohup
    # # set up libs
    stdin, stdout, stderr = ssh_client.exec_command(f"sudo apt-get install screen unzip nano zsh htop default-jre zip -y")
    # print("Output: \n", stdout.read().decode())
    print("Errors:", stderr.read().decode())

    # pull dataset
    stdin, stdout, stderr = ssh_client.exec_command(f"wget -O train_script.zip '{train_script_url}'")
    print("Errors:", stderr.read().decode())

    stdin, stdout, stderr = ssh_client.exec_command(f"unzip train_script.zip")
    print("Errors:", stderr.read().decode())

    activate_env_command = "source /opt/conda/bin/activate base"

    stdin, stdout, stderr = ssh_client.exec_command(f"{activate_env_command} && pip install -r requirements.txt")
    print("Output: \n", stdout.read().decode())
    print("Errors:", stderr.read().decode())


    stdin, stdout, stderr = ssh_client.exec_command(f"{activate_env_command} && source train.sh '{dataset_url}' '{dataset_label_url}' '{training_time}' '{presets}' 'trained_model.zip' '{saved_model_url}'")
    print("Output: \n", stdout.read().decode())
    print("Errors:", stderr.read().decode())


    stdin, stdout, stderr = ssh_client.exec_command(f"source cleanup.sh")
    print("Output: \n", stdout.read().decode())
    print("Errors:", stderr.read().decode())

    # Close the connection
    ssh_client.close()
    return {"status": "success"}