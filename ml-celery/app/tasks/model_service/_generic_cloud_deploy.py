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
from utils.aws import create_presigned_url, create_presigned_post, get_training_script_url, get_inference_script_url
from utils.ssh_utils import generate_ssh_key_pair, attach_ssh_key_to_instance, get_private_key_filename, connect_with_retries
import os
from pathlib import Path
import requests
from settings.config import CLOUD_INSTANCE_SERVICE_URL, REALTIME_INFERENCE_PORT
import paramiko

def deploy(request: dict):
    print("Cloud Deploy request received")
    start = perf_counter()
    try:
        # defines necessary urls
        # model_bucket_path = f"{request['userID']}/{request['projectID']}/{task_id}/trained_model.zip"
        # saved_model_url = create_presigned_post(model_bucket_path)
        # dataset_url = request["dataset_url"]
        # dataset_label_url = request["dataset_label_url"]
        # train_script_url = get_training_script_url(request['task'])
        
        model_bucket_path = f"{request['project_id']}/{request['task_id']}/trained_model.zip"
        saved_model_url = create_presigned_post(model_bucket_path)
        inference_script_url = get_inference_script_url(request['task'], request['deploy_type'])
        
        # create cloud instance
        
        # instance_payload = {
        #     "task": request["task"],
        #     "training_time": request["training_time"],
        #     "presets": request["presets"],
        # }
        
        # instance_info = requests.post(F"{CLOUD_INSTANCE_SERVICE_URL}/create_instance", json=instance_payload).json()
        # # response include instance_id, instance_ip, instance_port
        
        default_instance_id = 13636552
        instance_info = requests.get(f"{CLOUD_INSTANCE_SERVICE_URL}/instances/{default_instance_id}").json()
        print(instance_info)
        
        
        deploy_response = execute_deploy_process(saved_model_url, inference_script_url, request["task_id"], instance_info)
        
        print(deploy_response)
        
        # shutdown_response = requests.post(F"{CLOUD_INSTANCE_SERVICE_URL}/shutdown_instance", json={"instance_id": instance_info["id"]}).json()
        
        # print(shutdown_response)
        
        end = perf_counter()
        
        return {
            "deployed_model_url": f"{instance_info['public_ip']}:{instance_info['host_port']}",
            "deploy_time": end - start,
        }

    except Exception as e:
        print(e)
        return {"error": str(e)}


def execute_deploy_process(saved_model_url, infer_script_url, task_id, instance_info: dict):
    # Define your connection details (get from instance service)
    ip = instance_info["public_ip"]
    port = instance_info["ssh_port"]
    username = "root"
    
    
    
    private_key_path = get_private_key_filename(task_id)
    
    # Attach the public key to the instance
    attach_response = attach_ssh_key_to_instance(task_id, instance_info["id"])
    
    print(attach_response)

    try:
        ssh_client = connect_with_retries(ip, port, username, private_key_path, max_retries=10)
    except Exception as e:
        print(e)
        return {"error": str(e)}

    # screen, nohup
    # # set up libs
    stdin, stdout, stderr = ssh_client.exec_command(f"sudo apt-get install screen unzip nano zsh htop default-jre zip -y")
    # print("Output: \n", stdout.read().decode())
    print("Errors:", stderr.read().decode())

    # pull dataset
    stdin, stdout, stderr = ssh_client.exec_command(f"wget -O infer_script.zip '{infer_script_url}'")
    print("Errors:", stderr.read().decode())

    stdin, stdout, stderr = ssh_client.exec_command(f"unzip infer_script.zip")
    print("Errors:", stderr.read().decode())

    activate_env_command = "source /opt/conda/bin/activate base"


    stdin, stdout, stderr = ssh_client.exec_command(f"{activate_env_command} && source setup.sh '{saved_model_url}' '{REALTIME_INFERENCE_PORT}'")
    print("Output: \n", stdout.read().decode())
    print("Errors:", stderr.read().decode())

    # Close the connection
    ssh_client.close()
    return {"status": "success"}