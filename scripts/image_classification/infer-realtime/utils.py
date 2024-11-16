from pathlib import Path
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import requests
from time import time
import pandas as pd
import json
from urllib.parse import urlparse

IMAGE_SIZE = 224
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Annotated


class BaseRequest(BaseModel):
    userEmail: str = Field(description="User Email")
    projectName: str = Field(description="Project name")
    runName: str = Field(description="Run name")
    task: str = Field(description="task to be performed")
    task_id: str = Field(description="Task ID")

class DeployRequest(BaseRequest):
    task: str = Field(description="task to be performed", default="IMAGE_CLASSIFICATION")

# IMAGE CLASSIFICATION
class ImagePredictionRequest(BaseRequest):
    images: List[str] = Field(description="Image file paths")
    task: str = Field(description="task to be performed", default="IMAGE_CLASSIFICATION")
    
class ImageExplainRequest(BaseRequest):
    image: str = Field(description="Image file path")
    method: str = Field(description="Method to explain the image", default="lime")
    task: str = Field(description="task to be performed", default="IMAGE_CLASSIFICATION")


# Apply softmax to each row (each sample's logits)
def softmax(logits):
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def preprocess_image(data):
    # Preprocessing function
    transform = transforms.Compose([
        transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),  # Adjust size as needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def preprocess_single_image(image_input):
        if isinstance(image_input, str):
            if image_input.startswith("http"):
                img = Image.open(requests.get(image_input, stream=True).raw)
            else:
                img = Image.open(image_input)
        else:
            img = Image.fromarray(image_input, 'RGB')
        img = transform(img)
        return img

    start_time = time()
    images = [preprocess_single_image(image_input) for image_input in data]

    print(f"Preprocess time: {time() - start_time}")
    image_tensor = np.stack(images)
    valid_nums = np.array([1] * len(images), dtype=np.int64) # Create the timm_image_image_valid_num tensor
    
    image_tensor = np.expand_dims(image_tensor, axis=1)
    return image_tensor, valid_nums


def get_image_filename(url):
    parsed_url = urlparse(url)
    return os.path.basename(parsed_url.path)



