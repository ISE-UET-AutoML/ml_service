from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List
from PIL import Image as PILImage
import typing as t
import bentoml
from utils.requests import DeployRequest, ImagePredictionRequest
from utils.preprocess_data import preprocess_image, softmax
from time import time
import onnx
import onnxruntime as ort




@bentoml.service(
    resources={"cpu": "1"},
    traffic={"timeout": 10},
    http={
        "port": 8684,
        "cors": {
            "enabled": True,
            "access_control_allow_origins": ["http://localhost:8685", "https://localhost:8680"],
            "access_control_allow_methods": ["GET", "OPTIONS", "POST", "HEAD", "PUT"],
            "access_control_allow_credentials": True,
            "access_control_allow_headers": ["*"],
            "access_control_allow_origin_regex": "https://.*\.my_org\.com",
            "access_control_max_age": 1200,
            "access_control_expose_headers": ["Content-Length"]
        }
    }
    
)
class Test:
    def __init__(self) -> None:
        print("Init")


    @bentoml.api(input_spec=DeployRequest)
    def deploy(self, **params: t.Any) -> str:

        userEmail = params["userEmail"]
        projectName = params["projectName"]
        runName = params["runName"]

        try:
            self.ort_sess = ort.InferenceSession(f'../tmp/{userEmail}/{projectName}/trained_models/ISE/{runName}/model.onnx')
            self.input_names = [input.name for input in self.ort_sess.get_inputs()]
            print("Model deploy successfully")
        except Exception as e:
            print(e)
            return "Model deploy failed"
        return "Model deploy successfully"

    @bentoml.api(input_spec=ImagePredictionRequest)
    def predict(self, **params: t.Any) -> str:
        start_time = time()
        try:
            image_tensor, valid_nums = preprocess_image(params['images'])
            _, logits = self.ort_sess.run(None, {self.input_names[0]: image_tensor, self.input_names[1]: valid_nums})
            predictions = softmax(logits)
            print(predictions)
        except Exception as e:
            print(e)
            print("Prediction failed")
            return "Prediction failed"
        end_time = time()
        print(f"Time taken: {end_time - start_time}")

        return "success"