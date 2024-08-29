from __future__ import annotations

from numpy import ndarray
from pydantic import BaseModel, Field
from typing import List
from PIL import Image as PILImage
import typing as t
import bentoml
from utils.requests import DeployRequest, ImagePredictionRequest, ImageExplainRequest
from utils.preprocess_data import preprocess_image, softmax
from time import time
import onnx
import onnxruntime as ort





@bentoml.service(
    resources={"cpu": "1"},
    traffic={"timeout": 100},
    http={
        "port": 8684,
        "cors": {
            "enabled": True,
            # TODO: fix hardcode by using env vars
            "access_control_allow_origins": ["http://localhost:8685", "https://localhost:8680", "http://localhost:8670"],
            "access_control_allow_methods": ["GET", "OPTIONS", "POST", "HEAD", "PUT"],
            "access_control_allow_credentials": True,
            "access_control_allow_headers": ["*"],
            "access_control_allow_origin_regex": "https://.*\.my_org\.com",
            "access_control_max_age": 1200,
            "access_control_expose_headers": ["Content-Length"]
        }
    },
    
    
)
class ImageClassificationService:
    def __init__(self) -> None:
        print("Init")


    def load_model_and_model_info(self, userEmail: str, projectName: str, runName: str) -> t.Tuple[t.Any, t.List[str]]:
        try:
            self.ort_sess = ort.InferenceSession(f'../tmp/{userEmail}/{projectName}/trained_models/ISE/{runName}/model.onnx')
            self.input_names = [input.name for input in self.ort_sess.get_inputs()]
            print("Model deploy successfully")
            return None
        except Exception as e:
            print(e)
    
    def temp_predict(self, image_tensor, valid_nums):
        _, logits = self.ort_sess.run(None, {self.input_names[0]: image_tensor, self.input_names[1]: valid_nums})
        predictions = softmax(logits)
        print(len(predictions))
        return predictions


    @bentoml.api(input_spec=DeployRequest)
    def deploy(self, **params: t.Any) -> str:

        userEmail = params["userEmail"]
        projectName = params["projectName"]
        runName = params["runName"]

        try:
            self.load_model_and_model_info(userEmail, projectName, runName)
        except Exception as e:
            print(e)
            print("Model deploy failed")
            return "Model deploy failed"
        return "Model deploy successfully"

    @bentoml.api(input_spec=ImagePredictionRequest, route="image_classification/predict")
    def predict(self, **params: t.Any) -> ndarray:

        if not hasattr(self, 'ort_sess'):
            self.deploy(userEmail=params['userEmail'], projectName=params['projectName'], runName=params['runName'])
            print("Model not preloaded, loading now!")
        start_time = time()
        try:
            image_tensor, valid_nums = preprocess_image(params['images'])
            predictions = self.temp_predict(image_tensor, valid_nums)
            print("Prediction successful")
        except Exception as e:
            print(e)
            print("Prediction failed")
            return "Prediction failed"
        end_time = time()
        print(f"Time taken: {end_time - start_time}")

        return predictions
    
    @bentoml.api(input_spec=ImageExplainRequest, route="image_classification/explain")
    def explain(self, **params: t.Any) -> str:
        start_time = time()
        try:
            image_tensor, valid_nums = preprocess_image(params['image'])
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
