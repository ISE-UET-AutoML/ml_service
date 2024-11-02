from __future__ import annotations

from numpy import ndarray
from pydantic import BaseModel, Field
from typing import List
from PIL import Image as PILImage
import typing as t
import bentoml
from utils import DeployRequest, ImagePredictionRequest, ImageExplainRequest
from utils import preprocess_image, softmax
from time import time
import onnx
import onnxruntime as ort
from explainer import ImageExplainer
from base_service import BaseService
import uuid
from time import perf_counter
import base64
import pandas as pd
import numpy as np
import os

@bentoml.service(
    resources={"cpu": "1", "gpu": 1},
    traffic={"timeout": 500},
    http={
        "port": int(os.environ["INFERENCE_SERVICE_PORT"]),
        "cors": {
            "enabled": True,
            "access_control_allow_origins": ["*"],
            "access_control_allow_methods": ["GET", "OPTIONS", "POST", "HEAD", "PUT"],
            "access_control_allow_credentials": True,
            "access_control_allow_headers": ["*"],
            "access_control_max_age": 1200,
            "access_control_expose_headers": ["Content-Length"]
        }
    },
)
class ImageClassifyService(BaseService):
    def __init__(self) -> None:
        super().__init__()

    def predict_proba(self, data):
        image_tensor, valid_nums = data
        _, logits = self.ort_sess.run(None, {self.input_names[0]: image_tensor, self.input_names[1]: valid_nums})
        predictions = softmax(logits)
        print(len(predictions))
        return predictions


    @bentoml.api(route="/deploy", input_spec=DeployRequest)
    async def temp_deploy(self, **params: t.Any) -> dict:
        response = super().deploy(params)

        return response
    

    @bentoml.api(route="/predict", input_spec=ImagePredictionRequest)
    async def predict(self, **params: t.Any) -> dict:
        
        print(params)
        start_load = perf_counter()
        # FIX THIS
        await self.check_already_deploy(params)
        predictions = []
        
        try:
            data = preprocess_image(params["images"])
            load_time = perf_counter() - start_load
            
            inference_start = perf_counter()
            probas = self.predict_proba(data)
            
            for proba in probas:
                predictions.append(
                {
                    "key": str(uuid.uuid4()),
                    "class": self.model_metadata["class_names"][np.argmax(proba)],
                    "confidence": round(float(max(proba)), 2),
                }
            )
                
        except Exception as e:
            print(e)
            print("Prediction failed")
            return {"status": "failed", "message": "Prediction failed"}

        return {
            "status": "success",
            "message": "Prediction completed",
            "load_time": load_time,
            "inference_time": perf_counter() - inference_start,
            "predictions": predictions,
        }
    
    @bentoml.api(route="/explain", input_spec=ImageExplainRequest)
    async def explain(self, **params: t.Any) -> dict:

        start_load = perf_counter()
        # FIX THIS
        await self.check_already_deploy(params)
        os.makedirs("./tmp", exist_ok=True)

        try:
            explainer = ImageExplainer(params["method"], self.ort_sess, num_samples=100, batch_size=32)
            load_time = perf_counter() - start_load
            
            inference_start = perf_counter()
            image_explained_path = explainer.explain(params["image"], "./tmp")
            inference_time = perf_counter() - inference_start
            
            with open(image_explained_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            print(e)
            print("Prediction failed")
            return {"status": "failed", "message": "Explanation failed"}

        return {
            "status": "success",
            "message": "Explanation completed",
            "load_time": load_time,
            "inference_time": inference_time,
            "explanation": encoded_image,
        }


