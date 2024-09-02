from __future__ import annotations

from numpy import ndarray
from pydantic import BaseModel, Field
from typing import List
from PIL import Image as PILImage
import typing as t
import bentoml
from utils.requests import DeployRequest, ImagePredictionRequest, ImageExplainRequest, TextPredictionRequest, TextExplainRequest
from utils.preprocess_data import preprocess_image, softmax
from time import time
import onnx
import onnxruntime as ort
from explainers.ImageExplainer import ImageExplainer
from settings import FRONTEND_URL, BACKEND_URL, ML_SERVICE_URL, IMG_CLASSIFY_SERVICE_PORT
from services.base_service import BaseService


@bentoml.service(
    resources={"cpu": "1"},
    traffic={"timeout": 100},
    http={
        "port": int(IMG_CLASSIFY_SERVICE_PORT),
        "cors": {
            "enabled": True,
            "access_control_allow_origins": [BACKEND_URL, FRONTEND_URL, ML_SERVICE_URL],
            "access_control_allow_methods": ["GET", "OPTIONS", "POST", "HEAD", "PUT"],
            "access_control_allow_credentials": True,
            "access_control_allow_headers": ["*"],
            "access_control_allow_origin_regex": "https://.*\.my_org\.com",
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


    @bentoml.api(input_spec=DeployRequest)
    def deploy(self, **params: t.Any) -> dict:
        response = super().deploy(**params)

        return response

    @bentoml.api(input_spec=ImagePredictionRequest, route="image_classification/predict")
    def predict(self, **params: t.Any) -> ndarray:

        # FIX THIS
        self.check_already_deploy(userEmail=params['userEmail'], projectName=params['projectName'], runName=params['runName'])
        start_time = time()
        try:
            data = preprocess_image(params['images'])
            predictions = self.predict_proba(data)
            print("Prediction successful")
        except Exception as e:
            print(e)
            print("Prediction failed")
            return "Prediction failed"
        end_time = time()
        print(f"Time taken: {end_time - start_time}")

        return predictions
    
    @bentoml.api(input_spec=ImageExplainRequest, route="image_classification/explain")
    def explain(self, **params: t.Any) -> dict:

        if not hasattr(self, 'ort_sess'):
            print("Model not preloaded, loading now!")
            self.deploy(userEmail=params['userEmail'], projectName=params['projectName'], runName=params['runName'])

        start_time = time()
        try:
            explainer = ImageExplainer(params['method'], self.ort_sess, num_samples=200, batch_size=32)
            explainer.explain(params['image'], params['image_explained_path'])
        except Exception as e:
            print(e)
            print("Prediction failed")
            return {"status": "failed", "message": "Explanation failed"}
        end_time = time()
        print(f"Time taken: {end_time - start_time}")

        return {"status": "success"}





@bentoml.service(
    resources={"cpu": "1"},
    traffic={"timeout": 100},
    http={
        "port": IMG_CLASSIFY_SERVICE_PORT,
        "cors": {
            "enabled": True,
            "access_control_allow_origins": [BACKEND_URL, FRONTEND_URL, ML_SERVICE_URL],
            "access_control_allow_methods": ["GET", "OPTIONS", "POST", "HEAD", "PUT"],
            "access_control_allow_credentials": True,
            "access_control_allow_headers": ["*"],
            "access_control_allow_origin_regex": "https://.*\.my_org\.com",
            "access_control_max_age": 1200,
            "access_control_expose_headers": ["Content-Length"]
        }
    },
    
    
)
class TextClassifyService(BaseService):
    def __init__(self) -> None:
        super().__init__()

    def predict_proba(self, data):
        # image_tensor, valid_nums = data
        # _, logits = self.ort_sess.run(None, {self.input_names[0]: image_tensor, self.input_names[1]: valid_nums})
        # predictions = softmax(logits)
        # print(len(predictions))
        # return predictions
        pass


    @bentoml.api(input_spec=DeployRequest)
    def deploy(self, **params: t.Any) -> dict:
        response = super().deploy(**params)
        return response

    @bentoml.api(input_spec=TextPredictionRequest, route="text_classification/predict")
    def predict(self, **params: t.Any) -> ndarray:

        # FIX THIS
        # self.check_already_deploy(userEmail=params['userEmail'], projectName=params['projectName'], runName=params['runName'])
        # start_time = time()
        # try:
        #     print("Prediction successful")
        # except Exception as e:
        #     print(e)
        #     print("Prediction failed")
        #     return "Prediction failed"
        # end_time = time()
        # print(f"Time taken: {end_time - start_time}")

        # return predictions
        pass
    
    @bentoml.api(input_spec=ImageExplainRequest, route="image_classification/explain")
    def explain(self, **params: t.Any) -> dict:

        # self.check_already_deploy(userEmail=params['userEmail'], projectName=params['projectName'], runName=params['runName'])

        # start_time = time()
        # try:
        #     explainer = ImageExplainer(params['method'], self.ort_sess, num_samples=200, batch_size=32)
        #     explainer.explain(params['image'], params['image_explained_path'])
        # except Exception as e:
        #     print(e)
        #     print("Prediction failed")
        #     return {"status": "failed", "message": "Explanation failed"}
        # end_time = time()
        # print(f"Time taken: {end_time - start_time}")

        # return {"status": "success"}
        pass