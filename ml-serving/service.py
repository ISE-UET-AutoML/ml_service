from __future__ import annotations

from numpy import ndarray
from pydantic import BaseModel, Field
from typing import List
from PIL import Image as PILImage
import typing as t
import bentoml
from utils.requests import DeployRequest, ImagePredictionRequest, ImageExplainRequest, TextPredictionRequest, TextExplainRequest, BaseRequest
from utils.preprocess_data import preprocess_image, softmax, preprocess_text, combine_extra_request_fields
from time import time
import onnx
import onnxruntime as ort
from explainers.ImageExplainer import ImageExplainer
from explainers.TextExplainer import TextExplainer
from settings import FRONTEND_URL, BACKEND_URL, ML_SERVICE_URL, INFERENCE_SERVICE_PORT
from services.base_service import BaseService

@bentoml.service(
    resources={"cpu": "1"},
    traffic={"timeout": 500}, 
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


    @bentoml.api()
    async def deploy(self, params: DeployRequest) -> dict:
        response = super().deploy(params)

        return response
    
    @bentoml.api()
    def test_res(self, req: ImagePredictionRequest) -> dict:

        print(req)
        return {"status": "success", "task": "image_classification"}

    @bentoml.api()
    async def predict(self, params: ImagePredictionRequest) -> ndarray:

        # FIX THIS
        self.check_already_deploy(params)
        start_time = time()
        try:
            data = preprocess_image(params.images)
            predictions = self.predict_proba(data)
            print("Prediction successful")
        except Exception as e:
            print(e)
            print("Prediction failed")
            return "Prediction failed"
        end_time = time()
        print(f"Time taken: {end_time - start_time}")

        return predictions
    
    @bentoml.api()
    async def explain(self, params: ImageExplainRequest) -> dict:

        # FIX THIS
        self.check_already_deploy(params)

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
    traffic={"timeout": 500},
)
class TextClassifyService(BaseService):
    def __init__(self) -> None:
        super().__init__()

    def predict_proba(self, data):
        token_ids, segment_ids, valid_length = data
        _, logits = self.ort_sess.run(None, {self.input_names[0]: token_ids, self.input_names[1]: segment_ids, self.input_names[2]: valid_length})
        predictions = softmax(logits)
        print(len(predictions))
        return predictions


    @bentoml.api()
    def deploy(self, params: DeployRequest) -> dict:
        response = super().deploy(params)
        return response
    
    @bentoml.api()
    def test_res(self, params: TextPredictionRequest) -> dict:
        print(params)

        return {"status": "success", "task": "text_classification"}

    @bentoml.api()
    def predict(self, params: TextPredictionRequest) -> ndarray:

        # FIX THIS
        self.check_already_deploy(params)

        data = preprocess_text(params.text_file_path, params.text_col)
        try:
            predictions = self.predict_proba(data)
            print(predictions)
            print("Prediction successful")
        except Exception as e:
            print(e)
            print("Prediction failed")

        return predictions
    
    @bentoml.api()
    def explain(self, params: TextExplainRequest) -> dict:

        # FIX THIS
        self.check_already_deploy(**params)

        start_time = time()
        try:
            explainer = TextExplainer(params.method, self.ort_sess, class_names=params["class_names"])
            explanations = explainer.explain(params.text)
        except Exception as e:
            print(e)
            print("Prediction failed")
            return {"status": "failed", "message": "Explanation failed"}
        end_time = time()
        print(f"Time taken: {end_time - start_time}")

        return {"status": "success", "explanation": explanations}
    


@bentoml.service(
    resources={"cpu": "1"},
    traffic={"timeout": 500},
    http={
        "port": int(INFERENCE_SERVICE_PORT),
        "cors": {
            "enabled": True,
            "access_control_allow_origins": ["*"],
            "access_control_allow_methods": ["GET", "OPTIONS", "POST", "HEAD", "PUT"],
            "access_control_allow_credentials": True,
            "access_control_allow_headers": ["*"],
            "access_control_allow_origin_regex": "https://.*\.my_org\.com",
            "access_control_max_age": 1200,
            "access_control_expose_headers": ["Content-Length"]
        }
    },
)
class InferenceService:

    img_classify_service = bentoml.depends(ImageClassifyService)
    text_classify_service = bentoml.depends(TextClassifyService)
    def __init__(self):
        print("Init")
    
    @bentoml.api(route="/test")
    async def test(self, req: BaseRequest) -> dict:
        print(req)
        print(req.model_extra)
        
        params = combine_extra_request_fields(req)
        if req.task == "IMAGE_CLASSIFICATION":
            res = self.img_classify_service.test_res(params["params"])
        elif req.task == "TEXT_CLASSIFICATION":
            res = self.text_classify_service.test_res(combined_fields)
            
        return res
    
    
    @bentoml.api(route="deploy")
    async def deploy(self, req: DeployRequest):
        match(req.task):
            case "IMAGE_CLASSIFICATION":
                res = await self.img_classify_service.deploy(req)
            case "TEXT_CLASSIFICATION":
                res = await self.text_classify_service.deploy(req)
        return res
    
    @bentoml.api(route="/predict")
    async def predict(self, req: BaseRequest) -> dict:
        
        combined_params = combine_extra_request_fields(req)
        
        if req.task == "IMAGE_CLASSIFICATION":
            res = await self.img_classify_service.predict(combined_params["params"])
        elif req.task == "TEXT_CLASSIFICATION":
            res = await self.text_classify_service.predict(combined_params["params"])
        print(res)
        return {"status": "success"}