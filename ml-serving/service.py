from __future__ import annotations

from numpy import ndarray
from pydantic import BaseModel, Field
from typing import List
from PIL import Image as PILImage
import typing as t
import bentoml
from utils.requests import DeployRequest, ImagePredictionRequest, ImageExplainRequest, TextPredictionRequest, TextExplainRequest, BaseRequest, TabularExplainRequest, TabularPredictionRequest
from utils.preprocess_data import preprocess_image, softmax, preprocess_text, combine_extra_request_fields, preprocess_tabular
from time import time
import onnx
import onnxruntime as ort
from explainers.ImageExplainer import ImageExplainer
from explainers.TextExplainer import TextExplainer
from explainers.TabularExplainer import TabularExplainer
from settings import FRONTEND_URL, BACKEND_URL, ML_SERVICE_URL, INFERENCE_SERVICE_PORT
from services.base_service import BaseService
import uuid
from time import perf_counter
import base64
import pandas as pd
import numpy as np
import os

@bentoml.service(
    resources={
        "cpu": "1",
    },
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
    async def predict(self, params: ImagePredictionRequest) -> dict:
        
        start_load = perf_counter()
        # FIX THIS
        await self.check_already_deploy(params)
        predictions = []
        
        try:
            data = preprocess_image(params.images)
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
    
    @bentoml.api()
    async def explain(self, params: ImageExplainRequest) -> dict:

        start_load = perf_counter()
        # FIX THIS
        await self.check_already_deploy(params)
        os.makedirs("./tmp", exist_ok=True)

        try:
            explainer = ImageExplainer(params.method, self.ort_sess, num_samples=100, batch_size=32)
            load_time = perf_counter() - start_load
            
            inference_start = perf_counter()
            image_explained_path = explainer.explain(params.image, "./tmp")
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
    async def deploy(self, params: DeployRequest) -> dict:
        response = super().deploy(params)
        return response
    

    @bentoml.api()
    async def predict(self, params: TextPredictionRequest) -> dict:
        
        start_load = perf_counter()
        # FIX THIS
        await self.check_already_deploy(params)
        predictions = []
        try:
            data = preprocess_text(params.text_file_path, params.text_col)
            text_df = pd.read_csv(params.text_file_path)
            load_time = perf_counter() - start_load
            
            inference_start = perf_counter()
            probas = self.predict_proba(data)
            
            for i, proba in enumerate(probas):
                predictions.append(
                    {
                        "sentence": text_df[params.text_col].values[i],
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
    
    @bentoml.api()
    async def explain(self, params: TextExplainRequest) -> dict:
        
        start_load = perf_counter()
        # FIX THIS
        await self.check_already_deploy(params)
        print(self.model_metadata["class_names"])
        try:
            explainer = TextExplainer(params.method, self.ort_sess, class_names=self.model_metadata["class_names"])
            load_time = perf_counter() - start_load
            inference_start = perf_counter()
            
            explanations = explainer.explain(params.text)
            inference_time = perf_counter() - inference_start
            
        except Exception as e:
            print(e)
            print("Prediction failed")
            return {"status": "failed", "message": "Explanation failed"}

        return {
            "status": "success",
            "message": "Explanation completed",
            "load_time": load_time,
            "inference_time": inference_time,
            "explanation": explanations,
        }
    

@bentoml.service(
    resources={"cpu": "1"},
    traffic={"timeout": 500},
)
class TabularClassifyService(BaseService):
    def __init__(self) -> None:
        super().__init__()

    def predict_proba(self, data):
        predictions = self.ort_sess.predict_proba(data, as_pandas=False, as_multiclass=True)
        return predictions


    @bentoml.api()
    async def deploy(self, params: DeployRequest) -> dict:
        response = super().deploy(params)
        return response
    

    @bentoml.api()
    async def predict(self, params: TabularPredictionRequest) -> dict:
        
        start_load = perf_counter()
        # FIX THIS
        await self.check_already_deploy(params)
        predictions = []
        try:
            data = pd.read_csv(params.tab_file_path)
            # data = preprocess_tabular(data)
            load_time = perf_counter() - start_load
            
            inference_start = perf_counter()
            probas = self.predict_proba(data)
            for i, proba in enumerate(probas):
                predictions.append(
                    {
                        "key": str(uuid.uuid4()),
                        "class": str(self.ort_sess.class_labels[np.argmax(proba)]),
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
    
    @bentoml.api()
    async def explain(self, params: TabularExplainRequest) -> dict:
        
        start_load = perf_counter()
        # FIX THIS
        await self.check_already_deploy(params)
        
        data = pd.read_csv(params.tab_explain_file_path)
        data = preprocess_tabular(data)
        
        sample_data_path = f"../tmp/{params.userEmail}/{params.projectName}/trained_models/ISE/{params.runName}/sample_data.csv"
        
        try:
            explainer = TabularExplainer(params.method, self.ort_sess, class_names=self.ort_sess.class_labels, num_samples=100, sample_data_path=sample_data_path)
            load_time = perf_counter() - start_load
            inference_start = perf_counter()
            
            explanations = explainer.explain(data)
            inference_time = perf_counter() - inference_start
            
        except Exception as e:
            print(e)
            print("Prediction failed")
            return {"status": "failed", "message": "Explanation failed"}

        return {
            "status": "success",
            "message": "Explanation completed",
            "load_time": load_time,
            "inference_time": inference_time,
            "explanation": explanations,
        }


@bentoml.service(
    resources={"cpu": "1"},
    traffic={"timeout": 500},
)
class MultiModalClassifyService(BaseService):
    def __init__(self) -> None:
        super().__init__()

    def predict_proba(self, data):
        predictions = self.ort_sess.predict_proba(data, as_pandas=False, as_multiclass=True)
        print(type(predictions))
        return predictions


    @bentoml.api()
    async def deploy(self, params: DeployRequest) -> dict:
        response = super().deploy(params)
        return response
    

    @bentoml.api()
    async def predict(self, params: TabularPredictionRequest) -> dict:
        
        start_load = perf_counter()
        # FIX THIS
        await self.check_already_deploy(params)
        predictions = []
        try:
            data = pd.read_csv(params.tab_file_path)
            # data = preprocess_multimodal(data)
            load_time = perf_counter() - start_load
            
            inference_start = perf_counter()
            probas = self.predict_proba(data)
            for i, proba in enumerate(probas):
                predictions.append(
                    {
                        "key": str(uuid.uuid4()),
                        "class": str(self.ort_sess.class_labels[np.argmax(proba)]),
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
    
    @bentoml.api()
    async def explain(self, params: TabularExplainRequest) -> dict:
        
        return {"status": "Not implemented yet"}
        start_load = perf_counter()
        # FIX THIS
        await self.check_already_deploy(params)
        
        data = pd.read_csv(params.tab_explain_file_path)
        # data = preprocess_multimodal(data)
        
        try:
            explainer = MultiModalExplainer(params.method, self.ort_sess, class_names=self.ort_sess.class_labels, num_samples=100, sample_data_path=sample_data_path)
            load_time = perf_counter() - start_load
            inference_start = perf_counter()
            
            explanations = explainer.explain(data)
            inference_time = perf_counter() - inference_start
            
        except Exception as e:
            print(e)
            print("Prediction failed")
            return {"status": "failed", "message": "Explanation failed"}

        return {
            "status": "success",
            "message": "Explanation completed",
            "load_time": load_time,
            "inference_time": inference_time,
            "explanation": explanations,
        }
    



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

    # img_classify_service = bentoml.depends(ImageClassifyService)
    # text_classify_service = bentoml.depends(TextClassifyService)
    tabular_classify_service = bentoml.depends(TabularClassifyService)
    multimodal_classify_service = bentoml.depends(MultiModalClassifyService)
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
            res = self.text_classify_service.test_res(params["params"])
        elif req.task == "TABULAR_CLASSIFICATION":
            res = self.tabular_classify_service.test_res(params["params"])
            
        return res
    
    
    @bentoml.api(route="/deploy")
    async def deploy(self, req: DeployRequest):
        match(req.task):
            case "IMAGE_CLASSIFICATION":
                res = await self.img_classify_service.deploy(req)
            case "TEXT_CLASSIFICATION":
                res = await self.text_classify_service.deploy(req)
            case "TABULAR_CLASSIFICATION":
                res = await self.tabular_classify_service.deploy(req)
        return res
    
    @bentoml.api(route="/predict")
    async def predict(self, req: BaseRequest) -> dict:
        
        combined_params = combine_extra_request_fields(req)
        
        match(req.task):
            case "IMAGE_CLASSIFICATION":
                res = await self.img_classify_service.predict(combined_params["params"])
            case "TEXT_CLASSIFICATION":
                res = await self.text_classify_service.predict(combined_params["params"])
            case "TABULAR_CLASSIFICATION":
                res = await self.tabular_classify_service.predict(combined_params["params"])
            case "MULTIMODAL_CLASSIFICATION":
                res = await self.multimodal_classify_service.predict(combined_params["params"])
                
        return res
    
    @bentoml.api(route="/explain")
    async def explain(self, req: BaseRequest) -> dict:
        
        combined_params = combine_extra_request_fields(req)
        
        match(req.task):
            case "IMAGE_CLASSIFICATION":
                res = await self.img_classify_service.explain(combined_params["params"])
            case "TEXT_CLASSIFICATION":
                res = await self.text_classify_service.explain(combined_params["params"])
            case "TABULAR_CLASSIFICATION":
                res = await self.tabular_classify_service.explain(combined_params["params"])
            case "MULTIMODAL_CLASSIFICATION":
                res = await self.multimodal_classify_service.explain(combined_params["params"])
        return res