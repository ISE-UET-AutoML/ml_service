import time
from unittest import result
from celery import shared_task
import requests
from . import _image_classify
from . import _tabular_classify
from . import _object_detection
from . import _generic_mm_prediction
from . import _image_segmentation
from . import _named_entity_recognition
from . import _text_text_semantic_matching
from . import _img_img_semantic_matching
from . import _time_series
from settings.config import BACKEND_HOST


@shared_task(
    bind=True,
    name="time_test",
)
def time_test(self, request: dict):
    time.sleep(10)
    return "time_test"


@shared_task(
    bind=True,
    name="model_service.image_classify.train",
)
def image_classify_train(self, request: dict):
    result = _image_classify.train(self.request.id, request)
    # if "saved_model_path" in result:
    #     res = requests.get(
    #         f"{BACKEND_HOST}/experiments/deploy/?experiment_name={self.request.id}",
    #         params={"model_path": result["saved_model_path"]},
    #     )
    #     print(res.text)
    return result


@shared_task(
    bind=True,
    name="model_service.tabular_classify.train",
)
def tabular_classify_train(self, request: dict):
    return _tabular_classify.train(self.request.id, request)


@shared_task(
    bind=True,
    name="model_service.object_detection.train",
)
def object_detection_train(self, request: dict):
    return _object_detection.train(self.request.id, request)


@shared_task(
    bind=True,
    name="model_service.generic_mm_prediction.train",
)
def generic_mm_prediction_train(self, request: dict):
    return _generic_mm_prediction.train(self.request.id, request)


@shared_task(
    bind=True,
    name="model_service.image_segmentation.train",
)
def image_segmentation_train(self, request: dict):
    return _image_segmentation.train(self.request.id, request)


@shared_task(
    bind=True,
    name="model_service.named_entity_recognition.train",
)
def named_entity_recognition_train(self, request: dict):
    return _named_entity_recognition.train(self.request.id, request)


@shared_task(
    bind=True,
    name="model_service.text_text_semantic_matching.train",
)
def text_text_semantic_matching_train(self, request: dict):
    return _text_text_semantic_matching.train(self.request.id, request)


@shared_task(
    bind=True,
    name="model_service.img_img_semantic_matching.train",
)
def img_img_semantic_matching_train(self, request: dict):
    return _img_img_semantic_matching.train(self.request.id, request)


@shared_task(
    bind=True,
    name="model_service.time_series.train",
)
def time_series_train(self, request: dict):
    return _time_series.train(self.request.id, request)


import asyncio


@shared_task(
    bind=True,
    name="model_service.generic_multimodal.temp_predict",
)
def generic_multimodal_temp_predict(self, request: dict):
    return asyncio.run(_generic_mm_prediction.predict(self.request.id, request))
