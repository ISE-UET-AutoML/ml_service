import time
from celery import shared_task
from . import _image_classify
from . import _tabular_classify
from . import _object_detection
from . import _generic_mm_prediction
from . import _image_segmentation


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
    return _image_classify.train(self.request.id, request)


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
