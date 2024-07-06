from celery import shared_task
from . import image_classify
from . import tabular_classify


@shared_task(
    name="model_service.image_classify.train",
)
def image_classify_train(task_id: str, request: dict):
    return image_classify.train(task_id, request)


@shared_task(
    name="model_service.image_classify.predict",
)
def image_classify_predict(task_id: str, request: dict):
    return image_classify.predict(task_id, request)


@shared_task(
    name="model_service.tabular_classify.train",
)
def tabular_classify_train(task_id: str, request: dict):
    return tabular_classify.tabular_train(task_id, request)


@shared_task(
    name="model_service.tabular_classify.predict",
)
def tabular_classify_predict(task_id: str, request: dict):
    return {}
    # tabular_classify.tabular_predict(task_id, request)
