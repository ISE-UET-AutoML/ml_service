import time
from celery import shared_task
from . import image_classify
from . import tabular_classify


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
    return image_classify.train(self.request.id, request)


# @shared_task(
#     bind=True,
#     name="model_service.image_classify.predict",
# )
# def image_classify_predict(self, request: dict):
#     return image_classify.predict(self.request.id, request)


@shared_task(
    bind=True,
    name="model_service.tabular_classify.train",
)
def tabular_classify_train(self, request: dict):
    return tabular_classify.train(self.request.id, request)


# @shared_task(
#     bind=True,
#     name="model_service.tabular_classify.predict",
# )
# def tabular_classify_predict(self, request: dict):
#     return {}
#     # tabular_classify.tabular_predict(task_id, request)
