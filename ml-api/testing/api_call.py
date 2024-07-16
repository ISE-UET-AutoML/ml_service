import celery
from config import celery_client

task_id = "e5085630-2a9c-47cf-8b3f-10bac30b4d12"
celery_client.control.revoke(task_id, terminate=True)
