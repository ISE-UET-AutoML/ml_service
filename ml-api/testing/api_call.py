import celery
from config import celery_client

task_id = "71994d00-f59e-4b35-8a7d-09ce86e32fb1"
celery_client.control.revoke(task_id, terminate=True)
