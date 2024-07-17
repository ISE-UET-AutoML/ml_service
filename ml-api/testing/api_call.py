import celery
from config import celery_client

task_id = "06c61e0f-0875-4df1-bad2-65d67d95d648"
celery_client.control.revoke(task_id, terminate=True)
