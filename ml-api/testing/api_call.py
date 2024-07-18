import celery
from config import celery_client

task_id = "43c90c90-ad48-4654-b62a-ad5768caebeb"
celery_client.control.revoke(task_id, terminate=True, signal="SIGKILL")
