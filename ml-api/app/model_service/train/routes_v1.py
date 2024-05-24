
import re
import uuid
from fastapi import APIRouter
from settings.config import celery_client
from helpers import time as time_helper
from .TrainRequest import TabularTrainRequest
router=APIRouter()

@router.post("/tabular_classification",tags=["tabular_classification"])
async def train_tabular_classification(request: TabularTrainRequest):
    # this can also train tabular regression, but might change in the future
    print("Tabular Classification Training request received")

    time=time_helper.now_utc()
    task_id=uuid.uuid5(uuid.NAMESPACE_OID,
                       request.userEmail+'_'+request.projectName+'_'+request.runName+'_'+str(time))

    celery_client.send_task(
        'model_service.train',
        kwargs={
            'task_id':task_id,
            'request':request.dict(),
        },
        queue='ml_celery'
        )
    return {
        'task_id':task_id,
        'send_status':'SUCCESS',
    }
