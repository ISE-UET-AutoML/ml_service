
from ast import arg
import queue
import time
import celery
from settings import config
from fastapi import FastAPI


app=FastAPI()
celery_client=celery.Celery(broker=config.BROKER,backend=config.REDIS_BACKEND)
if not celery_client:
    exit()

@app.get("/")
def read_root():
    return {"Hello":"World"}

@app.post("/test_train")
def test_train():
    print("test train")
    result = celery_client.send_task(
        'model_service.train',
        kwargs={'task_id':'123'},
        queue='ml_celery'
        )
    time.sleep(10)
    print("result: ",result)
    return {"status":"success"}
    #return {"result":result}

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host=config.HOST,port=int(config.PORT))

