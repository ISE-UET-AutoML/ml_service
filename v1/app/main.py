import os
from fastapi import FastAPI, File, UploadFile
from contextlib import asynccontextmanager
import torch

import logging
from logging.handlers import TimedRotatingFileHandler

torch.cuda.empty_cache()
torch.set_float32_matmul_precision("medium")
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model_service.train import routes as training_service_routes

# ++++++++++++++++++++++++++++++++++++++++++++ DEFINE APP +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
app = FastAPI()
app = FastAPI(title="ml_service", openapi_url="/api/openapi.json", docs_url="/api/docs", redoc_url="/api/redoc")


@asynccontextmanager  # lifespan cua context, sau nay su dung de release resource sau khi serve xong model
def init():
    pass

@app.get("/")
def read_root():
    return {"Hello": "World"}

app.include_router(training_service_routes.router)


# @serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 8, "num_gpus": 1})
# @serve.ingress(app)
# class FastAPIWrapper:
#     pass

if __name__ == "__main__":
    # ray_app = FastAPIWrapper.bind()
    os.system("uvicorn main:app --reload")
    # os.system("serve run main:ray_app")
