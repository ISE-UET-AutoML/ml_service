import signal
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
from settings import config
from fastapi import FastAPI, Form
from settings.config import celery_client

from model_service.routes import router as model_service_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(debug=True)

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post(
    "/terminate_task",
    description="Terminate long running celery task by id\nNot recommended",
)
def terminate_task(task_id: str = Form(default=""), sig: str = Form(default="SIGTERM")):

    kill_sig = "SIGTERM"  # SIGTERM = exit task subprocess only - expected behavior
    if sig == "SIGINT":  # SIGINT = cannot exit task
        kill_sig = signal.SIGINT
    elif sig == "SIGKILL":  # Not tested
        kill_sig = signal.SIGKILL
    elif sig == "SIGUSR1":  # SIGUSR1 = error ?
        kill_sig = signal.SIGUSR1
    elif sig == "SIGSTOP":
        kill_sig = signal.SIGSTOP
    elif sig == "SIGCONT":
        kill_sig = signal.SIGCONT

    celery_client.control.revoke(task_id, terminate=True, signal=kill_sig)
    return {"task_id": task_id, "status": "terminated"}


app.include_router(
    model_service_router,
    prefix="/model_service",  # tags=["model_service"]
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=int(config.PORT),
        reload=True,
    )
