import signal
import os, sys

from fastapi.responses import FileResponse
from traitlets import default

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
from settings import config
from fastapi import FastAPI, Form
from settings.config import celery_client

from model_service.routes import router as model_service_router
from label_service.routes import router as label_service_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(debug=True)

origins = [
    "http://localhost:3000",
    "http://localhost:8673",
    "http://localhost:8674",
    "http://localhost:8675",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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

app.include_router(
    label_service_router, prefix="/label_service", tags=["label_service"]
)


@app.get("/public/media/{file_path}")
def read_file(file_path: str):
    return FileResponse(f"./public/media/{file_path}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=int(config.PORT),
        reload=True,
    )
