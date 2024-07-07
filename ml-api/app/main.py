import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import celery
from settings import config
from fastapi import FastAPI

from model_service.routes import router as model_service_router

app = FastAPI(debug=True)


@app.get("/")
def read_root():
    return {"Hello": "World"}


app.include_router(
    model_service_router, prefix="/model_service", tags=["model_service"]
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=int(config.PORT),
        reload=True,
    )
