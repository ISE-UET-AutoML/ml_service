import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import celery
from settings import config
from fastapi import FastAPI

from model_service.routes import router as model_service_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(debug=True)

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
