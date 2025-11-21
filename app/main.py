from fastapi import FastAPI
from .utils.env import load_env
from .api.routes import router

load_env()
app = FastAPI()
app.include_router(router)