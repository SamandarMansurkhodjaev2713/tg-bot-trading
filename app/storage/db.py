import os
from sqlalchemy import create_engine

def get_engine():
    url = os.getenv("POSTGRES_URL", "sqlite:///forex.db")
    return create_engine(url, pool_pre_ping=True)