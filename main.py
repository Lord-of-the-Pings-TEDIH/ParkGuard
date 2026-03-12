from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.database import init_db

# Ensure every model is imported so Base.metadata is complete
import app.models

# Placeholder for router imports
# from app.routers import user_router, item_router

# Create required directories before mounting static files
Path(settings.CROPS_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup: create all tables on startup
    print(f"Application startup: connecting to {settings.DATABASE_URL} ...")
    await init_db()
    print("Database tables created successfully.")
    yield
    # Teardown
    from app.core.database import async_engine
    await async_engine.dispose()
    print("Application shutdown: database connections closed.")

app = FastAPI(lifespan=lifespan, title="ParkGuard API")
app.mount("/crops", StaticFiles(directory=settings.CROPS_DIR), name="crops")

@app.get("/")
async def root():
    return {"status": "ok"}
