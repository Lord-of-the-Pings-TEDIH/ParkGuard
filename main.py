from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.database import init_db

# Ensure every model is imported so Base.metadata is complete
import app.models

from app.routers.sessions import router as sessions_router

# Placeholder for router imports
# from app.routers import user_router, item_router

# Create required directories before mounting static files
Path(settings.CROPS_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup: create all tables on startup
    try:
        print(f"Application startup: connecting to {settings.DATABASE_URL} ...")
        await init_db()
        print("Database tables created successfully.")
    except Exception as e:
        print(f"Warning: Database initialization failed: {e}")
    yield
    # Teardown
    from app.core.database import async_engine
    await async_engine.dispose()
    print("Application shutdown: database connections closed.")

app = FastAPI(lifespan=lifespan, title="ParkGuard API")
app.mount("/crops", StaticFiles(directory=settings.CROPS_DIR), name="crops")
app.include_router(sessions_router)

@app.get("/")
async def root():
    return {"status": "ok"}
