from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings

# Placeholder for router imports
# from app.routers import user_router, item_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup: Initialize Database, load models, etc.
    print(f"Application startup: Initializing Database at {settings.DATABASE_URL}...")
    yield
    # Teardown: Close database connections, clean up resources
    print("Application shutdown: Cleaning up...")

app = FastAPI(lifespan=lifespan, title="ParkGuard API")

@app.get("/")
async def root():
    return {"status": "ok"}
