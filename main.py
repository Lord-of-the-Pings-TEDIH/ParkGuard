from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.core.database import create_tables

# Ensure every model is imported so Base.metadata is complete
import app.models

# Placeholder for router imports
# from app.routers import user_router, item_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup: create all tables on startup
    print(f"Application startup: connecting to {settings.DATABASE_URL} ...")
    await create_tables()
    print("Database tables created successfully.")
    yield
    # Teardown
    from app.core.database import async_engine
    await async_engine.dispose()
    print("Application shutdown: database connections closed.")

app = FastAPI(lifespan=lifespan, title="ParkGuard API")

@app.get("/")
async def root():
    return {"status": "ok"}
