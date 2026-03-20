from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.core.database import init_db

# Ensure every model is imported so Base.metadata is complete
import app.models

from app.routers.sessions import router as sessions_router
from app.routers.plates import router as plates_router
from app.routers.parking import router as parking_router

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

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    if exc.status_code == 404:
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    # Default behavior for other HTTP exceptions: maintain {detail: string}
    detail = getattr(exc, "detail", "Unknown error")
    return JSONResponse({"detail": detail}, status_code=exc.status_code)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    errors = []
    for error in exc.errors():
        loc = ".".join([str(l) for l in error["loc"]])
        msg = error["msg"]
        errors.append(f"{loc}: {msg}")
    detail = "; ".join(errors)
    return JSONResponse({"detail": detail}, status_code=422)

app.mount("/crops", StaticFiles(directory=settings.CROPS_DIR), name="crops")
app.include_router(sessions_router, prefix="/sessions", tags=["sessions"])
app.include_router(plates_router, prefix="/plates", tags=["plates"])
app.include_router(parking_router, prefix="/parking", tags=["parking"])


class HealthCheckOut(BaseModel):
    status: str

@app.get("/", response_model=HealthCheckOut)
async def root():
    return HealthCheckOut(status="ok")
