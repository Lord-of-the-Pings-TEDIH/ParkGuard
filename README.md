# ParkGuard

FastAPI backend for Romanian license-plate detection and parking ticket validation.

## Backend setup

Use Python 3.12 (recommended for `paddlepaddle` compatibility).

```bash
python3.12 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Set `DATABASE_URL` in `.env` before starting the API.

## Run API

```bash
venv/bin/uvicorn main:app --reload
```

API routes are mounted under:
- `/api/sessions`
- `/api/plates`
- `/api/parking`

Model/OCR defaults from `.env.example`:
- `MODEL_PATH=./models/best.pt`
- PaddleOCR + robust OCR/tracking knobs (`OCR_*`, `TRACK_*`, `MIN_TRACK_VOTES`)

## Typical processing flow

1. Upload a video: `POST /api/sessions` (multipart field: `file`)
2. Process it: `POST /api/sessions/{session_id}/process`
3. Read detections: `GET /api/sessions/{session_id}/detections`
4. Read observed plates: `GET /api/plates`

## Tests

```bash
# full backend suite
venv/bin/python -m pytest tests -q

# single test example
venv/bin/python -m pytest tests/test_ticket_lookup.py::test_active_ticket -q
```

## Frontend (Vite)

```bash
cd frontend
npm install
npm run dev
```

For production build:

```bash
cd frontend
npm run build
```

## Seed / reset database

Seed test parking data:

```bash
# bash/Linux/macOS
docker exec -i -e PGPASSWORD=parkguard parkguard-db psql -U parkguard -d parkguard < seed/parking_data.sql

# Windows PowerShell
Get-Content seed/parking_data.sql | docker exec -i -e PGPASSWORD=parkguard parkguard-db psql -U parkguard -d parkguard
```

Reset schema:

```bash
docker exec -i -e PGPASSWORD=parkguard parkguard-db psql -U parkguard -d parkguard -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
```

After reset, restart API so `create_all` runs on startup, then seed again.
