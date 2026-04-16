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

Each detection now includes a `voting_tag`:
- `not_final` while track voting is still in progress
- `final` once voting converges and ticket status is resolved

Plate normalization uses color hints (`standard` / `temporary` / `diplomatic` / `probe` / `electric`).
County-prefixed raw strings with **5-6 trailing digits** are treated as special numeric families
(temporary-style), not coerced into standard 3-letter tails.
Temporary county formats are constrained to **exactly 5 or 6 digits** (4-digit variants are rejected).
Any plate normalized as `INVALID: ...` is excluded from voting.
Detection payload also includes `plate_annotation`; frontend groups by this annotation
and shows a single live card per plate (with `Seen: Nx`), preventing duplicates.
Session `total_frames` now reflects sampled frames at `fps_target` (progress bar tracks
actual processing workload, not raw source frame count).

## Standalone pipeline flow debugger

For step-by-step visualization (detect -> crop -> deskew -> OCR -> voting),
use the standalone script (it does not import from `app/...`):

```bash
venv/bin/python pipeline_flow_debugger.py \
  --video uploads/<your_video>.mp4 \
  --model models/best.pt \
  --out runs/pipeline_debug.mp4
```

Useful flags:
- `--show-window` for live preview (if OpenCV has GUI support)
- `--no-ocr` to inspect detection/tracking only
- `--max-sampled-frames 100` for a short debug run
- `--single-pass-ocr` to disable default multi-pass OCR preprocessing
- `--skip-ocr-when-stable` to mimic production optimization (otherwise OCR runs on every sample)

For hardcoded test videos already present in `uploads/`:

- List available files: `GET /api/sessions/test-files`
- Create a session from one test file: `POST /api/sessions/test-files/{filename}`

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
