# ParkGuard

FastAPI backend + Vite/React frontend for Romanian license-plate detection,
parking-ticket validation, and **GPS-based detection of illegal residential
spot use** (e.g. owners letting tenants park on a reserved spot).

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
- `MODEL_PATH=./models/best26s.pt` (the bundled YOLO weights — `.env.example`
  still references the older `best.pt`, so set this explicitly)
- PaddleOCR + robust OCR/tracking knobs (`OCR_*`, `TRACK_*`, `MIN_TRACK_VOTES`)
- Mobile-LPR / GPS knobs (`MOBILE_LPR_*`) — see the section below

## Typical processing flow

1. Upload a video: `POST /api/sessions` (multipart field: `file`).  GPS pose
   (`gps_latitude` / `gps_longitude` / `gps_heading_deg`) is optional — it is
   read from explicit form fields, then from the video container metadata
   (iOS `.MOV` files carry it automatically), then from the
   `MOBILE_LPR_DEFAULT_*` config defaults.
2. Process it: `POST /api/sessions/{session_id}/process`
3. Read detections: `GET /api/sessions/{session_id}/detections`
4. Read observed plates: `GET /api/plates`
5. Read suspicious occupancies: `GET /api/parking/spots/suspicious`
   (`?flagged_only=false` to include below-threshold rows)

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

## Mobile-LPR & residential spot-misuse detection

When a session has GPS pose set, the pipeline projects every plate's bounding
box to a GPS coordinate (`PlateGeolocationCalculator` in
`app/pipeline/geolocation.py`) and checks it against the registered
`ParkingSpot` rows.  A plate parked near a reserved spot whose
`assigned_plate` does not match the OCR result produces a `WRONG_PLATE`
verdict, and `record_occupancy_event` adds a row to
`spot_occupancy_events`.

The aggregated `spot_occupancy_records` table then powers the **Spot Misuse**
panel in the dashboard.  The score combines:
- per-event `time_weight` (1.5 at night/weekend, 1.0 otherwise)
- exponential time decay (15-day half-life over a 30-day window)
- `spread_bonus` for distinct patrol sessions
- `duration_bonus` for ≥2 sessions on the same calendar day

A score ≥ `FLAG_THRESHOLD` (8.0) flips `is_flagged=true`.  See
`app/services/occupancy.py` for the full formula.

`SpotOccupancyEvent.session_id` is `ON DELETE SET NULL`, and the
`/api/sessions/{id}` `DELETE` endpoint calls `cleanup_session_occupancy`
to drop only the events linked to that patrol session — events with
`session_id=NULL` (e.g. the demo seed) survive.  This is what makes
detections accumulate across multiple patrol passes even after the
original videos are removed.

Relevant config keys (`app/core/config.py`):
- `MOBILE_LPR_CAMERA_HEIGHT_M`, `_FOCAL_PX`, `_PITCH_DEG` — camera intrinsics
- `MOBILE_LPR_PLATE_HEIGHT_M` — assumed plate centre height above ground
- `MOBILE_LPR_SEARCH_RADIUS_M` — radius (default **40 m**) used to bind a
  projected GPS point to a registered `ParkingSpot`.  Sized for consumer
  smartphone GPS accuracy (5–15 m open sky, 20–40 m urban canyons).
- `MOBILE_LPR_MAX_DISTANCE_M` — clamp for plates near/beyond the geometric horizon
- `MOBILE_LPR_DEFAULT_LATITUDE`, `_LONGITUDE`, `_HEADING_DEG` — fallback pose
  applied to sessions uploaded without explicit GPS

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

### Residential spot-misuse demo

`app/core/seed.py` ships a 10-spot residential lot at the GPS coordinates of
the bundled `uploads/video1.MOV`.  The lot/spot seeding is idempotent — it
re-creates after any partial wipe (`DELETE FROM parking_spots`) without a
full `DROP SCHEMA`.

To populate the **Spot Misuse** panel from real video1 detections:

```bash
# 1. Upload + process video1.MOV via the API (it ships in uploads/) so the
#    Detection rows with target_latitude/longitude exist.
# 2. Then seed simulated multi-day patrol data:
PYTHONPATH=. venv/bin/python seed/seed_residential_misuse.py
```

The script:
1. Bootstraps the residential lot if missing (`_seed_residential_lot`)
2. Wipes existing `spot_occupancy_*` rows for an idempotent rerun
3. Reads every plate seen in video1 and binds it to its closest reserved spot
4. Generates 3–9 events per plate spread across the past 8 days, with
   `session_id=NULL` so the records survive deletion of the source session
5. Backfills `spot_match_status` on the existing detection rows so the
   per-card badges in the session view show `WRONG_PLATE` instead of the
   original `NO_SPOT_FOUND`

Output is a printed table (plate → spot → score → flag).  6 of the 13 video1
plates clear the flag threshold by default.
