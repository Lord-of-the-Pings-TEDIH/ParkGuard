# ParkGuard

## Quick Start

To seed the database with initial test data, run the following command in your terminal (using PowerShell/CMD):

```powershell
Get-Content seed/parking_data.sql | docker exec -i -e PGPASSWORD=parkguard parkguard-db psql -U parkguard -d parkguard
```
*(For bash/Linux/macOS, you can use: `docker exec -i -e PGPASSWORD=parkguard parkguard-db psql -U parkguard -d parkguard < seed/parking_data.sql`)*

## PaddleOCR Model Download

The first time PaddleOCR is initialized (API or POC), it downloads model weights
and caches them locally. Subsequent runs reuse cached files.

## YOLO11 Plate Detection POC

1. Use the trained detector from `models/best.pt`.
2. Run the proof-of-concept script on `vid.mp4`:
   ```bash
   venv/bin/python yolo11_plate_poc.py --video vid.mp4 --model models/best.pt
   ```
3. Inspect the annotated output video:
   `out/yolo11_plate_poc.mp4`

## Romanian Plate Detector + PaddleOCR on v1.mp4

Use Python 3.12 for this flow (`paddlepaddle` is typically unavailable on Python 3.13+).

Install PaddleOCR once (plus PaddlePaddle runtime):

```bash
venv/bin/pip install paddleocr paddlepaddle
```

Run the dedicated script for your trained Romanian plate model:

```bash
venv/bin/python ro_plates_video_poc.py
```

This POC now includes:
- deskew/perspective rectification before OCR,
- OCR at multiple angles (`--ocr-angles`),
- blur filtering via Laplacian sharpness (`--min-sharpness`),
- temporal track voting (`--min-track-votes`, `--track-max-age`).

The API endpoint `POST /sessions/{session_id}/process` now uses the same
robust flow (deskew + blur-aware OCR + multi-angle + temporal voting) via
environment settings:
`OCR_USE_ANGLE_CLS`, `OCR_MIN_CONF`, `OCR_ANGLES`, `OCR_MIN_SHARPNESS`,
`TRACK_MAX_AGE`, `TRACK_MIN_IOU`, `MIN_TRACK_VOTES`.

Example tuned for angled/moving plates:

```bash
venv/bin/python ro_plates_video_poc.py \
  --use-angle-cls \
  --ocr-angles "-15,-8,0,8,15" \
  --min-sharpness 80 \
  --min-ocr-conf 0.35 \
  --min-track-votes 3
```

Defaults:
- input video: `v1.mp4` (also resolves `v1.MP4`)
- model: `models/best.pt`
- output video: `out/v1_ro_plates_detected.mp4`

## YOLO11n + NVIDIA LPRNet Batch POC (v1..v4)

Use this script to run all MP4 samples (`v1.MP4`, `v2.MP4`, `v3.MP4`, `v4.MP4`) with:
- `models/best.pt` for detection
- NVIDIA LPRNet ONNX for character recognition

Download NVIDIA LPRNet ONNX once:

```bash
curl -L --fail 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/lprnet/deployable_onnx_v1.1/files?redirect=true&path=us_lprnet_baseline18_deployable.onnx' \
  -o models/us_lprnet_baseline18_deployable.onnx
```

The script uses `models/us_lprnet_baseline18_deployable.onnx` by default.
For NVIDIA LPRNet exports with LSTM ops, the script auto-falls back to `onnxruntime`.

```bash
venv/bin/python yolo11n_nvidia_lpr_poc.py \
  --videos v1.MP4 v2.MP4 v3.MP4 v4.MP4 \
  --model models/best.pt \
  --lprnet-onnx models/us_lprnet_baseline18_deployable.onnx
```

The script writes a summary JSON at:
`out/yolo11n_nvidia_lpr_summary.json`

## Reset DB

If you need to wipe the database and start fresh with a clean schema and seed data, follow these steps:

1. **Drop the Schema**
   This will completely drop all existing tables and data:
   ```bash
   docker exec -i -e PGPASSWORD=parkguard parkguard-db psql -U parkguard -d parkguard -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
   ```

2. **Recreate Tables (`create_all`)**
   Restart your backend server (or let it auto-reload if you're using `uvicorn --reload`). The application's `lifespan` startup routine will execute SQLAlchemy's `create_all` to recreate the empty tables according to your models.

3. **Re-Seed Data**
   Run the seed command again to insert the initial data:
   ```powershell
   Get-Content seed/parking_data.sql | docker exec -i -e PGPASSWORD=parkguard parkguard-db psql -U parkguard -d parkguard
   ```
