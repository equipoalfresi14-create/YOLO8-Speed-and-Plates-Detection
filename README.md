# License Plate Recognition & Speed Estimation using YOLO
Real-time vehicle tracking, speed calculation, and plate recognition using YOLOv8, DeepSORT, and PaddleOCR, deployed with FastAPI.

## Overview
This project processes a live video stream frame-by-frame to:
- Detect vehicles (YOLOv8)
- Track them across frames (DeepSORT)
- Estimate their speed from pixel displacement and time
- Detect license plates
- Run OCR to recognize the plate text
- Log speeding events as JSON
All processing is done on a FastAPI backend, and a lightweight JS frontend sends frames to the server.

## Running the server
`uvicorn server:app --host 0.0.0.0 --port 8000`
## Using ngrok for mobile devices
`ngrok http 8000`

## Output example (JSON)
```
{
  "track_id": 14,
  "plate": "ABC123",
  "speed_kmh": 4.70,
  "timestamp": "29-11-2025 17:32:58"
}
```
