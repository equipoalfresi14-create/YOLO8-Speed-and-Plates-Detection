import cv2 as cv
import time
import json
import pytz
import numpy as np
from datetime import datetime
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import supervision as sv
from paddleocr import PaddleOCR
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

tz = pytz.timezone("America/Hermosillo")
timestamp = datetime.now(tz)
fecha_str = timestamp.strftime("%d-%m-%Y %H:%M:%S")

#configuración
VELOCITY_LIMIT = 0.1
CAR_REAL_HEIGHT_M = 1.5
OUTPUT_JSON = "infracciones.json"

def estimate_distance_per_pixel(bbox):
    x1, y1, x2, y2 = bbox
    height_px = max(1, y2 - y1)  #evitar división entre cero
    dpp = CAR_REAL_HEIGHT_M / height_px
    return dpp

app = FastAPI()

# Permitir CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

car_model = YOLO("yolov8s.pt")
plate_model = YOLO("runs/detect/train2/weights/best.pt")
ocr = PaddleOCR(lang="en", use_angle_cls=True)
tracker = sv.ByteTrack()

last_pos = {}
speeds = {}
saved = {}

try:
    with open(OUTPUT_JSON, "r") as f:
        infracciones = json.load(f)
except:
    infracciones = []

plate_memory = defaultdict(list)

@app.post("/frame")
async def process_frame(frame: UploadFile = File(...)):
    #leer imagen
    bytes_data = await frame.read()
    np_data = np.frombuffer(bytes_data, np.uint8)
    img = cv.imdecode(np_data, cv.IMREAD_COLOR)

    #Detectar carros
    results = car_model(img, verbose=False)[0]
    dets = sv.Detections.from_ultralytics(results)
    dets = dets[dets.class_id == 2]  

    print(f"[DEBUG] Carros detectados en frame: {len(dets)}")

    tracked = tracker.update_with_detections(dets)
    timestamp = time.time()

    response_data = {"cars": []}

    if tracked is None or len(tracked) == 0:
        return response_data

    for bbox, track_id in zip(tracked.xyxy, tracked.tracker_id):
        x1, y1, x2, y2 = bbox.astype(int)
        track_id = int(track_id)
        
        #centro del vehículo
        cy = (y1 + y2) // 2

        #distancia por pixel dinámica
        dpp = estimate_distance_per_pixel((x1, y1, x2, y2))

        #calcular velocidad
        if track_id in last_pos:
            prev_cy, prev_t = last_pos[track_id]
            dy = abs(cy - prev_cy)
            dt = timestamp - prev_t
            if dt > 0:
                v = (dy * dpp / dt) * 3.6  # km/h
                speeds[track_id] = v
                print(f"[DEBUG] Track {track_id}: dy={dy}, dt={dt:.4f}, dpp={dpp:.5f}, v={v:.2f} km/h")
        # else:
        #     print(f"[DEBUG] Nuevo track_id: {track_id}")

        last_pos[track_id] = (cy, timestamp)
        v = speeds.get(track_id, 0)

        response_data["cars"].append({
            "id": int(track_id),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "speed": float(v)
        })

        # en caso de exceder la velocidad limite
        if v > VELOCITY_LIMIT and track_id not in saved:
            print(f"[DEBUG] >>> INFRACCIÓN: track {track_id}, v={v:.2f} km/h")

            #expandir crop
            h, w = img.shape[:2]
            margin = 20
            x1e = max(0, x1 - margin)
            y1e = max(0, y1 - margin)
            x2e = min(w, x2 + margin)
            y2e = min(h, y2 + margin)

            crop = img[y1e:y2e, x1e:x2e]

            #detectar placa
            plate_res = plate_model(crop, verbose=False)[0]
            plate_dets = sv.Detections.from_ultralytics(plate_res)

            print("[DEBUG] plate_dets:", plate_dets.xyxy if len(plate_dets) > 0 else "NO DETECTIONS")

            plate_text = "UNKNOWN"

            if len(plate_dets) > 0:
                #coordenadas dentro del crop
                px1, py1, px2, py2 = plate_dets.xyxy[0].astype(int)

                abs_x1 = x1e + px1
                abs_y1 = y1e + py1
                abs_x2 = x1e + px2
                abs_y2 = y1e + py2

                plate_crop = img[abs_y1:abs_y2, abs_x1:abs_x2]
                print("[DEBUG] plate_crop shape:", plate_crop.shape)

                #escalar placa si está muy chica
                h, w = plate_crop.shape[:2]
                if h < 80:
                    scale_factor = max(2, int(80 / h))
                    plate_crop = cv.resize(plate_crop, (w*scale_factor, h*scale_factor), interpolation=cv.INTER_CUBIC)
                    print("[DEBUG] plate_crop UPSCALED to:", plate_crop.shape)

                plate_text = "UNKNOWN"
                best_conf = 0.0

                MAX_OCR_TRIES = 5

                for attempt in range(MAX_OCR_TRIES):
                    try:
                        ocr_out = ocr.ocr(plate_crop)
                        print(f"[DEBUG] OCR output intento {attempt+1}:", ocr_out)

                        if (
                            isinstance(ocr_out, list)
                            and len(ocr_out) > 0
                            and isinstance(ocr_out[0], dict)
                            and "rec_texts" in ocr_out[0]
                            and len(ocr_out[0]["rec_texts"]) > 0
                        ):
                            text = ocr_out[0]["rec_texts"][0]
                            conf = ocr_out[0]["rec_scores"][0]

                            if conf > best_conf:   #mejor lectura
                                best_conf = conf
                                plate_text = text

                    except Exception as e:
                        print(f"[DEBUG] OCR FAIL intento {attempt+1}:", e)

                #guardar historial por track
                plate_memory[track_id].append(plate_text)

                #elegir por mayoría (para mejor estabilidad)
                from collections import Counter
                counts = Counter(plate_memory[track_id])
                plate_text = counts.most_common(1)[0][0]

            else:
                print("[DEBUG] OCR SKIPPED: plate_dets empty")
            #print(f">>> INFRACCION track {track_id}, placa={plate_text}, v={v:.2f}")

            #evitar guardar doble el mismo ID
            saved[track_id] = plate_text

            #guardar en el json
            infracciones.append({
                "track_id": int(track_id),
                "placa": plate_text,
                "velocidad": float(v),
                "timestamp": fecha_str

            })
            print("[DEBUG] Guardando infracción en JSON:", infracciones)

            with open(OUTPUT_JSON, "w") as f:
                json.dump(infracciones, f, indent=4)

    return response_data 
