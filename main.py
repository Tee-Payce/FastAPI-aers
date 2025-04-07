from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image
import io
import base64
import cv2
import numpy as np
import os
import tempfile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 Model
model = None

@app.on_event("startup")
async def load_model():
    global model
    if model is None:
        model = YOLO("best5.pt")
    print("✅ YOLOv8 Model Loaded")

@app.get("/")
def read_root():
    return {"message": "FastAPI is live on Render 🚀"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔗 WebSocket Connected")

    try:
        while True:
            data = await websocket.receive_text()
            image_bytes = base64.b64decode(data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Run YOLOv8 inference
            results = model(image)
            detections = process_detections(results)
            
            await websocket.send_json({
                "detections": detections,
                "alert": len(detections) > 0
            })

    except Exception as e:
        print(f"🔴 WebSocket Error: {e}")
    finally:
        print("⚠️ WebSocket Disconnected")

@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_path = temp_video.name
            contents = await file.read()
            temp_video.write(contents)
        
        # Process video
        results = process_video_frames(temp_path)
        
        # Clean up
        os.unlink(temp_path)
        
        return JSONResponse({
            "success": True,
            "detections": results,
            "alert": len(results) > 0
        })
        
    except Exception as e:
        raise HTTPException(500, detail=str(e))

def process_detections(results) -> List[Dict]:
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            confidence = float(box.conf)
            
            if class_name in ["crashed", "accident"]:
                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": box.xyxy[0].tolist()
                })
    return detections

def process_video_frames(video_path: str, frame_interval: int = 10) -> List[Dict]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    all_detections = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue
            
        # Convert to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Run detection
        results = model(pil_image)
        frame_detections = process_detections(results)
        
        if frame_detections:
            all_detections.extend(frame_detections)
    
    cap.release()
    return all_detections
