from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import base64
import cv2
import numpy as np
import os
import tempfile
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Accident Detection API",
    description="API for processing video and image data for accident detection",
    version="1.0.0",
    max_upload_size=1024 * 1024 * 100  # 100MB limit
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = YOLO("best5.pt")
        logger.info("✅ YOLOv8 Model Loaded Successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise

@app.get("/")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("🔗 WebSocket Connected")

    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                image_bytes = base64.b64decode(data)
                image = Image.open(io.BytesIO(image_bytes))
                logger.info(f"🖼 Received image ({len(data)} bytes)")

                results = model(image)
                detections = process_detections(results)
                
                await websocket.send_json({
                    "detections": detections,
                    "alert": len(detections) > 0
                })
            except Exception as e:
                logger.error(f"Image processing error: {str(e)}")
                await websocket.send_json({
                    "error": str(e),
                    "alert": False
                })

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        logger.info("⚠️ WebSocket Disconnected")
        await websocket.close()

@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    try:
        # Validate input
        if not file.content_type.startswith('video/'):
            raise HTTPException(
                status_code=400,
                detail="Only video files are accepted"
            )

        logger.info(f"📹 Processing video: {file.filename} ({file.size} bytes)")

        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_path = temp_video.name
            # Read in chunks for memory efficiency
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                temp_video.write(chunk)

        # Process video frames
        results = process_video_frames(temp_path)
        
        # Clean up
        os.unlink(temp_path)
        
        return JSONResponse({
            "success": True,
            "detections": results,
            "alert": len(results) > 0,
            "processed_frames": len(results),
            "filename": file.filename
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(e)}"
        )

def process_detections(results) -> List[Dict]:
    """Process YOLOv8 results into detection objects"""
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            confidence = float(box.conf)
            
            if class_name in ["crashed", "accident"]:
                detections.append({
                    "class": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": [round(x, 2) for x in box.xyxy[0].tolist()],
                    "timestamp": getattr(result, "speed", {}).get("preprocess", 0)
                })
    return detections

def process_video_frames(video_path: str, frame_interval: int = 10) -> List[Dict]:
    """Process video frames at specified intervals"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    all_detections = []
    
    logger.info(f"🎞 Processing video: {fps:.2f} FPS, {total_frames} frames")
    
    try:
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
                for det in frame_detections:
                    det["frame_number"] = frame_count
                    det["timestamp_sec"] = round(frame_count / fps, 2)
                all_detections.extend(frame_detections)
    
    finally:
        cap.release()
    
    logger.info(f"Processed {frame_count}/{total_frames} frames, found {len(all_detections)} detections")
    return all_detections
