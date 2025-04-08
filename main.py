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
import math
import psutil
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Accident Detection API",
    description="API for processing video and image data for accident detection",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
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
                logger.error(f"❌ Image processing error: {str(e)}")
                await websocket.send_json({
                    "error": str(e),
                    "alert": False
                })

    except Exception as e:
        logger.error(f"⚠️ WebSocket connection error: {str(e)}")
    finally:
        await websocket.close()
        logger.info("🔌 WebSocket Disconnected")

@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith("video/"):
            raise HTTPException(
                status_code=400,
                detail="Only video files are accepted"
            )

        logger.info(f"📹 Received video: {file.filename}")

        # Save video to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_path = temp_video.name
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                temp_video.write(chunk)

        # Process video
        results = process_video_frames(temp_path)
        logger.info("✅ Finished frame processing")
        logger.info(f"🔍 Raw detections: {results}")

        # Delete temp file
        os.unlink(temp_path)

        try:
            response_data = {
                "success": True,
                "detections": results,
                "alert": len(results) > 0,
                "processed_frames": len(results),
                "filename": file.filename
            }
            logger.info(f"✅ Response data: {response_data}")
            return JSONResponse(content=response_data)
        except Exception as e:
            logger.error(f"❌ Failed to serialize response: {str(e)}")
            raise HTTPException(status_code=500, detail="Response serialization failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Video processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

def process_detections(results) -> List[Dict]:
    detections = []
    for result in results:
        for box in result.boxes:
            try:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                confidence = float(box.conf)

                if class_name in ["crashed", "accident"]:
                    bbox = [round(x, 2) if isinstance(x, (float, int)) and not math.isnan(x) else 0.0 for x in box.xyxy[0].tolist()]
                    detections.append({
                        "class": class_name,
                        "confidence": round(confidence, 4),
                        "bbox": bbox,
                        "timestamp": getattr(result, "speed", {}).get("preprocess", 0)
                    })
            except Exception as e:
                logger.error(f"❌ Error processing box: {str(e)}")
    return detections

@app.post("/video-processing")
async def process_video(file: UploadFile = File(...)):
    logging.info("Received video file for processing.")
    try:
        os.makedirs('uploads', exist_ok=True)
        video_path = f"uploads/{file.filename}"

        with open(video_path, "wb") as f:
            f.write(await file.read())
        
        logging.info("Video saved successfully. Starting processing...")

        count_crashed = 0
        count_accident = 0
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.info("No more frames to read.")
                break

            frame_count += 1
            logging.info(f"Processing frame {frame_count}")
            frame = cv2.resize(frame, (640, 480))  # Resize frame

            try:
                results = model(frame)
                # Count detections...
            except Exception as e:
                logging.error(f"Error processing frame {frame_count}: {str(e)}")

        cap.release()
        os.remove(video_path)
        logging.info("Processing complete.")
        return JSONResponse(content={
            "crashed_count": count_crashed,
            "accident_count": count_accident
        })
    except MemoryError:
        logging.error("MemoryError: The application has run out of memory.")
        return JSONResponse(content={"error": "Out of memory"}, status_code=500)
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
        
def process_video_frames(video_path: str, frame_interval: int = 10) -> List[Dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    all_detections = []

    logger.info(f"🎞 FPS: {fps:.2f}, Total Frames: {total_frames}")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            results = model(pil_image)
            frame_detections = process_detections(results)

            for det in frame_detections:
                det["frame_number"] = frame_count
                det["timestamp_sec"] = round(frame_count / fps, 2)

            all_detections.extend(frame_detections)

    finally:
        cap.release()

    logger.info(f"✅ Processed {frame_count} frames. Found {len(all_detections)} detections.")
    return all_detections
