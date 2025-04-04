from fastapi import FastAPI, WebSocket
from ultralytics import YOLO
from PIL import Image
import io
import base64
import cv2

app = FastAPI()

# Load YOLOv8 Model
model = YOLO("best5.pt")
@app.get("/")
def read_root():
    return {"message": "FastAPI is live on Render 🚀"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔗 WebSocket Connected")

    try:
        while True:
            data = await websocket.receive_text()  # Receive base64 image
            image_bytes = base64.b64decode(data)
            image = Image.open(io.BytesIO(image_bytes))

            # Run YOLOv8 inference
            results = model(image)
            detections = []

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    confidence = float(box.conf)
                    
                    # Only store detections for "crashed" and "accident"
                    if class_name in ["crashed", "accident"]:
                        detections.append({
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": box.xyxy[0].tolist()
                        })

            # Send detection results back to the client
            await websocket.send_json({"detections": detections, "alert": len(detections) > 0})

    except Exception as e:
        print(f"🔴 WebSocket Error: {e}")
    finally:
        print("⚠️ WebSocket Disconnected")

