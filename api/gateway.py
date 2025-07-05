# FastAPI Gateway (Production) >> fastapi + tf serving (Cloud)

from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import asyncio
import aiohttp

app = FastAPI(
    title="Potato Disease Detection Gateway",
    description="FastAPI Gateway to TensorFlow Serving",
    version="2.0.0"
)

# Configuration
TF_SERVING_URL = "http://localhost:8501"
MODEL_NAME = "potato_model"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
IMAGE_SIZE = 256

class TFServingClient:
    """Simple client for TensorFlow Serving"""
    
    def __init__(self, base_url: str, model_name: str):
        self.predict_url = f"{base_url}/v1/models/{model_name}:predict"
        self.metadata_url = f"{base_url}/v1/models/{model_name}"
    
    async def check_health(self) -> dict:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.metadata_url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {"status": "healthy", "model_loaded": True}
                    else:
                        return {"status": "unhealthy", "model_loaded": False}
        except Exception as e:
            return {"status": "unhealthy", "model_loaded": False, "error": str(e)}
    
    async def predict(self, instances: list) -> dict:
        payload = {"instances": instances}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.predict_url, json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {"success": True, "predictions": result["predictions"]}
                    else:
                        error_text = await response.text()
                        return {"success": False, "error": f"TF Serving error: {error_text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

# Initialize TF Serving client
tf_client = TFServingClient(TF_SERVING_URL, MODEL_NAME)

def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image exactly like training"""
    # Resize to model input size
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(pil_image)
    
    # Normalize to [0, 1] range
    image = image.astype(np.float32) / 255.0
    
    return image

@app.get("/")
async def root():
    tf_health = await tf_client.check_health()
    
    return {
        "service": "Potato Disease Detection Gateway",
        "architecture": "FastAPI → TensorFlow Serving",
        "tensorflow_serving": {
            "url": TF_SERVING_URL,
            "model": MODEL_NAME,
            "status": tf_health["status"]
        },
        "classes": CLASS_NAMES,
        "endpoints": {
            "GET /": "This information",
            "GET /health": "Health check",
            "POST /predict": "Single image prediction"
        }
    }

@app.get("/health")
async def health_check():
    tf_health = await tf_client.check_health()
    
    overall_status = "healthy" if tf_health["model_loaded"] else "unhealthy"
    
    return {
        "overall_status": overall_status,
        "fastapi": {"status": "healthy"},
        "tensorflow_serving": tf_health
    }

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    # Check TF Serving health first
    tf_health = await tf_client.check_health()
    if not tf_health["model_loaded"]:
        raise HTTPException(status_code=503, detail="TensorFlow Serving unavailable")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        # Read and preprocess image
        image_data = await file.read()
        image = read_file_as_image(image_data)
        processed_image = preprocess_image(image)
        
        # Prepare data for TensorFlow Serving
        instances = [processed_image.tolist()]
        
        # Call TensorFlow Serving
        tf_result = await tf_client.predict(instances)
        
        if not tf_result["success"]:
            raise HTTPException(status_code=500, detail=tf_result["error"])
        
        # Process predictions
        raw_predictions = tf_result["predictions"][0]
        predicted_class_idx = np.argmax(raw_predictions)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(np.max(raw_predictions))
        
        # All predictions
        all_predictions = {
            CLASS_NAMES[i]: float(raw_predictions[i]) 
            for i in range(len(CLASS_NAMES))
        }
        sorted_predictions = dict(sorted(all_predictions.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_predictions": sorted_predictions,
            "model_backend": "TensorFlow Serving"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    print("+ Starting FastAPI Gateway to TensorFlow Serving")
    print(f"+ TensorFlow Serving: {TF_SERVING_URL}")
    print(f"+ Model: {MODEL_NAME}")
    print(f"+ Classes: {CLASS_NAMES}")
    print(f"+ API Docs: http://localhost:8000/docs")
    
    uvicorn.run("gateway:app", host='0.0.0.0', port=8000, reload=True)