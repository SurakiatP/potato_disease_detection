# FastAPI Only (Development)

from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
from pathlib import Path

app = FastAPI(
    title="Potato Disease Detection API",
    description="API for detecting potato diseases: Early Blight, Late Blight, and Healthy",
    version="2.0.0"
)

# Enhanced model loading with version detection
def load_latest_model():
    models_dir = Path("../models")
    if not models_dir.exists():
        models_dir = Path("models")  # fallback
    
    if not models_dir.exists():
        raise FileNotFoundError("Models directory not found!")
    
    # Find Lestest Model
    model_files = list(models_dir.glob("potato_model_v*.keras"))
    if not model_files:
        raise FileNotFoundError("No model files found!")
    
    # order by model versions
    def get_version(filename):
        try:
            return int(filename.stem.split('_v')[1])
        except:
            return 0
    
    latest_model = max(model_files, key=get_version)
    print(f"Loading model: {latest_model}")
    
    model = tf.keras.models.load_model(str(latest_model))
    return model, latest_model.name

try:
    MODEL, MODEL_NAME = load_latest_model()
    print(f"[Success!] Model loaded successfully: {MODEL_NAME}")
except Exception as e:
    print(f"[!] Error loading model: {e}")
    MODEL = None
    MODEL_NAME = "None"

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
IMAGE_SIZE = 256

@app.get("/")
async def root():
    return {
        "message": "Potato Disease Detection API",
        "model_loaded": MODEL is not None,
        "model_name": MODEL_NAME,
        "classes": CLASS_NAMES,
        "image_size": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "endpoints": {
            "GET /": "This info",
            "GET /ping": "Health check",
            "POST /predict": "Upload image for prediction",
            "GET /model-info": "Model information"
        }
    }

@app.get("/ping")
async def ping():
    return {
        "status": "alive",
        "model_loaded": MODEL is not None,
        "model_name": MODEL_NAME
    }

@app.get("/model-info")
async def model_info():
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Get model info
    total_params = MODEL.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in MODEL.trainable_weights])
    
    return {
        "model_name": MODEL_NAME,
        "classes": CLASS_NAMES,
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "input_shape": MODEL.input_shape,
        "output_shape": MODEL.output_shape
    }

def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data))
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """[/] Proper preprocessing exactly like training"""
    # Resize to target size
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(pil_image)
    
    # Normalize to 0-1 (same as training)
    image = image.astype(np.float32) / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        image_data = await file.read()
        image = read_file_as_image(image_data)
        
        print(f"Original image shape: {image.shape}")
        
        #[Success!] Proper preprocessing
        processed_image = preprocess_image(image)
        print(f"Processed image shape: {processed_image.shape}")
        print(f"Processed image range: {processed_image.min():.3f} - {processed_image.max():.3f}")
        
        # Predict
        predictions = MODEL.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(np.max(predictions[0]))
        
        # Get all class probabilities
        all_predictions = {
            CLASS_NAMES[i]: float(predictions[0][i]) 
            for i in range(len(CLASS_NAMES))
        }
        
        # Sort by confidence
        sorted_predictions = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_predictions": sorted_predictions,
            "model_used": MODEL_NAME
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Endpoint for batch prediction test
@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if len(files) > 10:  # Limit the number of image files
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    
    for file in files:
        try:
            image_data = await file.read()
            image = read_file_as_image(image_data)
            processed_image = preprocess_image(image)
            
            predictions = MODEL.predict(processed_image, verbose=0)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))
            
            results.append({
                "filename": file.filename,
                "predicted_class": predicted_class,
                "confidence": confidence
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "model_name": MODEL_NAME,
        "tensorflow_version": tf.__version__
    }

if __name__ == "__main__":
    print(f"+ Starting Potato Disease Detection API")
    print(f"+ Model: {MODEL_NAME}")
    print(f"+ Classes: {CLASS_NAMES}")
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)