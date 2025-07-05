#!/usr/bin/env python3
"""
TensorFlow Lite Model Converter
Convert Keras model to TF Lite for mobile deployment
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import os
from PIL import Image

def load_latest_model():
    """Load the latest Keras model"""
    models_dir = Path("../models")
    if not models_dir.exists():
        models_dir = Path("models")  # fallback
    
    if not models_dir.exists():
        raise FileNotFoundError("Models directory not found!")
    
    # Find latest model
    model_files = list(models_dir.glob("potato_model_v*.keras"))
    if not model_files:
        raise FileNotFoundError("No model files found!")
    
    # Order by model versions
    def get_version(filename):
        try:
            return int(filename.stem.split('_v')[1])
        except:
            return 0
    
    latest_model = max(model_files, key=get_version)
    print(f"Loading model: {latest_model}")
    
    model = tf.keras.models.load_model(str(latest_model))
    return model, latest_model.name

def convert_to_tflite(model, model_name, optimization_level="default"):
    """Convert Keras model to TF Lite with different optimization levels"""
    
    output_dir = Path("../mobile_models")
    output_dir.mkdir(exist_ok=True)
    
    base_name = model_name.replace('.keras', '')
    
    if optimization_level == "default":
        # Default optimization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        output_path = output_dir / f"{base_name}_default.tflite"
        
    elif optimization_level == "float16":
        # Float16 quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        output_path = output_dir / f"{base_name}_float16.tflite"
        
    elif optimization_level == "int8":
        # INT8 quantization (requires representative dataset)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Representative dataset for quantization
        def representative_dataset():
            # Generate dummy data for quantization
            for _ in range(100):
                yield [np.random.random((1, 256, 256, 3)).astype(np.float32)]
        
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        tflite_model = converter.convert()
        output_path = output_dir / f"{base_name}_int8.tflite"
    
    else:
        raise ValueError("Invalid optimization level")
    
    # Save model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"[Success] TF Lite model saved: {output_path}")
    return output_path, len(tflite_model)

def create_model_metadata(model_name, model_info):
    """Create metadata JSON file for mobile app"""
    
    metadata = {
        "model_name": model_name,
        "version": "1.0.0",
        "classes": ["Early Blight", "Late Blight", "Healthy"],
        "input_size": [256, 256, 3],
        "preprocessing": {
            "resize": [256, 256],
            "normalize": "0-1",
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0]
        },
        "postprocessing": {
            "output_type": "classification",
            "num_classes": 3
        },
        "model_info": model_info,
        "disease_info": {
            "Early Blight": {
                "name": "Early Blight",
                "symptoms": "Brown spots on leaves with concentric rings",
                "treatment": "Apply fungicide, improve air circulation",
                "prevention": "Avoid watering leaves, maintain proper spacing"
            },
            "Late Blight": {
                "name": "Late Blight",
                "symptoms": "Dark brown spots, leaves wilt and dry",
                "treatment": "Apply fungicide every 7-10 days",
                "prevention": "Ensure good drainage, reduce humidity"
            },
            "Healthy": {
                "name": "Healthy",
                "symptoms": "Green fresh leaves, no abnormal spots",
                "treatment": "No treatment needed",
                "prevention": "Continue proper care"
            }
        }
    }
    
    output_dir = Path("../mobile_models")
    metadata_path = output_dir / "model_metadata.json"
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"[Success] Metadata saved: {metadata_path}")
    return metadata_path

def test_tflite_model(tflite_path):
    """Test TF Lite model with dummy data"""
    
    # Load TF Lite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\n+ Model Info:")
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input type: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output type: {output_details[0]['dtype']}")
    
    # Test with dummy data
    input_shape = input_details[0]['shape']
    if input_details[0]['dtype'] == np.uint8:
        # INT8 quantized model
        dummy_input = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
    else:
        # Float model
        dummy_input = np.random.random(input_shape).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"[Success] Test prediction shape: {output_data.shape}")
    print(f"[Success] Test prediction sample: {output_data[0]}")
    
    return True

def main():
    """Main conversion process"""
    
    print("+ Starting TF Lite Conversion Process...")
    
    try:
        # Load latest model
        model, model_name = load_latest_model()
        print(f"+ Loaded model: {model_name}")
        
        # Get model info
        model_info = {
            "total_parameters": int(model.count_params()),
            "trainable_parameters": int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])),
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape)
        }
        
        print(f"+ Model Parameters: {model_info['total_parameters']:,}")
        
        # Convert to different TF Lite formats
        conversions = []
        
        # 1. Default optimization
        print("\n1. Converting with default optimization...")
        path, size = convert_to_tflite(model, model_name, "default")
        conversions.append(("Default", path, size))
        
        # 2. Float16 optimization  
        print("\n2. Converting with Float16 optimization...")
        path, size = convert_to_tflite(model, model_name, "float16")
        conversions.append(("Float16", path, size))
        
        # 3. INT8 optimization
        print("\n3. Converting with INT8 optimization...")
        path, size = convert_to_tflite(model, model_name, "int8")
        conversions.append(("INT8", path, size))
        
        # Create metadata
        print("\n+ Creating model metadata...")
        create_model_metadata(model_name, model_info)
        
        # Test models
        print("\n+ Testing TF Lite models...")
        for name, path, size in conversions:
            print(f"\n--- Testing {name} model ---")
            test_tflite_model(path)
        
        # Summary
        print("\n" + "="*50)
        print("+ CONVERSION SUMMARY")
        print("="*50)
        
        for name, path, size in conversions:
            size_mb = size / (1024 * 1024)
            print(f"{name:10} | {size_mb:6.2f} MB | {path.name}")
        
        print("\n[Success] All conversions completed successfully!")
        print("+ Models ready for mobile deployment!")
        
        # Recommendations
        print("\n+ RECOMMENDATIONS:")
        print("• Default: Good balance of size and accuracy")
        print("• Float16: Smaller size, minimal accuracy loss")
        print("• INT8: Smallest size, fastest inference (may have accuracy loss)")
        print("• Test all models on your target device to choose the best one")
        
    except Exception as e:
        print(f"[!] Error: {e}")
        raise

if __name__ == "__main__":
    main()