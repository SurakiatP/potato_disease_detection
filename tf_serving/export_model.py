"""
Export Keras Model to TensorFlow SavedModel Format for TensorFlow Serving

This script converts trained .keras models to SavedModel format required by
TensorFlow Serving infrastructure for high-performance model serving.

Usage:
    python export_model.py

The script will:
1. Find the latest trained model in ../models/
2. Export it to SavedModel format in ./models/potato_model/{version}/
3. Verify the export was successful
"""

import tensorflow as tf
import os
from pathlib import Path
import sys

def find_latest_model():
    """Find the latest trained model in the models directory"""
    
    # Get project root (parent of tf_serving)
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    print(f"+ Looking for models in: {models_dir}")
    
    if not models_dir.exists():
        print(f"[!] Models directory not found: {models_dir}")
        return None
    
    # Find all .keras model files
    keras_files = list(models_dir.glob("potato_model_v*.keras"))
    
    if not keras_files:
        print("[!] No .keras model files found!")
        print("   Please train a model first and save it as potato_model_v{N}.keras")
        return None
    
    # Get the latest version
    def get_version_number(filepath):
        try:
            # Extract version from filename like "potato_model_v2.keras" -> 2
            version_str = filepath.stem.split('_v')[1]
            return int(version_str)
        except (IndexError, ValueError):
            return 0
    
    # Sort by version number and get the latest
    latest_model = max(keras_files, key=get_version_number)
    version = get_version_number(latest_model)
    
    print(f"[Success] Found latest model: {latest_model.name} (version {version})")
    return latest_model, version

def export_model(model_path, version):
    """Export model to SavedModel format for TensorFlow Serving"""
    
    print(f"\n+ Starting model export...")
    
    # Load the Keras model
    try:
        print(f"+ Loading model from: {model_path}")
        model = tf.keras.models.load_model(str(model_path))
        print("[Success] Model loaded successfully")
        
        # Print model summary for verification
        print(f"\n+ Model Summary:")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Total parameters: {model.count_params():,}")
        
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        return False
    
    # Prepare export path
    tf_serving_dir = Path(__file__).parent
    export_path = tf_serving_dir / "models" / "potato_model" / str(version)
    
    # Create directory if it doesn't exist
    export_path.mkdir(parents=True, exist_ok=True)
    
    print(f"+ Export destination: {export_path}")
    
    # Export model to SavedModel format
    try:
        print("+ Exporting model to SavedModel format...")
        model.export(str(export_path))
        print("+ Model export completed!")
        
    except Exception as e:
        print(f"[!] Export failed: {e}")
        return False
    
    # Verify the export
    return verify_export(export_path, version)

def verify_export(export_path, version):
    """Verify that the export was successful"""
    
    print(f"\n+ Verifying export...")
    
    # Check required files
    required_files = [
        export_path / "saved_model.pb",
        export_path / "variables"
    ]
    
    missing_files = []
    for required_file in required_files:
        if not required_file.exists():
            missing_files.append(str(required_file))
    
    if missing_files:
        print(f"[!] Export verification failed!")
        print(f"   Missing files: {missing_files}")
        return False
    
    # Try to load the exported model
    try:
        print("+ Testing model loading...")
        loaded_model = tf.saved_model.load(str(export_path))
        print("[Success] Exported model loads successfully")
        
        # Check signatures
        signatures = list(loaded_model.signatures.keys())
        print(f"+ Available signatures: {signatures}")
        
    except Exception as e:
        print(f"[!] Error loading exported model: {e}")
        return False
    
    # Display file structure
    print(f"\n+ Exported file structure:")
    for root, dirs, files in os.walk(export_path):
        level = root.replace(str(export_path), '').count(os.sep)
        indent = '  ' * level
        folder_name = os.path.basename(root) if level > 0 else f"potato_model/{version}/"
        print(f"{indent}{folder_name}")
        
        # Show files with sizes
        subindent = '  ' * (level + 1)
        for file in files:
            file_path = Path(root) / file
            try:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"{subindent}{file} ({size_mb:.1f} MB)")
            except:
                print(f"{subindent}{file}")
    
    # Calculate total size
    total_size = sum(
        f.stat().st_size for f in export_path.rglob('*') if f.is_file()
    ) / (1024 * 1024)
    
    print(f"\n+ Export Summary:")
    print(f"   + Status: Successful")
    print(f"   + Location: {export_path}")
    print(f"   + Total size: {total_size:.1f} MB")
    print(f"   + Version: {version}")
    print(f"   + Ready for TensorFlow Serving!")
    
    return True

def main():
    """Main function"""
    print("Potato Disease Detection - Model Export for TensorFlow Serving")
    print("=" * 70)
    
    # Find latest model
    model_info = find_latest_model()
    if model_info is None:
        print("\n[!] No models found to export!")
        print("\n>> Next steps:")
        print("   1. Train a model using your training notebook")
        print("   2. Save it as 'potato_model_v{N}.keras' in the models/ directory")
        print("   3. Run this script again")
        return False
    
    model_path, version = model_info
    
    # Export model
    success = export_model(model_path, version)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)