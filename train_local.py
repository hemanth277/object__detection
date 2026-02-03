"""
Train YOLOv8 Model Locally on Windows

This script trains a custom YOLOv8 model on your physics equipment dataset.

Requirements:
- Python 3.8+
- NVIDIA GPU (optional but recommended)
- At least 8GB RAM

Installation:
    pip install ultralytics opencv-python pillow

Usage:
    python train_local.py
"""

from ultralytics import YOLO
import os
from pathlib import Path

def main():
    print("=" * 60)
    print("YOLOv8 Local Training - Physics Equipment Detector")
    print("=" * 60)
    print()
    
    # Dataset configuration
    dataset_path = r"C:\Users\admin\Desktop\object detection\dataset"
    data_yaml = os.path.join(dataset_path, "data.yaml")
    
    # Check if dataset exists
    if not os.path.exists(data_yaml):
        print(f"[ERROR] Dataset not found at: {data_yaml}")
        print("Please make sure the dataset folder exists.")
        return
    
    print(f"Dataset location: {dataset_path}")
    print(f"Data config: {data_yaml}")
    print()
    
    # Load a pretrained YOLOv8 nano model (smallest/fastest)
    print("Loading YOLOv8 nano model...")
    model = YOLO('yolov8n.pt')
    print("[OK] Model loaded")
    print()
    
    # Training configuration
    print("Starting training...")
    print("Configuration:")
    print("  - Model: YOLOv8 nano")
    print("  - Epochs: 50")
    print("  - Image size: 640x640")
    print("  - Batch size: 8 (adjust if out of memory)")
    print()
    
    try:
        # Train the model
        results = model.train(
            data=data_yaml,
            epochs=50,              # Training iterations
            imgsz=640,              # Image size
            batch=8,                # Batch size (lower if out of memory)
            patience=10,            # Early stopping patience
            save=True,              # Save checkpoints
            project='runs/detect',  # Save location
            name='physics_equipment',
            plots=True,             # Generate training plots
            verbose=True,
            device='cpu'            # Use CPU (no GPU available)
        )
        
        print()
        print("=" * 60)
        print("[SUCCESS] Training complete!")
        print("=" * 60)
        print()
        print("Model saved at:", model.trainer.best)
        print()
        print("Next steps:")
        print("1. Check training results in: runs/detect/physics_equipment")
        print("2. Find your trained model: runs/detect/physics_equipment/weights/best.pt")
        print("3. Use the model with: python detect_custom.py")
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        print()
        print("Common issues:")
        print("1. Out of memory: Reduce batch size (try batch=4 or batch=2)")
        print("2. No GPU: Change device='cpu' (will be slower)")
        print("3. Dataset issue: Check data.yaml paths are correct")

if __name__ == "__main__":
    main()
