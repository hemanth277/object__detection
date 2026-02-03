"""
Object Detection with Custom Trained YOLOv8 Model

This script uses your custom-trained model to detect physics equipment
from the webcam in real-time.

Requirements:
    pip install ultralytics opencv-python

Usage:
    python detect_custom.py

Controls:
    - Press 'q' to quit
    - Press 's' to save screenshot
"""

from ultralytics import YOLO
import cv2

def main():
    print("=" * 60)
    print("Custom Physics Equipment Detection")
    print("=" * 60)
    print()
    
    # Path to your trained model
    # Update this after training completes
    model_path = "runs/detect/physics_equipment/weights/best.pt"
    
    print(f"Loading model: {model_path}")
    
    try:
        model = YOLO(model_path)
        print("[OK] Model loaded successfully!")
        print()
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        print()
        print("Make sure you've trained the model first using: python train_local.py")
        return
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return
    
    print("[OK] Webcam opened")
    print()
    print("=" * 60)
    print("CONTROLS:")
    print("  Press 'q' to quit")
    print("  Press 's' to save screenshot")
    print("=" * 60)
    print()
    print("Show your physics equipment to the camera!")
    print()
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        
        # Run inference
        results = model(frame, conf=0.25, verbose=False)
        
        # Draw results on frame
        annotated_frame = results[0].plot()
        
        # Display frame counter
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Physics Equipment Detection', annotated_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            filename = f"detection_{frame_count}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"Saved screenshot: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()
