import cv2
import numpy as np
import urllib.request
import os

def download_yolo_files():
    """Download YOLO model files if they don't exist"""
    # Try YOLOv3-tiny first (smaller, faster, easier to download)
    files = {
        'yolov3-tiny.weights': 'https://pjreddie.com/media/files/yolov3-tiny.weights',
        'yolov3-tiny.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg',
        'coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
    }
    
    print("Attempting to download YOLOv3-tiny model (smaller and faster)...")
    
    for filename, url in files.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                # Add headers to avoid 403 errors
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    with open(filename, 'wb') as out_file:
                        out_file.write(response.read())
                print(f"Downloaded {filename} successfully!")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                return False
        else:
            print(f"{filename} already exists.")
    
    return True

def load_yolo_model():
    """Load YOLO model and class names"""
    # Load class names
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Load YOLO-tiny (faster and lighter)
    net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Generate random colors for each class
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    return net, classes, colors, output_layers

def detect_objects(frame, net, classes, colors, output_layers, confidence_threshold=0.3):
    """Detect objects in a frame"""
    height, width, channels = frame.shape
    
    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-max suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
    
    # Draw bounding boxes and labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with background
            label_text = f"{label}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label_text, font, 0.6, 2)
            
            # Draw background rectangle for text
            cv2.rectangle(frame, (x, y - 25), (x + label_size[0], y), color, -1)
            
            # Draw text
            cv2.putText(frame, label_text, (x, y - 5), font, 0.6, (0, 0, 0), 2)
    
    return frame, len(indexes)



def main():
    """
    Real-time object detection using webcam with YOLOv3.
    Detects 80 different object classes from COCO dataset.
    """
    
    print("=" * 60)
    print("YOLO Object Detection - Real-time Webcam")
    print("=" * 60)
    
    # Download YOLO files if needed
    print("\nChecking for YOLO model files...")
    if not download_yolo_files():
        print("Failed to download model files. Please check your internet connection.")
        return
    
    # Initialize webcam with DirectShow backend (more stable on Windows)
    print("\nInitializing webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("DirectShow failed, trying default backend...")
        cap = cv2.VideoCapture(0)
        
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        print("Please check:")
        print("1. Webcam is properly connected")
        print("2. No other application is using the webcam")
        print("3. Webcam permissions are granted")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frames
    
    print("Webcam opened successfully!")
    
    # Warm up the camera (read a few frames to stabilize)
    print("Warming up camera...")
    for i in range(5):
        ret, _ = cap.read()
        if ret:
            break
    
    if not ret:
        print("Error: Could not read from webcam")
        cap.release()
        return
    
    # Load YOLO model
    print("\nLoading YOLO model (this may take a moment)...")
    try:
        net, classes, colors, output_layers = load_yolo_model()
        print(f"Model loaded successfully! Can detect {len(classes)} object types.")
        model_loaded = True
    except Exception as e:
        print(f"Error loading model: {e}")
        model_loaded = False
        cap.release()
        return
    
    print("\n" + "=" * 60)
    print("CONTROLS:")
    print("  Press 'q' to quit")
    print("  Press 's' to save current frame")
    print("=" * 60)
    print()
    
    frame_count = 0
    failed_frames = 0
    max_failed_frames = 10  # Allow some failed frames before giving up
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            failed_frames += 1
            print(f"Warning: Failed to capture frame ({failed_frames}/{max_failed_frames})")
            if failed_frames >= max_failed_frames:
                print("Error: Too many failed frames. Exiting...")
                break
            continue
        
        # Reset failed frame counter on successful read
        failed_frames = 0
        frame_count += 1
        
        # Perform object detection every frame
        if model_loaded:
            frame, num_objects = detect_objects(frame, net, classes, colors, output_layers, confidence_threshold=0.25)
            
            # Display object count and frame number
            cv2.putText(frame, f"Objects: {num_objects} | Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display instructions
        instructions = "Press 'q' to quit | 's' to save"
        cv2.putText(frame, instructions, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the frame
        cv2.imshow('YOLO Object Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            filename = f"detection_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved frame as {filename}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed.")
    print("\nThank you for using YOLO Object Detection!")

if __name__ == "__main__":
    main()
