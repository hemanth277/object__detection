# How to Train Your Custom Model

## Why Your Objects Aren't Detected

The current `object_detection.py` uses a **pre-trained YOLO model** trained on the **COCO dataset** (80 common objects).

Your physics equipment objects (motor, power supply, weights, etc.) are **NOT** in the COCO dataset, so they won't be detected!

## Solution: Train a Custom Model

### Method 1: Google Colab (Recommended - FREE GPU)

#### Step 1: Upload Dataset to Google Drive
1. Compress your dataset folder:
   ```powershell
   Compress-Archive -Path "C:\Users\admin\Desktop\object detection\dataset" -DestinationPath "dataset.zip"
   ```

2. Upload `dataset.zip` to Google Drive
3. Unzip it in Google Drive (right-click → Extract)

#### Step 2: Open Google Colab
1. Go to: https://colab.research.google.com
2. Sign in with your Google account
3. Upload the `train_yolov8_colab.ipynb` file
4. Change Runtime to GPU: Runtime → Change runtime type → GPU

#### Step 3: Run Training
1. Update the `dataset_path` in the notebook to match your Google Drive path
2. Run all cells (Runtime → Run all)
3. Training will take 15-30 minutes
4. Download the trained model: `runs/detect/physics_equipment_detector/weights/best.pt`

#### Step 4: Use Your Custom Model

Once trained, update `object_detection.py` to use your custom model:

```python
# Instead of YOLOv3-tiny, use YOLOv8 with your trained weights
from ultralytics import YOLO

# Load your custom trained model
model = YOLO('best.pt')  # Your downloaded model

# Run inference on webcam
results = model.predict(source=0, show=True, conf=0.25)
```

### Method 2: Use YOLOv8 Locally (Requires Good GPU)

If you have a NVIDIA GPU with CUDA:

```powershell
# Install YOLOv8
pip install ultralytics

# Train the model (from your dataset directory)
cd "C:\Users\admin\Desktop\object detection"
yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

## Expected Results

After training on your 14 annotated images:
- **Training time**: 15-30 minutes on Google Colab GPU
- **Model performance**: Should detect your physics equipment with ~70-90% accuracy
- **Classes detected**: All 15 of your custom classes

## Quick Alternative: Pre-annotate More Images

Currently you only have **14 images** annotated. For better results:

1. Annotate at least **50-100 images** per class using labelImg
2. Use data augmentation to increase dataset size
3. Ensure variety (different angles, lighting, backgrounds)

## Need Help?

- **YOLOv8 Documentation**: https://docs.ultralytics.com
- **Google Colab Tutorial**: https://colab.research.google.com/notebooks/intro.ipynb
- The `train_yolov8_colab.ipynb` file has step-by-step instructions
