"""
Helper script to organize your annotated dataset for YOLO training

This script helps you:
1. Copy your annotated images from labelImg to a proper dataset structure
2. Create a classes.txt file with all your object classes
3. Prepare your dataset for YOLO model training

Your current annotation work:
- Images: C:\\Users\\admin\\Desktop\\models_dataset\\
- Labels: C:\\Users\\admin\\Desktop\\new models\\
"""

import os
import shutil
from pathlib import Path

# Define source and destination paths
SOURCE_IMAGES = r"C:\Users\admin\Desktop\models_dataset"
SOURCE_LABELS = r"C:\Users\admin\Desktop\new models"
DEST_BASE = r"C:\Users\admin\Desktop\object detection\dataset"

# Your annotated classes (from your labelImg work)
CLASSES = [
    "motor",
    "wire or cable",
    "power supply box (12kv DC)",
    "motor holder",
    "stands / holders",
    "weight",
    "sliding weight",
    "scale",
    "angle scale",
    "center pivot",
    "base platform",
    "connector",
    "switch",
    "weight holder",
    "wire or chains"
]

def create_dataset_structure():
    """Create the proper directory structure for YOLO dataset"""
    print("Creating dataset directory structure...")
    
    dirs = [
        f"{DEST_BASE}/images/train",
        f"{DEST_BASE}/images/val",
        f"{DEST_BASE}/labels/train",
        f"{DEST_BASE}/labels/val"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  [OK] Created: {dir_path}")
    
    print()

def copy_dataset(train_split=0.8):
    """
    Copy images and labels to dataset folders
    
    Args:
        train_split: Percentage of data to use for training (0.0 to 1.0)
    """
    print("Copying images and labels...")
    
    # Get all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(Path(SOURCE_IMAGES).glob(ext))
    
    if not image_files:
        print("  [WARNING] No images found in source directory!")
        return
    
    total = len(image_files)
    train_count = int(total * train_split)
    
    print(f"  Found {total} images")
    print(f"  Train: {train_count} images ({train_split*100:.0f}%)")
    print(f"  Val: {total - train_count} images ({(1-train_split)*100:.0f}%)")
    print()
    
    copied_train = 0
    copied_val = 0
    
    for i, img_path in enumerate(image_files):
        # Determine if this goes to train or val
        subset = "train" if i < train_count else "val"
        
        # Copy image
        dest_img = f"{DEST_BASE}/images/{subset}/{img_path.name}"
        shutil.copy2(img_path, dest_img)
        
        # Copy corresponding label file
        label_name = img_path.stem + ".txt"
        label_path = Path(SOURCE_LABELS) / label_name
        
        if label_path.exists():
            dest_label = f"{DEST_BASE}/labels/{subset}/{label_name}"
            shutil.copy2(label_path, dest_label)
            
            if subset == "train":
                copied_train += 1
            else:
                copied_val += 1
        else:
            print(f"  [WARNING] Label not found for {img_path.name}")
    
    print(f"  [OK] Copied {copied_train} training pairs")
    print(f"  [OK] Copied {copied_val} validation pairs")
    print()

def create_classes_file():
    """Create classes.txt file with all class names"""
    print("Creating classes.txt file...")
    
    classes_path = f"{DEST_BASE}/classes.txt"
    with open(classes_path, 'w', encoding='utf-8') as f:
        for class_name in CLASSES:
            f.write(f"{class_name}\n")
    
    print(f"  [OK] Created {classes_path}")
    print(f"  [OK] Added {len(CLASSES)} classes")
    print()

def create_data_yaml():
    """Create data.yaml file for YOLOv5/v8 training"""
    print("Creating data.yaml file...")
    
    yaml_content = f"""# Dataset configuration for YOLO training
# Generated for physics equipment detection

# Dataset paths (relative or absolute)
path: {DEST_BASE}
train: images/train
val: images/val

# Number of classes
nc: {len(CLASSES)}

# Class names
names:
"""
    for i, class_name in enumerate(CLASSES):
        yaml_content += f"  {i}: {class_name}\n"
    
    yaml_path = f"{DEST_BASE}/data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"  [OK] Created {yaml_path}")
    print()

def main():
    print("=" * 60)
    print("Dataset Organization Tool")
    print("=" * 60)
    print()
    
    # Check if source directories exist
    if not os.path.exists(SOURCE_IMAGES):
        print(f"[ERROR] Images directory not found: {SOURCE_IMAGES}")
        return
    
    if not os.path.exists(SOURCE_LABELS):
        print(f"[ERROR] Labels directory not found: {SOURCE_LABELS}")
        return
    
    print("Source directories:")
    print(f"  Images: {SOURCE_IMAGES}")
    print(f"  Labels: {SOURCE_LABELS}")
    print()
    
    print("Destination:")
    print(f"  Dataset: {DEST_BASE}")
    print()
    
    # Create dataset structure
    create_dataset_structure()
    
    # Copy files
    copy_dataset(train_split=0.8)
    
    # Create classes file
    create_classes_file()
    
    # Create data.yaml for YOLO training
    create_data_yaml()
    
    print("=" * 60)
    print("[SUCCESS] Dataset organization complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Review the dataset in:", DEST_BASE)
    print("2. Use data.yaml for YOLOv5/v8 training")
    print("3. Follow the guide in dataset_guide.md for training")
    print()

if __name__ == "__main__":
    main()
