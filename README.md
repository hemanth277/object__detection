# Object Detection + Image Recognition

Real-time object detection with AI-powered image analysis.

## Quick Start

### 1. Set API Key (in PowerShell)
```powershell
$env:GEMINI_API_KEY = "your_api_key_here"
```

**Already set for this session!** ✓

### 2. Run Application
```bash
python object_detection.py
```

## Controls

- **'i'** - Analyze image (describe picture content)
- **'s'** - Save current frame
- **'c'** - Clear description
- **'q'** - Quit

## Features

### Object Detection (Automatic)
- Detects 80 types of physical objects
- Real-time bounding boxes and labels
- Works on objects in your environment

### Image Recognition (Press 'i')
- Analyzes pictures/photos using Gemini AI
- Describes what's shown in the image
- Perfect for analyzing printed photos or images on screens

## Usage Example

1. Run the script
2. Show a picture to your webcam
3. Press **'i'** to analyze
4. Read the AI description at the bottom
5. Press **'c'** to clear the description

## Files

- `object_detection.py` - Main application
- `setup_api_key.py` - API key setup helper
- `requirements.txt` - Dependencies

## Get API Key

Free API key: https://makersuite.google.com/app/apikey

---

**Note:** You may see a deprecation warning about `google.generativeai`. The app still works fine. To remove the warning, we can update to the newer `google.genai` package if needed.
