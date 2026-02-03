import urllib.request
import os

def download_file(url, filename):
    """Download a file from URL"""
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✓ {filename} downloaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Error downloading {filename}: {e}")
        return False

def main():
    # Model files URLs
    prototxt_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt"
    caffemodel_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel"
    
    # Download files
    print("Downloading MobileNet SSD model files...\n")
    
    success1 = download_file(prototxt_url, "MobileNetSSD_deploy.prototxt")
    success2 = download_file(caffemodel_url, "MobileNetSSD_deploy.caffemodel")
    
    if success1 and success2:
        print("\n✅ All model files downloaded successfully!")
        print("You can now run: python object_detection.py")
    else:
        print("\n⚠️ Some files failed to download. Please check your internet connection.")

if __name__ == "__main__":
    main()
