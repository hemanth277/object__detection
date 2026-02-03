"""
Helper script to set up Gemini API key for image recognition
"""

import os
import sys

def setup_api_key():
    print("=" * 60)
    print("Gemini API Key Setup")
    print("=" * 60)
    print()
    print("To use image recognition, you need a free Gemini API key.")
    print()
    print("Steps:")
    print("1. Visit: https://makersuite.google.com/app/apikey")
    print("2. Sign in with your Google account")
    print("3. Click 'Create API Key'")
    print("4. Copy the API key")
    print()
    
    api_key = input("Paste your API key here (or press Enter to skip): ").strip()
    
    if not api_key:
        print("\nNo API key provided. You can run this script again later.")
        return
    
    # Save to config file
    try:
        with open('.api_key', 'w') as f:
            f.write(api_key)
        print("\n✓ API key saved to .api_key file!")
    except Exception as e:
        print(f"\n✗ Error saving API key: {e}")
        return
    
    # Also set environment variable for current session
    os.environ['GEMINI_API_KEY'] = api_key
    
    print("\n✓ API key is ready to use!")
    print("\nYou can now run: python object_detection.py")
    print("\nNote: The API key is saved in .api_key file and will work automatically.")

if __name__ == "__main__":
    setup_api_key()
