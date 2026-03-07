import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

from app.cnn.model import _get_classes, CLASS_INDICES_PATH

def test_mapping():
    print(f"Checking path: {CLASS_INDICES_PATH}")
    print(f"File exists: {os.path.exists(CLASS_INDICES_PATH)}")
    
    classes = _get_classes()
    print(f"Loaded classes: {classes}")

if __name__ == "__main__":
    test_mapping()
