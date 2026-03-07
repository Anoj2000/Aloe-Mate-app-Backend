import torch
import os

model_path = r"e:\aloe-maturity-system\backend\app\cnn\best_aloe_vera_model_4class_fixed.pth"

try:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        keys = sorted(checkpoint.keys())
        
        print("Features 1 Keys:")
        for k in keys:
            if k.startswith("features.1."):
                print(f"{k} -> {list(checkpoint[k].shape)}")
                
        print("\nFeatures 2 Keys:")
        for k in keys:
            if k.startswith("features.2."):
                print(f"{k} -> {list(checkpoint[k].shape)}")
                
    else:
        print(f"Not a dict: {type(checkpoint)}")
except Exception as e:
    print(f"Error: {e}")
