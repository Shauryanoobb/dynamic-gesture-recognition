import torch
import torch.nn as nn
from torchvision.models.video import s3d
import numpy as np

device = "cpu"
GESTURES = ['FistHalt', 'Swipe', 'ThumbsUp', 'Wave', 'ZoomIn']

model = s3d(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Conv3d(1024, len(GESTURES), kernel_size=1, stride=1, bias=True)
)
model.load_state_dict(torch.load("s3d_gesture.pth", map_location=device))
model.eval()
print("✅ Model loaded successfully")

# Manual quantization to int8
quantized_state_dict = {}
scales = {}

for name, param in model.state_dict().items():
    if param.dtype == torch.float32 or param.dtype == torch.float16:
        # Calculate scale for symmetric quantization
        max_val = param.abs().max().item()
        scale = max_val / 127.0 if max_val > 0 else 1.0
        
        # Quantize to int8
        quantized = torch.clamp(torch.round(param / scale), -128, 127).to(torch.int8)
        
        quantized_state_dict[name] = quantized
        scales[name] = scale
        print(f"Quantized {name}: {param.dtype} -> int8")
    else:
        quantized_state_dict[name] = param

# Save quantized weights and scales
torch.save({
    'state_dict': quantized_state_dict,
    'scales': scales,
    'gestures': GESTURES
}, "s3d_gesture_int8.pth")
print("✅ Quantized model saved")

# Verify size reduction
import os
original_size = os.path.getsize("s3d_gesture.pth") / (1024**2)
quantized_size = os.path.getsize("s3d_gesture_int8.pth") / (1024**2)
print(f"\nOriginal: {original_size:.2f} MB")
print(f"Quantized: {quantized_size:.2f} MB")
print(f"Compression: {original_size/quantized_size:.2f}x")