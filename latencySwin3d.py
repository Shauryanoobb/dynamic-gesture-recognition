import time
import torch
import torch.nn as nn
import numpy as np
from torchvision.models.video import swin3d_t

# ------------------------------
# CONFIG
# ------------------------------
FRAMES = 16
IMG_SIZE = 112
MODEL_PATH = "models/pretrainedVideoVits/best_swin3d_model (1).pth"
NUM_WARMUP = 10
NUM_ITERATIONS = 200
device = "cuda" if torch.cuda.is_available() else "cpu"

GESTURES = [
    'Again', 'FistHalt', 'Shoot', 'Sign', 'Swipe', 
    'Talk', 'Teacher', 'ThumbsUp', 'Wave', 'ZoomIn'
]

# ------------------------------
# LOAD MODEL
# ------------------------------
print(f"📦 Loading model: {MODEL_PATH}")
print(f"🖥️  Device: {device}")

model = swin3d_t(pretrained=False)
num_features = model.head.in_features
model.head = nn.Linear(num_features, len(GESTURES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print("✅ Model loaded successfully!")
print(f"Model output classes: {len(GESTURES)}")

# ------------------------------
# TIMING STORAGE
# ------------------------------
inference_times = []

# ------------------------------
# GENERATE RANDOM INPUT SEQUENCE
# ------------------------------
def generate_random_sequence():
    """Generate random 16-frame sequence in PyTorch format."""
    # Shape: (1, 3, T, H, W) = (1, 3, 16, 112, 112)
    # Random values in [0, 1] range
    video = torch.rand(1, 3, FRAMES, IMG_SIZE, IMG_SIZE, dtype=torch.float32).to(device)
    return video

# ------------------------------
# INFERENCE FUNCTION
# ------------------------------
def run_inference(video):
    """Run inference and measure time."""
    
    # ---- TIME START ----
    if device == "cuda":
        torch.cuda.synchronize()  # Ensure all operations are complete
    
    t1 = time.time()
    
    with torch.no_grad():
        preds = model(video)
        probs = torch.softmax(preds, dim=1)[0]
        conf, pred_class = torch.max(probs, dim=0)
    
    if device == "cuda":
        torch.cuda.synchronize()  # Wait for GPU to finish
    
    # ---- TIME END ----
    t2 = time.time()
    
    # Timing in ms
    infer_ms = (t2 - t1) * 1000
    
    pred_class = pred_class.item()
    confidence = conf.item()
    
    return GESTURES[pred_class], confidence, infer_ms

# ------------------------------
# WARMUP PHASE
# ------------------------------
print(f"\n🔥 Running {NUM_WARMUP} warmup iterations...")
for i in range(NUM_WARMUP):
    video = generate_random_sequence()
    run_inference(video)
print("✅ Warmup complete")

# ------------------------------
# BENCHMARK PHASE
# ------------------------------
print(f"\n🔬 Starting inference timing test ({NUM_ITERATIONS} iterations)...")
print("-" * 70)

for i in range(NUM_ITERATIONS):
    # Generate random 16-frame sequence
    video = generate_random_sequence()
    
    # Run inference
    gesture, conf, infer_ms = run_inference(video)
    inference_times.append(infer_ms)
    
    # Print every 20 iterations
    if (i + 1) % 20 == 0:
        print(f"[{i+1:3d}/{NUM_ITERATIONS}] {gesture:12s} ({conf:.2f})  |  Time = {infer_ms:.2f} ms")

# ------------------------------
# STATISTICS
# ------------------------------
print("\n" + "=" * 70)
print("INFERENCE LATENCY STATISTICS")
print("=" * 70)

if inference_times:
    inference_times = np.array(inference_times)
    
    print(f"Device:               {device.upper()}")
    print(f"Total inferences:     {len(inference_times)}")
    print(f"Mean inference time:  {np.mean(inference_times):.2f} ms")
    print(f"Median inference:     {np.median(inference_times):.2f} ms")
    print(f"Std deviation:        {np.std(inference_times):.2f} ms")
    print(f"Min inference time:   {np.min(inference_times):.2f} ms")
    print(f"Max inference time:   {np.max(inference_times):.2f} ms")
    
    print(f"\nPercentiles:")
    print(f"  P50 (median):       {np.percentile(inference_times, 50):.2f} ms")
    print(f"  P90:                {np.percentile(inference_times, 90):.2f} ms")
    print(f"  P95:                {np.percentile(inference_times, 95):.2f} ms")
    print(f"  P99:                {np.percentile(inference_times, 99):.2f} ms")
    
    # Throughput
    avg_time_s = np.mean(inference_times) / 1000
    throughput = 1.0 / avg_time_s if avg_time_s > 0 else 0
    print(f"\nThroughput:           {throughput:.2f} sequences/sec")
    print(f"FPS equivalent:       {throughput * FRAMES:.2f} frames/sec")
    
    # Memory usage (if CUDA)
    if device == "cuda":
        print(f"\nGPU Memory:")
        print(f"  Allocated:          {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"  Reserved:           {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
else:
    print("❌ No inferences were made.")

print("\n✅ Benchmark complete!")