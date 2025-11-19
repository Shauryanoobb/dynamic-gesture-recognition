import time
import numpy as np
import tensorflow as tf
from collections import deque

# ------------------------------
# CONFIG
# ------------------------------
FRAMES = 16
IMG_SIZE = 112
MODEL_PATH = "models/cnn/3dCNN9Signs.keras"
NUM_WARMUP = 10
NUM_ITERATIONS = 200

GESTURES = [
    'Again', 'Shoot', 'Sign', 'Swipe', 'Talk', 
    'Teacher', 'ThumbsUp', 'Wave', 'ZoomIn'
]

# ------------------------------
# LOAD MODEL
# ------------------------------
print(f"📦 Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

# ------------------------------
# TIMING STORAGE
# ------------------------------
inference_times = []

# ------------------------------
# GENERATE RANDOM INPUT SEQUENCE
# ------------------------------
def generate_random_sequence():
    """Generate random 16-frame sequence."""
    # Shape: (16, 112, 112, 3) - normalized to [0, 1]
    sequence = np.random.rand(FRAMES, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    return sequence

# ------------------------------
# INFERENCE FUNCTION
# ------------------------------
def run_inference(sequence):
    """Run inference and measure time."""
    # Add batch dimension: (1, 16, 112, 112, 3)
    x = np.expand_dims(sequence, axis=0)
    
    # ---- TIME START ----
    t1 = time.time()
    
    pred = model.predict(x, verbose=0)[0]
    
    # ---- TIME END ----
    t2 = time.time()
    
    # Timing in ms
    infer_ms = (t2 - t1) * 1000
    
    pred_class = np.argmax(pred)
    confidence = float(pred[pred_class])
    
    return GESTURES[pred_class], confidence, infer_ms

# ------------------------------
# WARMUP PHASE
# ------------------------------
print(f"\n🔥 Running {NUM_WARMUP} warmup iterations...")
for i in range(NUM_WARMUP):
    sequence = generate_random_sequence()
    run_inference(sequence)
print("✅ Warmup complete")

# ------------------------------
# BENCHMARK PHASE
# ------------------------------
print(f"\n🔬 Starting inference timing test ({NUM_ITERATIONS} iterations)...")
print("-" * 70)

for i in range(NUM_ITERATIONS):
    # Generate random 16-frame sequence
    sequence = generate_random_sequence()
    
    # Run inference
    gesture, conf, infer_ms = run_inference(sequence)
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
else:
    print("❌ No inferences were made.")

print("\n✅ Benchmark complete!")