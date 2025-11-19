import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from keras_vision.MobileViT_v1 import build_MobileViT_v1
import time
from statistics import mean, stdev, median

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 112
SEQ_LEN = 16
GESTURES = ['Again', 'Shoot', 'Sign', 'Swipe', 'Talk', 'Teacher', 'ThumbsUp', 'Wave', 'ZoomIn']
MODEL_PATH = "models/LstmVit/mobilevit_lstm_9.keras"

# Benchmark settings
WARMUP_RUNS = 20
BENCHMARK_RUNS = 200

# -----------------------------
# REBUILD MODEL ARCHITECTURE
# -----------------------------
print("🔧 Building model architecture...")

backbone = build_MobileViT_v1(
    model_type="XXS",
    pretrained=False,  # Set to False when loading trained weights
    include_top=False,
    num_classes=0,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
backbone.trainable = False

video_in = layers.Input((SEQ_LEN, IMG_SIZE, IMG_SIZE, 3))
x = layers.TimeDistributed(backbone)(video_in)
x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
x = layers.LSTM(128, dropout=0.3, return_sequences=False)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(len(GESTURES), activation='softmax')(x)

model = Model(video_in, out, name="MobileViT_LSTM_Gesture")

# Build with dummy input to initialize weights
print("🔨 Initializing model...")
dummy = tf.random.uniform((1, SEQ_LEN, IMG_SIZE, IMG_SIZE, 3))
_ = model(dummy)

# Load trained weights
print(f"📦 Loading weights from {MODEL_PATH}...")
try:
    model.load_weights(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading weights: {e}")
    print("⚠️  Continuing with random weights for latency testing...")

# -----------------------------
# MODEL INFO
# -----------------------------
print(f"\n📊 Model Summary:")
print(f"   Input shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")
print(f"   Total params: {model.count_params():,}")

# -----------------------------
# CREATE DUMMY INPUT
# -----------------------------
print(f"\n🎯 Creating dummy input with shape: (1, {SEQ_LEN}, {IMG_SIZE}, {IMG_SIZE}, 3)")
dummy_input = np.random.rand(1, SEQ_LEN, IMG_SIZE, IMG_SIZE, 3).astype('float32')

# -----------------------------
# WARMUP PHASE
# -----------------------------
print(f"\n🔥 Warming up with {WARMUP_RUNS} iterations...")
for i in range(WARMUP_RUNS):
    _ = model.predict(dummy_input, verbose=0)
    if (i + 1) % 5 == 0:
        print(f"   Warmup progress: {i + 1}/{WARMUP_RUNS}")

# -----------------------------
# BENCHMARK PHASE
# -----------------------------
print(f"\n⏱️  Running {BENCHMARK_RUNS} benchmark iterations...")
latencies_ms = []

for i in range(BENCHMARK_RUNS):
    start = time.perf_counter()
    predictions = model.predict(dummy_input, verbose=0)
    end = time.perf_counter()
    
    latency_ms = (end - start) * 1000
    latencies_ms.append(latency_ms)
    
    if (i + 1) % 20 == 0:
        current_avg = mean(latencies_ms)
        print(f"   Progress: {i + 1}/{BENCHMARK_RUNS} | Current: {latency_ms:.2f} ms | Avg so far: {current_avg:.2f} ms")

# -----------------------------
# CALCULATE STATISTICS
# -----------------------------
avg_latency = mean(latencies_ms)
std_latency = stdev(latencies_ms) if len(latencies_ms) > 1 else 0
median_latency = median(latencies_ms)
min_latency = min(latencies_ms)
max_latency = max(latencies_ms)
fps = 1000 / avg_latency

# Calculate percentiles
latencies_sorted = sorted(latencies_ms)
p50 = latencies_sorted[int(len(latencies_sorted) * 0.50)]
p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
print("\n" + "=" * 70)
print("📈 LATENCY BENCHMARK RESULTS")
print("=" * 70)
print(f"Model:                  {model.name}")
print(f"Number of runs:         {BENCHMARK_RUNS}")
print(f"Input shape:            {dummy_input.shape}")
print("-" * 70)
print(f"Average latency:        {avg_latency:.2f} ms")
print(f"Median latency:         {median_latency:.2f} ms")
print(f"Standard deviation:     {std_latency:.2f} ms")
print("-" * 70)
print(f"Min latency:            {min_latency:.2f} ms")
print(f"Max latency:            {max_latency:.2f} ms")
print("-" * 70)
print(f"50th percentile (P50):  {p50:.2f} ms")
print(f"95th percentile (P95):  {p95:.2f} ms")
print(f"99th percentile (P99):  {p99:.2f} ms")
print("-" * 70)
print(f"Throughput (FPS):       {fps:.2f} FPS")
print(f"Real-time @ 30 FPS:     {'✅ Yes' if avg_latency < 33.33 else '❌ No'} (< 33.33 ms needed)")
print(f"Real-time @ 15 FPS:     {'✅ Yes' if avg_latency < 66.67 else '❌ No'} (< 66.67 ms needed)")
print("=" * 70)

# -----------------------------
# LATENCY DISTRIBUTION
# -----------------------------
print("\n📊 LATENCY DISTRIBUTION:")
bins = [0, 20, 40, 60, 80, 100, 150, 200, float('inf')]
bin_labels = ['0-20ms', '20-40ms', '40-60ms', '60-80ms', '80-100ms', '100-150ms', '150-200ms', '200ms+']
counts = [0] * len(bin_labels)

for lat in latencies_ms:
    for i, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
        if lower <= lat < upper:
            counts[i] += 1
            break

for label, count in zip(bin_labels, counts):
    percentage = (count / BENCHMARK_RUNS) * 100
    bar = '█' * int(percentage / 2)
    print(f"{label:12s}: {bar:50s} {count:3d} ({percentage:5.1f}%)")

# -----------------------------
# FRAME SKIP ANALYSIS
# -----------------------------
print("\n🎬 FRAME SKIP RECOMMENDATIONS:")
frame_skip_options = [1, 2, 3, 4, 5]
print("-" * 70)
print(f"{'Frame Skip':<12} {'Effective FPS':<15} {'Latency Budget':<18} {'Status'}")
print("-" * 70)

for skip in frame_skip_options:
    effective_fps = 30 / skip  # Assuming 30 FPS camera
    latency_budget = 1000 / effective_fps
    status = "✅ OK" if avg_latency < latency_budget else "❌ Too slow"
    print(f"{skip:<12} {effective_fps:<15.1f} {latency_budget:<18.1f} ms     {status}")

print("-" * 70)

# -----------------------------
# INFERENCE BREAKDOWN
# -----------------------------
print("\n🔍 INFERENCE TIME BREAKDOWN (Estimated):")
print(f"   Per frame processing: ~{avg_latency / SEQ_LEN:.2f} ms/frame")
print(f"   Total sequence ({SEQ_LEN} frames): {avg_latency:.2f} ms")

# -----------------------------
# RECOMMENDATIONS
# -----------------------------
print("\n💡 RECOMMENDATIONS:")
if avg_latency < 33.33:
    print("   ✅ Model is fast enough for real-time (30 FPS) gesture recognition")
    print("   ✅ You can use frame_skip=1 for maximum responsiveness")
elif avg_latency < 66.67:
    print("   ⚠️  Model can run at 15 FPS (frame_skip=2 recommended)")
    print("   💡 Consider optimizing with TFLite for better performance")
elif avg_latency < 100:
    print("   ⚠️  Model is moderately slow (frame_skip=3-4 recommended)")
    print("   💡 Consider using quantization or a smaller backbone")
else:
    print("   ❌ Model is too slow for real-time gesture recognition")
    print("   💡 Recommendations:")
    print("      - Use TFLite with quantization")
    print("      - Reduce IMG_SIZE or SEQ_LEN")
    print("      - Use a smaller MobileViT variant")

print("\n✅ Benchmark completed!")