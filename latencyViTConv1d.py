import tensorflow as tf
from tensorflow.keras import layers, Model
from keras_vision.MobileViT_v1 import build_MobileViT_v1
import numpy as np
import time
from statistics import mean, stdev, median

# === CONFIG ===
GESTURES = ['Again', 'FistHalt', 'Shoot', 'Sign', 'Swipe', 'Talk', 'Teacher', 'ThumbsUp', 'Wave', 'ZoomIn']
frames = 16
img_size = 112
num_classes = len(GESTURES)

# Path to your trained model weights
WEIGHTS_PATH = "models/VitConv1D/reshapewala.h5"  # Change this to your weights file

# === Build Model Architecture ===
print("🔧 Building model architecture...")

base = build_MobileViT_v1(
    model_type="XXS",
    pretrained=False,  # Set to False when loading trained weights
    include_top=False,
    num_classes=0,
    input_shape=(img_size, img_size, 3)
)

video_input = tf.keras.Input((frames, img_size, img_size, 3))

# Step 1: Merge frames into channels
x = layers.Reshape((img_size, img_size, frames * 3))(video_input)

# Step 2: 1x1 Conv to compress temporal channels
x = layers.Conv2D(3, (1, 1), activation="relu", padding="same")(x)

# Step 3: Pass through MobileViT backbone
x = base(x)

# Step 4: Global pooling (spatial)
x = layers.GlobalAveragePooling2D()(x)

# Step 5: Temporal feature learning
x = layers.RepeatVector(frames)(x)
x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)

# Step 6: Temporal pooling
x = layers.GlobalAveragePooling1D()(x)

# Step 7: Classification head
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(num_classes, activation="softmax")(x)

model = Model(video_input, output, name="MobileViT_XXS_TemporalConv_QFixed")

# === Load Trained Weights ===
print(f"📦 Loading weights from {WEIGHTS_PATH}...")
try:
    model.load_weights(WEIGHTS_PATH)
    print("✅ Weights loaded successfully!")
except Exception as e:
    print(f"❌ Error loading weights: {e}")
    print("⚠️  Continuing with random weights for latency testing...")

model.trainable = False

# === Benchmark Configuration ===
WARMUP_RUNS = 20
BENCHMARK_RUNS = 200

print(f"\n📊 Model Summary:")
print(f"   Input shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")
print(f"   Total params: {model.count_params():,}")

# === Create Dummy Input ===
print(f"\n🎯 Creating dummy input with shape: (1, {frames}, {img_size}, {img_size}, 3)")
dummy_input = np.random.rand(1, frames, img_size, img_size, 3).astype('float32')

# === Warmup Phase ===
print(f"\n🔥 Warming up with {WARMUP_RUNS} iterations...")
for i in range(WARMUP_RUNS):
    _ = model.predict(dummy_input, verbose=0)
    if (i + 1) % 5 == 0:
        print(f"   Warmup progress: {i + 1}/{WARMUP_RUNS}")

# === Benchmark Phase ===
print(f"\n⏱️  Running {BENCHMARK_RUNS} benchmark iterations...")
latencies_ms = []

for i in range(BENCHMARK_RUNS):
    start = time.perf_counter()
    predictions = model.predict(dummy_input, verbose=0)
    end = time.perf_counter()
    
    latency_ms = (end - start) * 1000
    latencies_ms.append(latency_ms)
    
    if (i + 1) % 20 == 0:
        print(f"   Progress: {i + 1}/{BENCHMARK_RUNS} | Current: {latency_ms:.2f} ms")

# === Calculate Statistics ===
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

# === Display Results ===
print("\n" + "=" * 70)
print("📈 LATENCY BENCHMARK RESULTS")
print("=" * 70)
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
print(f"Real-time capable:      {'✅ Yes' if avg_latency < 33.33 else '❌ No'} (for 30 FPS)")
print("=" * 70)

# === Additional Analysis ===
print("\n📊 LATENCY DISTRIBUTION:")
bins = [0, 20, 40, 60, 80, 100, float('inf')]
bin_labels = ['0-20ms', '20-40ms', '40-60ms', '60-80ms', '80-100ms', '100ms+']
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

print("\n✅ Benchmark completed!")