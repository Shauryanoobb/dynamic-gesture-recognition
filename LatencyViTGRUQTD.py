import tensorflow as tf
from tensorflow.keras.models import Model
from keras_vision.MobileViT_v1 import build_MobileViT_v1
import numpy as np
import time

# ===== CONFIG =====
GESTURES = ['Again', 'FistHalt', 'Shoot', 'Sign', 'Swipe', 'Talk', 'Teacher', 'ThumbsUp', 'Wave', 'ZoomIn']
frames = 16
img_size = 112
num_classes = len(GESTURES)

# ===== QuantizableTimeDistributed class =====
@tf.keras.utils.register_keras_serializable(package="Custom", name="QuantizableTimeDistributed")
class QuantizableTimeDistributed(tf.keras.layers.Wrapper):
    """
    A TimeDistributed-like wrapper implemented by reshaping:
      (batch, time, ...) -> (batch * time, ...)
    call the wrapped layer once, then reshape back to (batch, time, ...).
    This avoids Python loops / TensorList ops and is more TFLite-friendly.
    """

    def __init__(self, layer: tf.keras.layers.Layer, **kwargs):
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError("`QuantizableTimeDistributed` must wrap a "
                             "`tf.keras.layers.Layer` instance.")
        super().__init__(layer, **kwargs)
        self.supports_masking = True

    def _get_child_input_shape(self, input_shape):
        """Return the child input shape expected by the wrapped layer."""
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) < 3:
            raise ValueError("`QuantizableTimeDistributed` expects an input "
                             "shape with rank >= 3 (batch, time, ...). "
                             f"Got: {input_shape}")
        return (None, *input_shape[2:])

    def build(self, input_shape):
        child_input_shape = self._get_child_input_shape(input_shape)
        super().build(child_input_shape)

    def compute_output_shape(self, input_shape):
        child_input_shape = self._get_child_input_shape(input_shape)
        child_output_shape = self.layer.compute_output_shape(child_input_shape)
        return (child_output_shape[0], input_shape[1], *child_output_shape[1:])

    def call(self, inputs, training=None, mask=None):
        input_shape = tf.shape(inputs)
        batch = input_shape[0]
        time = input_shape[1]
        tail = input_shape[2:]
        new_batch = batch * time
        new_shape = tf.concat([[new_batch], tail], axis=0)
        reshaped_inputs = tf.reshape(inputs, new_shape)

        reshaped_mask = None
        if mask is not None:
            mask = tf.convert_to_tensor(mask)
            def _reshape_mask():
                mshape = tf.shape(mask)
                rest = mshape[2:]
                m_new_shape = tf.concat([[new_batch], rest], axis=0)
                return tf.reshape(mask, m_new_shape)
            reshaped_mask = _reshape_mask()

        call_kwargs = {}
        if self.layer._call_has_training_arg:
            call_kwargs["training"] = training
        if self.layer._call_has_mask_arg and reshaped_mask is not None:
            call_kwargs["mask"] = reshaped_mask

        outputs = self.layer(reshaped_inputs, **call_kwargs)
        out_shape = tf.shape(outputs)
        target_shape = tf.concat([[batch, time], out_shape[1:]], axis=0)
        outputs_reshaped = tf.reshape(outputs, target_shape)
        return outputs_reshaped

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        mask = tf.convert_to_tensor(mask)
        mshape = tf.shape(mask)
        batch = mshape[0]
        time = mshape[1]
        rest = mshape[2:]
        new_mask_shape = tf.concat([[batch * time], rest], axis=0) if tf.rank(mask) > 2 else tf.concat([[batch * time]], axis=0)
        mask_reshaped = tf.reshape(mask, new_mask_shape)

        child_mask = None
        if self.layer._call_has_mask_arg:
            child_mask = self.layer.compute_mask(mask_reshaped)
        if child_mask is None:
            return None

        child_shape = tf.shape(child_mask)
        target_shape = tf.concat([[batch, time], child_shape[1:]], axis=0)
        return tf.reshape(child_mask, target_shape)

    def get_config(self):
        config = super().get_config()
        return config


# ===== Build the same architecture =====
base = build_MobileViT_v1(
    model_type="XXS",
    pretrained=False,
    include_top=False,
    input_shape=(img_size, img_size, 3),
    num_classes=0
)

video_input = tf.keras.Input((frames, img_size, img_size, 3))
x = QuantizableTimeDistributed(base)(video_input)
x = QuantizableTimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
x = tf.keras.layers.GRU(32, return_sequences=False)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = Model(video_input, output)

# ===== Load pretrained weights =====
model.load_weights("models/GruVitQTD/10signs_5_people_QT_last_112_mobilevit_gru.h5")
model.trainable = False

print("✅ Model loaded successfully.")
print(f"📊 Model input shape: {model.input_shape}")

# ===== Benchmark Configuration =====
WARMUP_RUNS = 10  # Number of warmup runs to stabilize inference
BENCHMARK_RUNS = 200  # Number of runs for averaging

# ===== Create dummy input =====
dummy_input = np.random.rand(1, frames, img_size, img_size, 3).astype('float32')
print(f"🎯 Dummy input shape: {dummy_input.shape}")

# ===== Warmup runs =====
print(f"\n🔥 Running {WARMUP_RUNS} warmup iterations...")
for i in range(WARMUP_RUNS):
    _ = model.predict(dummy_input, verbose=0)

# ===== Benchmark runs =====
print(f"\n⏱️  Running {BENCHMARK_RUNS} benchmark iterations...")
inference_times = []

for i in range(BENCHMARK_RUNS):
    start_time = time.time()
    _ = model.predict(dummy_input, verbose=0)
    end_time = time.time()
    
    inference_time_ms = (end_time - start_time) * 1000
    inference_times.append(inference_time_ms)
    
    if (i + 1) % 10 == 0:
        print(f"  Progress: {i + 1}/{BENCHMARK_RUNS}")

# ===== Calculate statistics =====
avg_inference_time = np.mean(inference_times)
std_inference_time = np.std(inference_times)
min_inference_time = np.min(inference_times)
max_inference_time = np.max(inference_times)
fps = 1000 / avg_inference_time

# ===== Display results =====
print("\n" + "="*60)
print("📈 INFERENCE SPEED BENCHMARK RESULTS")
print("="*60)
print(f"Average inference time: {avg_inference_time:.2f} ms")
print(f"Standard deviation:     {std_inference_time:.2f} ms")
print(f"Min inference time:     {min_inference_time:.2f} ms")
print(f"Max inference time:     {max_inference_time:.2f} ms")
print(f"Frames per second:      {fps:.2f} FPS")
print("="*60)