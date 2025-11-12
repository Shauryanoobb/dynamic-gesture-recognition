import tensorflow as tf
from tensorflow.keras import layers, Model
from keras_vision.MobileViT_v1 import build_MobileViT_v1

# === CONFIG ===
GESTURES = ['FistHalt', 'Swipe', 'ThumbsUp', 'Wave', 'ZoomIn']
frames = 16
img_size = 112
num_classes = len(GESTURES)


# quantizable_time_distributed_tfkeras.py
import tensorflow as tf
from typing import Tuple, Union, Optional

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

    def _get_child_input_shape(self, input_shape: Union[Tuple, list]):
        """Return the child input shape expected by the wrapped layer."""
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) < 3:
            raise ValueError("`QuantizableTimeDistributed` expects an input "
                             "shape with rank >= 3 (batch, time, ...). "
                             f"Got: {input_shape}")
        # child sees a batch dimension; we return (None, ...) for batch
        return (None, *input_shape[2:])

    def build(self, input_shape):
        child_input_shape = self._get_child_input_shape(input_shape)
        super().build(child_input_shape)

    def compute_output_shape(self, input_shape):
        # input_shape is (batch, time, ...)
        child_input_shape = self._get_child_input_shape(input_shape)
        child_output_shape = self.layer.compute_output_shape(child_input_shape)
        # child_output_shape is (batch, ...) where batch corresponds to (batch*time)
        # return (batch, time, ...)
        # If child_output_shape has None/unknown batch dim, keep it as-is.
        return (child_output_shape[0], input_shape[1], *child_output_shape[1:])

    def call(self, inputs, training=None, mask=None):
        """
        inputs: tensor with shape (batch, time, ...)
        The implementation uses tf.reshape to merge batch and time dims,
        calls wrapped layer once, then reshapes outputs back to (batch, time, ...).
        """
        # dynamic shapes
        input_shape = tf.shape(inputs)
        batch = input_shape[0]
        time = input_shape[1]

        # Build new shape: [batch*time] + input_shape[2:]
        tail = input_shape[2:]
        new_batch = batch * time
        new_shape = tf.concat([[new_batch], tail], axis=0)
        reshaped_inputs = tf.reshape(inputs, new_shape)

        # reshape mask if present
        reshaped_mask = None
        if mask is not None:
            mask = tf.convert_to_tensor(mask)
            mask_rank = tf.rank(mask)
            # handle mask ranks of 2 (batch,time) or higher (batch,time,...)
            def _reshape_mask():
                mshape = tf.shape(mask)
                rest = mshape[2:]
                m_new_shape = tf.concat([[new_batch], rest], axis=0)
                return tf.reshape(mask, m_new_shape)

            # If mask is symbolic, we attempt reshape (works in most common cases).
            reshaped_mask = _reshape_mask()

        # Prepare kwargs if wrapped layer accepts training/mask
        call_kwargs = {}
        if self.layer._call_has_training_arg:
            call_kwargs["training"] = training
        if self.layer._call_has_mask_arg and reshaped_mask is not None:
            call_kwargs["mask"] = reshaped_mask

        outputs = self.layer(reshaped_inputs, **call_kwargs)

        # outputs shape is (batch*time, ...)
        out_shape = tf.shape(outputs)
        # target shape: [batch, time] + out_shape[1:]
        target_shape = tf.concat([[batch, time], out_shape[1:]], axis=0)
        outputs_reshaped = tf.reshape(outputs, target_shape)
        return outputs_reshaped

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        # reshape mask => (batch*time, ...)
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
        # saving wrapped layer is handled by Wrapper.get_config
        return config


# === Backbone ===
base = build_MobileViT_v1(
    model_type="XXS",
    pretrained=False,
    include_top=False,
    num_classes=0,
    input_shape=(img_size, img_size, 3)
)

# === Video model ===
video_input = tf.keras.Input((frames, img_size, img_size, 3))

# extract per-frame features
# x = tf.keras.layers.TimeDistributed(base)(video_input)
x = QuantizableTimeDistributed(base)(video_input)
# x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
x = QuantizableTimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
# shape: (batch, frames, features)

# === Replace GRU with Quantization-Friendly Temporal Block ===
# Temporal convolution simulates sequence modeling but is quantizable
x = tf.keras.layers.Conv1D(
    filters=64,
    kernel_size=3,
    activation="relu",
    padding="same"
)(x)
x = tf.keras.layers.Conv1D(
    filters=32,
    kernel_size=3,
    activation="relu",
    padding="same"
)(x)

# Global average pooling over time
x = tf.keras.layers.GlobalAveragePooling1D()(x)

# === Classification head ===
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(video_input, output)

dummy_input = tf.random.uniform((1, frames,img_size,img_size, 3))
model(dummy_input)

MODEL_PATH = "fucktimedistributed/best_ftd_99.weights.h5"
model.load_weights(MODEL_PATH)

model.trainable = False
num_classes = model.output_shape[-1]

print("Model loaded")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("fucktimedistributed/9926_trained_quantised_ftd.tflite","wb") as f:
    f.write(tflite_model)
    
print("goat nano")