import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Conv2D, Dropout, MaxPool2D, Conv2DTranspose,
    concatenate, Add, Multiply, BatchNormalization
)
from tensorflow.keras.models import Sequential, Model
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Encoder Block
# -------------------------------
class EncoderBlock(Layer):
    def __init__(self, filters, rate=0.0, pooling=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.rate = rate
        self.pooling = pooling

        self.convs = Sequential([
            Conv2D(filters, 3, padding='same', activation='relu', kernel_initializer='he_normal'),
            Dropout(rate),
            Conv2D(filters, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        ])
        self.pool = MaxPool2D() if pooling else None

    def call(self, x):
        x = self.convs(x)
        y = self.pool(x) if self.pool else x
        return (y, x) if self.pool else x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            "rate": self.rate,
            "pooling": self.pooling
        }

# -------------------------------
# Decoder Block
# -------------------------------
class DecoderBlock(Layer):
    def __init__(self, filters, rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.rate = rate

        self.up = Conv2DTranspose(filters, 2, strides=2, padding="same")
        self.net = EncoderBlock(filters, rate, pooling=False)

    def call(self, inputs):
        x, skip_x = inputs
        x = self.up(x)
        x = concatenate([x, skip_x], axis=-1)
        return self.net(x)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            "rate": self.rate
        }

# -------------------------------
# Attention Gate
# -------------------------------
class AttentionGate(Layer):
    def __init__(self, filters, use_bn=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.use_bn = use_bn

        self.g_conv = Conv2D(filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")
        self.x_conv = Conv2D(filters, 3, strides=2, padding="same", activation="relu", kernel_initializer="he_normal")
        self.psi = Conv2D(1, 1, padding="same", activation="sigmoid")
        self.up = Conv2DTranspose(1, 2, strides=2, padding="same")
        self.bn = BatchNormalization() if use_bn else None

    def call(self, inputs):
        g, x = inputs
        g_proj = self.g_conv(g)
        x_proj = self.x_conv(x)

        attn = Add()([g_proj, x_proj])
        attn = self.psi(attn)
        attn = self.up(attn)

        out = Multiply()([x, attn])
        if self.bn:
            out = self.bn(out)

        return out

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "filters": self.filters, "use_bn": self.use_bn}

# -------------------------------
# Training History Plot
# -------------------------------
def plot_training(hist, metrics=("accuracy",)):
    """
    Plot training & validation history (loss + given metrics).
    Marks best epoch for both.
    """
    epochs = range(1, len(hist.history["loss"]) + 1)

    plt.figure(figsize=(20, 8))
    plt.style.use("fivethirtyeight")

    # --- Loss Plot ---
    val_loss = hist.history["val_loss"]
    best_loss_idx = np.argmin(val_loss)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist.history["loss"], "r", label="Training Loss")
    plt.plot(epochs, val_loss, "g", label="Validation Loss")
    plt.scatter(best_loss_idx+1, val_loss[best_loss_idx], s=150, c="blue",
                label=f"Best epoch = {best_loss_idx+1}")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend()

    # --- Metric Plot ---
    metric = metrics[0]
    val_metric = hist.history[f"val_{metric}"]
    best_metric_idx = np.argmax(val_metric)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist.history[metric], "r", label=f"Training {metric.title()}")
    plt.plot(epochs, val_metric, "g", label=f"Validation {metric.title()}")
    plt.scatter(best_metric_idx+1, val_metric[best_metric_idx], s=150, c="blue",
                label=f"Best epoch = {best_metric_idx+1}")
    plt.title(f"Training & Validation {metric.title()}")
    plt.xlabel("Epochs"); plt.ylabel(metric.title()); plt.legend()

    plt.tight_layout()
    plt.show()

# -------------------------------
# Build Model Helper
# -------------------------------
def build_model(input_shape=(128, 128, 3), num_classes=1):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    p1, s1 = EncoderBlock(64, rate=0.1)(inputs)
    p2, s2 = EncoderBlock(128, rate=0.1)(p1)
    p3, s3 = EncoderBlock(256, rate=0.2)(p2)
    p4, s4 = EncoderBlock(512, rate=0.2)(p3)

    # Bridge
    b1 = Conv2D(1024, 3, padding="same", activation="relu")(p4)
    b1 = Conv2D(1024, 3, padding="same", activation="relu")(b1)

    # Decoder with Attention
    s4 = AttentionGate(512)([b1, s4])
    d1 = DecoderBlock(512)([b1, s4])

    s3 = AttentionGate(256)([d1, s3])
    d2 = DecoderBlock(256)([d1, s3])

    s2 = AttentionGate(128)([d2, s2])
    d3 = DecoderBlock(128)([d2, s2])

    s1 = AttentionGate(64)([d3, s1])
    d4 = DecoderBlock(64)([d3, s1])

    # Output
    outputs = Conv2D(num_classes, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="Attention_UNet")
    return model
