"""
models.py  -  the three deep-learning models we compare.

  A. custom_cnn     : a small CNN built from scratch (our baseline)
  B. mobilenet      : MobileNetV2 (pretrained on ImageNet, small and fast)
  C. efficientnet   : EfficientNetB0 (pretrained on ImageNet, accurate and efficient)

"Transfer learning" = we start from a network that already learned to see
edges, shapes and textures from 1 million+ photos, and we only teach it our
dental classes. This works much better when you have a small dataset.

All models take a 224x224 RGB image with pixel values 0-255 and do their own
scaling inside, so training and prediction always match.
"""
from tensorflow import keras
from tensorflow.keras import layers

import config

MODEL_NAMES = ["custom_cnn", "mobilenet", "efficientnet"]


def _augmentation():
    """Random changes to training images (only active while training).
    No vertical flip and no big rotations: teeth photos have a natural direction."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.1, value_range=(0, 255)),
    ], name="augmentation")


def _head(x, num_classes):
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    return layers.Dense(num_classes, activation="softmax", name="predictions")(x)


def build_custom_cnn(num_classes, pretrained=False):
    inputs = keras.Input(shape=(*config.IMG_SIZE, 3), name="image")
    x = _augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)                      # pixels 0-255 -> 0-1
    for filters in (32, 64, 128, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    outputs = _head(x, num_classes)
    return keras.Model(inputs, outputs, name="custom_cnn"), None


def _build_transfer(name, app_class, rescale, num_classes, pretrained):
    inputs = keras.Input(shape=(*config.IMG_SIZE, 3), name="image")
    x = _augmentation()(inputs)
    if rescale is not None:
        x = rescale(x)
    base = app_class(include_top=False,
                     weights="imagenet" if pretrained else None,
                     input_tensor=x)
    outputs = _head(base.output, num_classes)
    return keras.Model(inputs, outputs, name=name), base


def build_mobilenet(num_classes, pretrained=True):
    # MobileNetV2 expects pixels in the range -1 to +1
    return _build_transfer("mobilenet", keras.applications.MobileNetV2,
                           layers.Rescaling(1.0 / 127.5, offset=-1), num_classes, pretrained)


def build_efficientnet(num_classes, pretrained=True):
    # EfficientNetB0 already contains its own scaling layer (expects 0-255)
    return _build_transfer("efficientnet", keras.applications.EfficientNetB0,
                           None, num_classes, pretrained)


BUILDERS = {
    "custom_cnn": build_custom_cnn,
    "mobilenet": build_mobilenet,
    "efficientnet": build_efficientnet,
}


def build_model(name, num_classes, pretrained):
    """Returns (model, base). 'base' is the pretrained part (None for the custom CNN)."""
    if name not in BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}")
    return BUILDERS[name](num_classes, pretrained)


def unfreeze_top_layers(base, n_layers):
    """Fine-tuning: allow the last n layers of the pretrained part to learn.
    BatchNormalization layers stay frozen (this keeps fine-tuning stable)."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers[-n_layers:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
