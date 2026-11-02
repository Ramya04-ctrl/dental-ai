"""
gradcam.py  -  shows WHERE in the image the model looked.

Grad-CAM makes a "heat map": red/yellow = areas that influenced the decision most.
It is an explanation aid. It does NOT prove the model found the real problem.
"""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


def find_last_conv_layer(model):
    """The last layer whose output is a 4D feature map (batch, height, width, channels)."""
    for layer in reversed(model.layers):
        shape = getattr(layer, "output", None)
        shape = getattr(shape, "shape", None)
        if shape is not None and len(shape) == 4:
            return layer
    raise ValueError("No convolution layer found for Grad-CAM")


def make_gradcam_heatmap(model, img_batch, class_index=None):
    """img_batch: shape (1, 224, 224, 3). Returns a heat map with values 0-1."""
    conv_layer = find_last_conv_layer(model)
    grad_model = keras.Model(model.inputs, [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model([img_batch], training=False)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        score = preds[:, class_index]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(rgb, heatmap, alpha=0.4):
    """Paint the heat map on top of the picture. rgb is uint8 (H, W, 3)."""
    h, w = rgb.shape[:2]
    heat = cv2.resize(heatmap, (w, h))
    colored = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)[:, :, ::-1]  # BGR -> RGB
    return cv2.addWeighted(rgb, 1 - alpha, colored, alpha, 0)
