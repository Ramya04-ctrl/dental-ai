"""
preprocess.py  -  turns any valid photo into the exact format the model expects.

The SAME function is used when we prepare the training data AND when a patient
uploads a photo. This is very important: if training and prediction use
different preprocessing, the model gives wrong answers.

Steps:
  1. Resize to 224 x 224 (INTER_AREA is the best method for shrinking)
  2. Light denoising with OpenCV (removes camera grain)

We do NOT change the colours (no histogram equalisation) because colour is
important for gingivitis (red gums) and discoloration.

Normalisation (pixel scaling) is done INSIDE the model (see training/models.py),
so here pixels stay as normal 0-255 numbers.
"""
import cv2
import numpy as np

import config


def preprocess_for_model(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 image (any size) -> RGB uint8 image of size 224 x 224."""
    w, h = config.IMG_SIZE[1], config.IMG_SIZE[0]
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    denoised = cv2.fastNlMeansDenoisingColored(resized, None, 3, 3, 7, 21)
    return denoised


def to_model_input(rgb224: np.ndarray) -> np.ndarray:
    """Add the 'batch' dimension -> shape (1, 224, 224, 3), float32."""
    return np.expand_dims(rgb224.astype("float32"), axis=0)
