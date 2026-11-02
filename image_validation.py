"""
image_validation.py  -  checks that an uploaded file is a good JPG / PNG photo.

We check:
  1. the file name ends with .jpg / .jpeg / .png
  2. the file is not empty and not too big
  3. the file CONTENT is really what the name says (someone can rename a PNG to .jpg!)
  4. the picture is not corrupted
  5. the picture is big enough
  6. the picture is not blurry      (variance of the Laplacian)
  7. the picture is not too dark / too bright

IMPORTANT: this code cannot check that a *tooth* is visible. See the README
("Limitations") for how to improve that later.
"""
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

import config


def load_rgb_image(file_bytes: bytes) -> np.ndarray:
    """Read image bytes -> RGB numpy array. Also fixes phone-photo rotation (EXIF)."""
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.array(img)


def blur_score(rgb: np.ndarray) -> float:
    """Higher = sharper. We shrink big images first so the score is fair."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    scale = 512.0 / max(h, w)
    if scale < 1:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(rgb: np.ndarray) -> float:
    """Average gray level, 0 (black) to 255 (white)."""
    return float(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).mean())


def validate_image(file_bytes: bytes, filename: str) -> dict:
    """
    Returns a dict:
        ok       : True / False
        reasons  : list of simple messages (why it was rejected)
        metrics  : width, height, blur, brightness, size_mb
        image    : the RGB array (only if the file could be read)
    """
    result = {"ok": False, "reasons": [], "metrics": {}, "image": None}
    reasons = result["reasons"]

    # 1. file extension
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        reasons.append("Only JPG, JPEG or PNG files are accepted.")

    # 2. size
    size_mb = len(file_bytes) / (1024 * 1024)
    result["metrics"]["size_mb"] = round(size_mb, 2)
    if len(file_bytes) == 0:
        reasons.append("The file is empty.")
        return result
    if size_mb > config.MAX_FILE_MB:
        reasons.append(f"The file is too big ({size_mb:.1f} MB). Maximum is {config.MAX_FILE_MB} MB.")

    # 3 + 4. real content and corruption
    try:
        probe = Image.open(io.BytesIO(file_bytes))
        real_format = probe.format
        probe.verify()
    except Exception:
        reasons.append("The file is corrupted or is not an image.")
        return result
    if real_format not in config.ALLOWED_FORMATS.values():
        reasons.append(f"The file content is {real_format}, which is not supported. Use JPG or PNG.")
        return result
    expected = config.ALLOWED_FORMATS.get(ext)
    if expected and real_format != expected:
        reasons.append(f"The file name says {expected} but the content is {real_format}.")
        return result

    try:
        rgb = load_rgb_image(file_bytes)
    except Exception:
        reasons.append("The image could not be read.")
        return result

    result["image"] = rgb
    h, w = rgb.shape[:2]
    blur = blur_score(rgb)
    bright = brightness_score(rgb)
    result["metrics"].update({"width": w, "height": h,
                              "blur_score": round(blur, 1),
                              "brightness": round(bright, 1)})

    # 5. resolution
    if min(h, w) < config.MIN_SIDE_PIXELS:
        reasons.append(f"The image is too small ({w}x{h}). Use at least "
                       f"{config.MIN_SIDE_PIXELS}x{config.MIN_SIDE_PIXELS} pixels.")
    # 6. blur
    if blur < config.BLUR_THRESHOLD:
        reasons.append("The image is too blurry.")
    # 7. brightness
    if bright < config.MIN_BRIGHTNESS:
        reasons.append("The image is too dark.")
    elif bright > config.MAX_BRIGHTNESS:
        reasons.append("The image is too bright.")

    result["ok"] = len(reasons) == 0
    if not result["ok"]:
        reasons.append("Unable to analyze this image. Please upload a clear dental photograph.")
    return result
