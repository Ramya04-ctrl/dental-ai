"""
Unit tests for the parts that do NOT need a trained model.
Run:   python -m unittest discover -s tests -v
"""
import io
import os
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

os.environ["DENTAL_WORKDIR"] = tempfile.mkdtemp()      # tests never touch real data

import config                                                    # noqa: E402
from database import db                                          # noqa: E402
from preprocessing.image_validation import validate_image        # noqa: E402
from preprocessing.preprocess import preprocess_for_model        # noqa: E402
from recommendation_engine.engine import build_recommendation    # noqa: E402
from reports.pdf_report import build_pdf                         # noqa: E402


def jpeg_bytes(img, fmt=".jpg"):
    ok, buf = cv2.imencode(fmt, img)
    return buf.tobytes()


def sharp_image(size=300, mean=128):
    rng = np.random.default_rng(1)
    return np.clip(rng.normal(mean, 40, (size, size, 3)), 0, 255).astype(np.uint8)


class TestValidation(unittest.TestCase):
    def test_good_image_passes(self):
        self.assertTrue(validate_image(jpeg_bytes(sharp_image()), "a.jpg")["ok"])

    def test_png_renamed_to_jpg_is_rejected(self):
        self.assertFalse(validate_image(jpeg_bytes(sharp_image(), ".png"), "fake.jpg")["ok"])

    def test_real_png_accepted(self):
        self.assertTrue(validate_image(jpeg_bytes(sharp_image(), ".png"), "a.png")["ok"])

    def test_unsupported_extension_rejected(self):
        self.assertFalse(validate_image(jpeg_bytes(sharp_image(), ".bmp"), "a.bmp")["ok"])

    def test_corrupted_file_rejected(self):
        self.assertFalse(validate_image(b"not an image at all", "a.jpg")["ok"])

    def test_blurry_image_rejected(self):
        blurry = cv2.GaussianBlur(sharp_image(), (51, 51), 20)
        self.assertFalse(validate_image(jpeg_bytes(blurry), "a.jpg")["ok"])

    def test_dark_image_rejected(self):
        self.assertFalse(validate_image(jpeg_bytes(sharp_image(mean=5) // 8), "a.jpg")["ok"])

    def test_tiny_image_rejected(self):
        self.assertFalse(validate_image(jpeg_bytes(sharp_image(size=100)), "a.jpg")["ok"])

    def test_preprocess_output_shape(self):
        out = preprocess_for_model(sharp_image(400))
        self.assertEqual(out.shape, (224, 224, 3))


class TestRecommendation(unittest.TestCase):
    base = {"pain_level": 2, "swelling": 0, "fever": False, "breathing_trouble": False}

    def test_healthy_low_risk_is_routine(self):
        r = build_recommendation("healthy", 0.9, "low", self.base)
        self.assertEqual(r["urgency_level"], "routine")

    def test_red_flag_is_always_urgent(self):
        r = build_recommendation("healthy", 0.9, "low", {**self.base, "fever": True})
        self.assertEqual(r["urgency_level"], "urgent")
        self.assertIsNotNone(r["urgent_message"])

    def test_never_prescribes_surgery(self):
        r = build_recommendation("caries", 0.9, "high", self.base)
        text = " ".join(r["pathways"]).lower()
        self.assertIn("dentist", text)
        self.assertNotIn("you need surgery", text)

    def test_unknown_class_uses_generic_text(self):
        self.assertTrue(build_recommendation("new_class", 0.8, "low", self.base)["pathways"])


class TestDatabaseAndPdf(unittest.TestCase):
    def test_save_and_read_history(self):
        db.save_screening({"ref_id": "T1", "age_group": "18-40", "condition": "caries", "confidence": 0.8,
                           "risk": "low", "urgency": "routine", "blur_score": 90.0, "brightness": 120.0})
        self.assertIn("T1", db.get_history()["ref_id"].tolist())

    def test_pdf_is_created(self):
        rec = build_recommendation("caries", 0.8, "low", {"pain_level": 1})
        img = sharp_image(224)
        pdf = build_pdf({"ref_id": "T1", "date": "today", "quality": {"blur_score": 90}, "rec": rec,
                         "risk_method": "test"}, img, img)
        self.assertTrue(pdf.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
