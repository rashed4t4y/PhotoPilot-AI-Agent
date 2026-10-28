"""Offline tests using generated images only; no app or Gemini imports."""

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageFilter

from utils.image_tools import ImageValidationError, check_image_quality, validate_image


def encoded(image, file_format="PNG", **options):
    stream = BytesIO()
    image.save(stream, format=file_format, **options)
    return stream.getvalue()


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.data = encoded(Image.new("RGB", (16, 12), (40, 80, 120)))

    def test_supported_formats_and_independent_decoded_image(self):
        for fmt in ("JPEG", "PNG"):
            with self.subTest(fmt=fmt):
                result = validate_image(encoded(Image.new("RGB", (8, 6)), fmt))
                self.addCleanup(result.image.close)
                self.assertEqual(result.file_format, fmt)
                self.assertEqual(result.image.size, (8, 6))
                self.assertEqual(result.image.mode, "RGB")
                self.assertEqual(result.image.getpixel((0, 0)), (0, 0, 0))

    def test_empty_and_non_image_rejected(self):
        for data in (b"", b"not a photograph"):
            with self.subTest(data=data), self.assertRaises(ImageValidationError):
                validate_image(data)

    def test_actual_format_not_filename(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fake.jpg"
            data = encoded(Image.new("RGB", (8, 8)), "BMP")
            path.write_bytes(data)
            with self.assertRaisesRegex(ImageValidationError, "Only JPEG and PNG"):
                validate_image(path)
            self.assertEqual(path.read_bytes(), data)

    def test_valid_temporary_file_is_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "generated.png"
            path.write_bytes(self.data)
            result = check_image_quality(path)
            self.assertEqual(result["file_size_bytes"], len(self.data))
            self.assertEqual(path.read_bytes(), self.data)

    def test_size_limit_and_boundary(self):
        with self.assertRaisesRegex(ImageValidationError, "file limit"):
            validate_image(self.data, max_file_bytes=len(self.data) - 1)
        result = validate_image(self.data, max_file_bytes=len(self.data))
        result.image.close()

    def test_pixel_limit_before_decode(self):
        with patch.object(Image.Image, "load", side_effect=AssertionError("must not decode")):
            with self.assertRaisesRegex(ImageValidationError, "pixel limit"):
                validate_image(self.data, max_pixels=191)
        result = validate_image(self.data, max_pixels=192)
        result.image.close()

    def test_stream_position_preserved_on_success_and_failure(self):
        stream = BytesIO(self.data)
        stream.seek(5)
        result = validate_image(stream)
        result.image.close()
        self.assertEqual(stream.tell(), 5)
        with self.assertRaises(ImageValidationError):
            validate_image(stream, max_file_bytes=10)
        self.assertEqual(stream.tell(), 5)
        self.assertFalse(stream.closed)

    def test_corrupt_png_and_truncated_jpeg(self):
        corrupt = bytearray(self.data)
        corrupt[29] ^= 255  # damage the IHDR checksum
        jpeg = encoded(Image.new("RGB", (32, 32)), "JPEG")
        for data in (bytes(corrupt), jpeg[:-20], self.data[:40]):
            with self.subTest(size=len(data)), self.assertRaises(ImageValidationError):
                validate_image(data)

    def test_exif_orientation(self):
        exif = Image.Exif()
        exif[274] = 6  # rotate 90 degrees clockwise for display
        data = encoded(Image.new("RGB", (8, 4)), "JPEG", exif=exif)
        result = check_image_quality(data)
        self.assertEqual((result["width"], result["height"]), (4, 8))

    def test_transparency_is_composited_on_white(self):
        data = encoded(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))
        self.assertEqual(check_image_quality(data)["brightness_mean"], 255)

    def test_palette_transparency(self):
        image = Image.new("P", (8, 8), 0)
        data = encoded(image, transparency=0)
        self.assertEqual(check_image_quality(data)["brightness_mean"], 255)

    def test_animated_png_rejected(self):
        first = Image.new("RGB", (8, 8), "red")
        second = Image.new("RGB", (8, 8), "blue")
        data = encoded(first, save_all=True, append_images=[second], duration=100)
        with self.assertRaisesRegex(ImageValidationError, "Animated"):
            validate_image(data)

    def test_pillow_bomb_protection_retained(self):
        for threshold in (100, 50):
            with self.subTest(threshold=threshold), patch.object(Image, "MAX_IMAGE_PIXELS", threshold):
                with self.assertRaisesRegex(ImageValidationError, "safe decoded-size"):
                    validate_image(self.data)

    def test_invalid_limits(self):
        for name in ("max_file_bytes", "max_pixels"):
            for value in (0, -1, True, 1.5):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    validate_image(self.data, **{name: value})


class QualityTests(unittest.TestCase):
    def test_uniform_image_exact_measurements(self):
        data = encoded(Image.new("L", (20, 10), 128))
        report = check_image_quality(data)
        self.assertEqual((report["width"], report["height"]), (20, 10))
        self.assertEqual(report["megapixels"], 0.0002)
        self.assertEqual(report["file_size_bytes"], len(data))
        self.assertEqual(report["brightness_mean"], 128)
        self.assertEqual(report["contrast_stddev"], 0)
        self.assertEqual(report["laplacian_variance"], 0)
        self.assertIn("do not guarantee", report["limitations"])

    def test_checkerboard_known_laplacian_and_contrast(self):
        pixels = ((np.indices((10, 10)).sum(axis=0) % 2) * 255).astype(np.uint8)
        report = check_image_quality(encoded(Image.fromarray(pixels)))
        self.assertEqual(report["brightness_mean"], 127.5)
        self.assertEqual(report["contrast_stddev"], 127.5)
        self.assertEqual(report["laplacian_variance"], 1020 ** 2)

    def test_gaussian_blur_reduces_metric(self):
        pixels = (((np.indices((64, 64)) // 4).sum(axis=0) % 2) * 255).astype(np.uint8)
        sharp = Image.fromarray(pixels)
        blurred = sharp.filter(ImageFilter.GaussianBlur(2))
        sharp_score = check_image_quality(encoded(sharp))["laplacian_variance"]
        blurred_score = check_image_quality(encoded(blurred))["laplacian_variance"]
        self.assertGreater(sharp_score, blurred_score * 10)

    def test_tiny_image_has_no_blur_score(self):
        report = check_image_quality(encoded(Image.new("RGB", (2, 1))))
        self.assertIsNone(report["laplacian_variance"])

    def test_brightness_flags_are_separate_from_measurements(self):
        for value, word in ((0, "dark"), (255, "bright")):
            report = check_image_quality(encoded(Image.new("L", (8, 8), value)))
            self.assertEqual(report["brightness_mean"], value)
            self.assertTrue(any(word in flag for flag in report["heuristic_flags"]))


if __name__ == "__main__":
    unittest.main()
