"""Resolution boundary tests use generated images only."""
from io import BytesIO
import unittest

from PIL import Image

from utils.stock_readiness import assess_image_readiness, assess_resolution


class ReadinessTests(unittest.TestCase):
    def test_below_exactly_at_and_above_four_mp(self):
        for width, expected in ((1999, "below_minimum"), (2000, "passed"), (2001, "passed")):
            with self.subTest(width=width):
                stream = BytesIO()
                with Image.new("L", (width, 2000), 100) as image:
                    image.save(stream, "PNG")
                result = assess_image_readiness(stream)
                self.assertEqual(result["width"], width)
                self.assertEqual(result["height"], 2000)
                self.assertEqual(result["megapixels"], width * 2000 / 1_000_000)
                self.assertEqual(result["resolution_result"], expected)
                self.assertIn("do not guarantee", result["disclaimer"])

    def test_threshold_is_configurable_and_not_rounded(self):
        self.assertEqual(assess_resolution(1999, 2000, 3.998)["resolution_result"], "passed")
        self.assertEqual(assess_resolution(1999, 2000, 3.998001)["resolution_result"], "below_minimum")

    def test_missing_or_corrupt_data_never_passes(self):
        for data in (None, b"", b"bad image"):
            with self.subTest(data=data):
                result = assess_image_readiness(data)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["resolution_result"], "unavailable")
                self.assertTrue(result["message"])

    def test_invalid_dimensions_and_limits_are_explained(self):
        for width, height in ((0, 2), (-1, 2), (None, 2), (True, 2), (2.5, 2)):
            self.assertEqual(assess_resolution(width, height)["resolution_result"], "unavailable")
        for limit in (0, -1, float("nan"), float("inf"), True, "4"):
            self.assertEqual(assess_resolution(2000, 2000, limit)["status"], "failed")

    def test_pass_only_establishes_resolution(self):
        result = assess_resolution(3000, 2000)
        self.assertIn("Resolution check passed", result["message"])
        self.assertIn("No other submission requirements", result["message"])
        self.assertEqual(result["megapixels"], 6.0)


if __name__ == "__main__":
    unittest.main()
