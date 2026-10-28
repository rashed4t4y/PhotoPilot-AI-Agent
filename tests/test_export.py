import csv
from io import StringIO
import unittest

from utils.export import export_csv, export_txt, with_edits
from utils.stock_readiness import assess_resolution


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.result = {
            "status": "partial", "outputs": {"caption": "Original", "description": "Text", "keywords": ["green"]},
            "image_properties": {"width": 12}, "tasks": {"caption": {"status": "completed"}},
            "warnings": ["Review"],
        }

    def test_exports_use_edits_without_mutating_original(self):
        edited = with_edits(self.result, {"caption": "Changed", "keywords": "green, tree\nleaf"})
        self.assertEqual(self.result["outputs"]["caption"], "Original")
        self.assertIn("Changed", export_txt(edited))
        self.assertIn("Changed", export_csv(edited))
        self.assertNotIn("Original", export_csv(edited))
        self.assertEqual(edited["outputs"]["keywords"], ["green", "tree", "leaf"])
        self.assertIn("caption", edited["edited_fields"])

    def test_csv_quotes_newlines_unicode_and_commas(self):
        text = 'A "green", leafy scene\nবাংলা'
        edited = with_edits(self.result, {"caption": text})
        parsed = dict(list(csv.reader(StringIO(export_csv(edited))))[1:])
        self.assertEqual(parsed["outputs.caption"], text)
        self.assertEqual(parsed["image_properties.width"], "12")
        self.assertEqual(parsed["tasks.caption.status"], "completed")

    def test_formula_injection_is_neutralized(self):
        for value in ("=1+1", "+CMD", "-1+2", "@SUM(A1)", " \t=1", "\ufeff=1", "\nhello"):
            edited = with_edits(self.result, {"caption": value})
            parsed = dict(list(csv.reader(StringIO(export_csv(edited))))[1:])
            with self.subTest(value=value):
                self.assertEqual(parsed["outputs.caption"], "'" + value)

    def test_cleared_text_is_exported_not_replaced_with_original(self):
        edited = with_edits(self.result, {"caption": "", "keywords": ""})
        self.assertEqual(edited["outputs"]["caption"], "")
        self.assertEqual(edited["outputs"]["keywords"], [])
        self.assertNotIn("Original", export_txt(edited))

    def test_quality_and_failures_are_included(self):
        self.result["outputs"]["quality"] = {"laplacian_variance": None, "heuristic_flags": ["Possibly dark"]}
        self.result["tasks"]["description"] = {"status": "failed", "message": "Unavailable"}
        text = export_txt(self.result)
        self.assertIn("Not available", text)
        self.assertIn("Possibly dark", text)
        self.assertIn("Unavailable", text)

    def test_readiness_warnings_and_disclaimer_in_both_formats(self):
        self.result["stock_readiness"] = assess_resolution(1000, 1000)
        for report in (export_txt(self.result), export_csv(self.result)):
            self.assertIn("Below the configured Shutterstock minimum", report)
            self.assertIn("minimum_megapixels", report)
            self.assertIn("do not guarantee platform acceptance", report)

    def test_optional_metadata_can_be_missing(self):
        for result in ({}, {"outputs": {}}, {"stock_readiness": None}):
            self.assertIn("do not guarantee", export_txt(result))
            self.assertIn("do not guarantee", export_csv(result))

    def test_edited_keyword_count_normalizes_duplicates_without_changing_history(self):
        self.result["keyword_review"] = {"actual_unique_count": 1, "requested_count": 5}
        edited = with_edits(self.result, {"keywords": "green, GREEN,  leaf , ,"})
        self.assertEqual(edited["outputs"]["keywords"], ["green", "leaf"])
        self.assertEqual(edited["current_keyword_count"], 2)
        self.assertEqual(edited["keyword_review"]["actual_unique_count"], 1)
        self.assertEqual(self.result["outputs"]["keywords"], ["green"])


if __name__ == "__main__":
    unittest.main()
