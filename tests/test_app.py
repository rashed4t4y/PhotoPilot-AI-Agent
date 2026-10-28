"""Streamlit AppTest integration: real UI/agent with mocked upload and Gemini.

This is headless widget testing, not a browser automation test.
"""
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from utils import gemini

APP = Path(__file__).resolve().parents[1] / "app.py"


def generated_upload(colour="green"):
    stream = BytesIO()
    with Image.new("RGB", (12, 8), colour) as image:
        image.save(stream, format="PNG")
    stream.seek(0)
    return stream


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.models.generate_content.return_value = SimpleNamespace(
            text=json.dumps({"caption": "Green surface."}), candidates=[],
        )
        guards = [
            patch("utils.gemini.load_dotenv", side_effect=AssertionError("No real secrets")),
            patch("utils.gemini.genai.Client", side_effect=AssertionError("No live client")),
            patch("utils.gemini._create_client", return_value=self.client),
            patch("streamlit.file_uploader", return_value=generated_upload()),
        ]
        handles = [guard.start() for guard in guards]
        self.factory, self.upload = handles[2:]
        for guard in guards:
            self.addCleanup(guard.stop)

    def app(self):
        at = AppTest.from_file(str(APP), default_timeout=20).run()
        self.assertEqual(len(at.exception), 0)
        return at

    def run_caption(self):
        at = self.app()
        at.text_area(key="pp_goal").set_value("Caption only").run()
        next(button for button in at.button if button.label == "Run Agent").click().run()
        self.assertEqual(len(at.exception), 0)
        return at

    def test_empty_state_starts_without_configuration(self):
        self.upload.return_value = None
        at = self.app()
        self.factory.assert_not_called()
        self.assertTrue(next(button for button in at.button if button.label == "Run Agent").disabled)

    def test_caption_run_edit_rerun_and_export_without_new_call(self):
        at = self.run_caption()
        self.assertEqual(at.session_state["pp_result"]["status"], "completed")
        at.text_area(key="pp_edit_caption").set_value('Edited, "caption"').run()
        self.assertIn('Edited, "caption"', at.session_state["pp_download_txt"])
        self.assertIn('Edited, ""caption""', at.session_state["pp_download_csv"])
        at.run()
        self.assertEqual(at.text_area(key="pp_edit_caption").value, 'Edited, "caption"')
        self.client.models.generate_content.assert_called_once()

    def test_local_quality_works_without_key(self):
        self.factory.side_effect = AssertionError("No key/client needed")
        at = self.app()
        at.text_area(key="pp_goal").set_value("Check quality").run()
        next(button for button in at.button if button.label == "Run Agent").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["pp_result"]["status"], "completed")
        self.assertIn("laplacian_variance", at.session_state["pp_download_txt"])
        self.factory.assert_not_called()

    def test_missing_key_is_message_not_ui_crash(self):
        self.factory.side_effect = gemini.GeminiError("GEMINI_API_KEY is not configured.")
        at = self.run_caption()
        self.assertEqual(at.session_state["pp_result"]["status"], "failed")
        self.assertIn("not configured", at.session_state["pp_download_txt"])

    def test_corrupt_upload_disables_run(self):
        self.upload.return_value = BytesIO(b"corrupt")
        at = self.app()
        self.assertTrue(at.error)
        self.assertTrue(next(button for button in at.button if button.label == "Run Agent").disabled)
        self.factory.assert_not_called()

    def test_switching_image_clears_stale_outputs(self):
        at = self.run_caption()
        self.upload.return_value = generated_upload("red")
        at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertNotIn("pp_result", at.session_state)
        self.assertNotIn("pp_download_txt", at.session_state)

    def test_clear_results_does_not_rerun_generation(self):
        at = self.run_caption()
        next(button for button in at.button if button.label.startswith("Clear results")).click().run()
        self.assertNotIn("pp_result", at.session_state)
        self.client.models.generate_content.assert_called_once()

    def test_preset_changes_goal_but_does_not_run(self):
        at = self.app()
        at.selectbox(key="pp_preset").select("Complete stock-photo package").run()
        self.assertIn("Shutterstock", at.text_area(key="pp_goal").value)
        self.factory.assert_not_called()

    def test_original_caption_description_flow_is_connected(self):
        self.client.models.generate_content.return_value = SimpleNamespace(
            text=json.dumps({"caption": "Green.", "description": "visible " * 80}), candidates=[],
        )
        at = self.app()
        next(button for button in at.button if button.label == "Run Agent").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["pp_result"]["status"], "completed")
        self.assertEqual(at.text_area(key="pp_edit_caption").value, "Green.")
        self.assertIn("visible", at.text_area(key="pp_edit_description").value)

    def test_full_package_ui_preserves_partial_results(self):
        self.client.models.generate_content.return_value = SimpleNamespace(
            text=json.dumps({"analysis": "A green field.", "caption": "Green.",
                             "description": None, "keywords": ["green"]}), candidates=[],
        )
        at = self.app()
        at.selectbox(key="pp_preset").select("Complete stock-photo package").run()
        next(button for button in at.button if button.label == "Run Agent").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["pp_result"]["status"], "partial")
        self.assertEqual(at.text_area(key="pp_edit_caption").value, "Green.")
        self.assertIn("quality", at.session_state["pp_result"]["outputs"])
        self.assertIn("failed", at.session_state["pp_download_txt"])
        self.assertEqual(self.client.models.generate_content.call_count, 2)
        self.assertTrue(any("Below the configured Shutterstock minimum" in warning.value for warning in at.warning))
        self.assertIn("stock_readiness", at.session_state["pp_download_txt"])

    def test_readiness_warning_and_configured_threshold(self):
        at = self.app()
        at.selectbox(key="pp_preset").select("Stock readiness (local, no API)").run()
        at.number_input[0].set_value(5.0).run()
        next(button for button in at.button if button.label == "Run Agent").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["pp_result"]["stock_readiness"]["minimum_megapixels"], 5.0)
        self.assertTrue(any("Below the configured" in warning.value for warning in at.warning))
        self.factory.assert_not_called()

    def test_changed_goal_and_settings_keep_edits_and_warn(self):
        at = self.run_caption()
        at.text_area(key="pp_edit_caption").set_value("My edit").run()
        at.text_area(key="pp_goal").set_value("Write a description").run()
        self.assertTrue(any("previous run" in warning.value for warning in at.warning))
        self.assertEqual(at.text_area(key="pp_edit_caption").value, "My edit")
        at.number_input[0].set_value(6.0).run()
        self.assertIn("My edit", at.session_state["pp_download_txt"])
        self.client.models.generate_content.assert_called_once()

    def test_edited_keyword_count_and_reason_displayed(self):
        self.client.models.generate_content.side_effect = [
            SimpleNamespace(text='{"keywords":["green"]}', candidates=[]),
            SimpleNamespace(text='{"keywords":[]}', candidates=[]),
        ]
        at = self.app()
        at.text_area(key="pp_goal").set_value("Generate 5 keywords").run()
        next(button for button in at.button if button.label == "Run Agent").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any("Requested 5; returned 1" in item.value for item in at.text))
        at.text_area(key="pp_edit_keywords").set_value("green, GREEN, leaf, ").run()
        self.assertTrue(any("Current edited unique keywords: 2" in item.value for item in at.caption))
        self.assertEqual(self.client.models.generate_content.call_count, 2)


if __name__ == "__main__":
    unittest.main()
